from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import zipfile

import numpy as np


@dataclass(frozen=True)
class DelayedRenewalCageParams:
    """Parameters for the delayed renewal cage model.

    All variances are one-dimensional. The local cage contribution is Gaussian with
    variance cage_variance * (1 - exp(-t/cage_tau)). Cage renewal events are modeled
    as an inhomogeneous Poisson process with rate

        renewal_rate * (1 - exp(-t/renewal_delay))**2.

    Each renewal contributes an independent Gaussian cage-center jump with variance
    jump_variance.
    """

    cage_variance: float
    cage_tau: float
    jump_variance: float
    renewal_rate: float
    renewal_delay: float


@dataclass(frozen=True)
class GammaExchangeParams:
    """Finite-exchange gamma heterogeneity for the renewal count.

    ``shape`` controls the instantaneous mobility dispersion. Smaller values
    mean broader dynamic heterogeneity. ``exchange_renewal_count`` is the
    renewal-count scale over which independent mobility environments are sampled,
    so the effective gamma shape grows as the trajectory self-averages.
    """

    shape: float
    exchange_renewal_count: float


@dataclass(frozen=True)
class TemperatureLawParams:
    """Dimensionless Arrhenius-like temperature law for the renewal cage model.

    Temperatures are reduced units. Cooling below ``reference_temperature`` makes
    ``delta = 1/T - 1/T_ref`` positive. The law keeps the model phenomenological:
    renewal events slow down, the delayed onset grows, cages stiffen, and q/A can
    increase with cooling.
    """

    reference_temperature: float
    cage_variance_ref: float
    cage_tau_ref: float
    jump_to_cage_ref: float
    renewal_rate_ref: float
    renewal_delay_ref: float
    rate_activation: float
    delay_activation: float
    cage_stiffening: float = 0.0
    jump_to_cage_growth: float = 0.0
    cage_tau_activation: float = 0.0


@dataclass(frozen=True)
class ConfigurationalEntropyParams:
    """Minimal Kauzmann/Adam-Gibbs configurational entropy law.

    ``entropy_ref`` is the configurational entropy per rearranging unit at
    ``reference_temperature``. The linear extrapolation reaches zero at
    ``kauzmann_temperature``.
    """

    reference_temperature: float
    entropy_ref: float
    kauzmann_temperature: float


@dataclass(frozen=True)
class MCTBetaParams:
    """Effective MCT beta-relaxation window around the cage plateau.

    The envelope is not a microscopic mode-coupling memory kernel. It is a
    diagnostic closure for the two beta-window power laws:
    ``phi-f_c ~ t^{-a}`` before the plateau and
    ``f_c-phi ~ t^b`` in the von Schweidler departure.
    """

    plateau: float
    critical_amplitude: float
    von_schweidler_amplitude: float
    critical_exponent: float
    von_schweidler_exponent: float
    beta_time: float


@dataclass(frozen=True)
class ActivatedBarrierParams:
    """Activated-barrier interpretation of the temperature law."""

    reference_temperature: float
    cage_variance_ref: float
    cage_tau_ref: float
    jump_to_cage_ref: float
    renewal_rate_ref: float
    renewal_delay_ref: float
    renewal_rate_barrier: float
    delay_onset_barrier: float
    cage_stiffening_barrier: float = 0.0
    jump_to_cage_barrier: float = 0.0
    cage_tau_barrier: float = 0.0


@dataclass(frozen=True)
class FacilitatedExchangeLawParams:
    """Activated facilitation law for finite-exchange heterogeneity.

    Cooling can broaden the instantaneous mobility distribution and slow the
    exchange between mobility environments. The former decreases ``shape``;
    the latter increases ``exchange_renewal_count``. Their sum controls the
    growth of the finite-exchange ratio ``c=R_x/kappa_0``.
    """

    reference_temperature: float
    shape_ref: float
    exchange_renewal_count_ref: float
    shape_broadening_barrier: float
    exchange_slowing_barrier: float


@dataclass(frozen=True)
class PersistenceExchangeParams:
    """Persistence/exchange renewal cage parameters.

    The first cage escape has mean ``persistence_mean``. After the first escape,
    subsequent cage exchanges have mean ``exchange_mean``. This makes the
    structural relaxation clock separable from the long-time diffusion clock.
    """

    cage_variance: float
    cage_tau: float
    jump_variance: float
    persistence_mean: float
    exchange_mean: float


@dataclass(frozen=True)
class LangevinCageLandscapeParams:
    """Overdamped Langevin landscape inputs for an effective renewal bridge.

    This is a microscopic-parameter bridge for a local metastable basin and an
    activated escape saddle. It does not derive the many-body landscape itself.
    """

    temperature: float
    friction: float
    cage_curvature: float
    saddle_curvature: float
    barrier_height: float
    jump_length: float
    persistence_barrier_extra: float = 0.0
    exchange_barrier_extra: float = 0.0
    dimension: int = 2


@dataclass(frozen=True)
class TranslationRotationExchangeParams:
    """Coupled translational and rotational persistence/exchange clocks.

    Translation uses the existing renewal-cage displacement parameters.
    Rotation is represented by a second renewal clock and a per-exchange
    orientational correlation factor. This is an effective diagnostic for
    translation-rotation decoupling, not a microscopic orientational potential.
    """

    cage_variance: float
    cage_tau: float
    jump_variance: float
    translational_persistence_mean: float
    translational_exchange_mean: float
    rotational_persistence_mean: float
    rotational_exchange_mean: float
    rotational_step_correlation: float


def _validate(params: DelayedRenewalCageParams) -> None:
    for name in ("cage_variance", "cage_tau", "jump_variance", "renewal_rate", "renewal_delay"):
        value = getattr(params, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")


def _validate_gamma_exchange(heterogeneity: GammaExchangeParams) -> None:
    for name in ("shape", "exchange_renewal_count"):
        value = getattr(heterogeneity, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")


def _validate_temperature_law(law: TemperatureLawParams) -> None:
    for name in (
        "reference_temperature",
        "cage_variance_ref",
        "cage_tau_ref",
        "jump_to_cage_ref",
        "renewal_rate_ref",
        "renewal_delay_ref",
    ):
        value = getattr(law, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in (
        "rate_activation",
        "delay_activation",
        "cage_stiffening",
        "jump_to_cage_growth",
        "cage_tau_activation",
    ):
        value = getattr(law, name)
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")


def _validate_configurational_entropy_law(law: ConfigurationalEntropyParams) -> None:
    if law.reference_temperature <= 0.0:
        raise ValueError("reference_temperature must be positive")
    if law.entropy_ref <= 0.0:
        raise ValueError("entropy_ref must be positive")
    if law.kauzmann_temperature <= 0.0:
        raise ValueError("kauzmann_temperature must be positive")
    if law.kauzmann_temperature >= law.reference_temperature:
        raise ValueError("kauzmann_temperature must be below reference_temperature")


def _validate_mct_beta_params(params: MCTBetaParams) -> None:
    if not (0.0 < params.plateau < 1.0):
        raise ValueError("plateau must lie between zero and one")
    for name in ("critical_amplitude", "von_schweidler_amplitude"):
        if getattr(params, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not (0.0 < params.critical_exponent < 0.5):
        raise ValueError("critical_exponent must lie between zero and one half")
    if not (0.0 < params.von_schweidler_exponent < 1.0):
        raise ValueError("von_schweidler_exponent must lie between zero and one")
    if params.beta_time <= 0.0:
        raise ValueError("beta_time must be positive")


def _validate_facilitated_exchange_law(law: FacilitatedExchangeLawParams) -> None:
    for name in ("reference_temperature", "shape_ref", "exchange_renewal_count_ref"):
        value = getattr(law, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("shape_broadening_barrier", "exchange_slowing_barrier"):
        value = getattr(law, name)
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")


def _validate_persistence_exchange(params: PersistenceExchangeParams) -> None:
    for name in ("cage_variance", "cage_tau", "jump_variance", "persistence_mean", "exchange_mean"):
        value = getattr(params, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")


def _validate_langevin_landscape(params: LangevinCageLandscapeParams) -> None:
    for name in (
        "temperature",
        "friction",
        "cage_curvature",
        "saddle_curvature",
        "barrier_height",
        "jump_length",
    ):
        value = getattr(params, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("persistence_barrier_extra", "exchange_barrier_extra"):
        value = getattr(params, name)
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if int(params.dimension) != params.dimension or params.dimension <= 0:
        raise ValueError("dimension must be a positive integer")


def _validate_translation_rotation(params: TranslationRotationExchangeParams) -> None:
    for name in (
        "cage_variance",
        "cage_tau",
        "jump_variance",
        "translational_persistence_mean",
        "translational_exchange_mean",
        "rotational_persistence_mean",
        "rotational_exchange_mean",
    ):
        value = getattr(params, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 < params.rotational_step_correlation < 1.0:
        raise ValueError("rotational_step_correlation must lie between zero and one")


def activated_barrier_temperature_law(barrier: ActivatedBarrierParams) -> TemperatureLawParams:
    """Convert activated cage-breaking barriers into the temperature law.

    The renewal rate is proportional to exp(-E_lambda/T), while the delayed
    onset time is proportional to exp(E_d/T). Relative to a reference
    temperature this gives the same Delta_T law used by TemperatureLawParams.
    """

    law = TemperatureLawParams(
        reference_temperature=barrier.reference_temperature,
        cage_variance_ref=barrier.cage_variance_ref,
        cage_tau_ref=barrier.cage_tau_ref,
        jump_to_cage_ref=barrier.jump_to_cage_ref,
        renewal_rate_ref=barrier.renewal_rate_ref,
        renewal_delay_ref=barrier.renewal_delay_ref,
        rate_activation=barrier.renewal_rate_barrier,
        delay_activation=barrier.delay_onset_barrier,
        cage_stiffening=barrier.cage_stiffening_barrier,
        jump_to_cage_growth=barrier.jump_to_cage_barrier,
        cage_tau_activation=barrier.cage_tau_barrier,
    )
    _validate_temperature_law(law)
    return law


def langevin_bare_diffusion(params: LangevinCageLandscapeParams) -> float:
    """Einstein diffusion coefficient for an overdamped Langevin particle."""

    _validate_langevin_landscape(params)
    return params.temperature / params.friction


def langevin_cage_ou_parameters(params: LangevinCageLandscapeParams) -> dict[str, float]:
    """OU cage variance and relaxation time from equipartition and friction."""

    _validate_langevin_landscape(params)
    return {
        "cage_variance": params.temperature / params.cage_curvature,
        "cage_tau": params.friction / params.cage_curvature,
    }


def kramers_escape_rate(
    *,
    temperature: float,
    friction: float,
    basin_curvature: float,
    saddle_curvature: float,
    barrier_height: float,
) -> float:
    """Overdamped one-dimensional Kramers escape rate for a local barrier."""

    for name, value in (
        ("temperature", temperature),
        ("friction", friction),
        ("basin_curvature", basin_curvature),
        ("saddle_curvature", saddle_curvature),
        ("barrier_height", barrier_height),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    prefactor = math.sqrt(basin_curvature * saddle_curvature) / (2.0 * math.pi * friction)
    return prefactor * math.exp(-barrier_height / temperature)


def langevin_to_persistence_exchange(params: LangevinCageLandscapeParams) -> PersistenceExchangeParams:
    """Coarse-grain a local Langevin barrier model to persistence/exchange parameters."""

    _validate_langevin_landscape(params)
    ou = langevin_cage_ou_parameters(params)
    persistence_rate = kramers_escape_rate(
        temperature=params.temperature,
        friction=params.friction,
        basin_curvature=params.cage_curvature,
        saddle_curvature=params.saddle_curvature,
        barrier_height=params.barrier_height + params.persistence_barrier_extra,
    )
    exchange_rate = kramers_escape_rate(
        temperature=params.temperature,
        friction=params.friction,
        basin_curvature=params.cage_curvature,
        saddle_curvature=params.saddle_curvature,
        barrier_height=params.barrier_height + params.exchange_barrier_extra,
    )
    return PersistenceExchangeParams(
        cage_variance=ou["cage_variance"],
        cage_tau=ou["cage_tau"],
        jump_variance=params.jump_length**2 / float(params.dimension),
        persistence_mean=1.0 / persistence_rate,
        exchange_mean=1.0 / exchange_rate,
    )


def langevin_first_principles_bridge_audit(params: LangevinCageLandscapeParams) -> dict[str, float | str]:
    """Audit what the Langevin-to-renewal bridge derives and what it assumes."""

    effective = langevin_to_persistence_exchange(params)
    return {
        "bridge_stage": "langevin_kramers_to_renewal_effective_theory",
        "langevin_equation_specified": 1.0,
        "ou_cage_params_derived": 1.0,
        "kramers_rates_derived": 1.0,
        "persistence_exchange_params_derived": 1.0,
        "bare_diffusion": langevin_bare_diffusion(params),
        "cage_variance": effective.cage_variance,
        "cage_tau": effective.cage_tau,
        "jump_variance": effective.jump_variance,
        "persistence_mean": effective.persistence_mean,
        "exchange_mean": effective.exchange_mean,
        "persistence_exchange_ratio": effective.persistence_mean / effective.exchange_mean,
        "full_many_body_first_principles_claim_allowed": 0.0,
        "remaining_assumption": "metastable_basin_partition_and_barrier_inputs",
    }


def temperature_dependent_params(temperature: float, law: TemperatureLawParams) -> DelayedRenewalCageParams:
    """Map reduced temperature to delayed-renewal cage parameters."""

    _validate_temperature_law(law)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    inverse_temperature_shift = 1.0 / temperature - 1.0 / law.reference_temperature
    cage_variance = law.cage_variance_ref * math.exp(-law.cage_stiffening * inverse_temperature_shift)
    jump_to_cage = law.jump_to_cage_ref * math.exp(law.jump_to_cage_growth * inverse_temperature_shift)
    return DelayedRenewalCageParams(
        cage_variance=cage_variance,
        cage_tau=law.cage_tau_ref * math.exp(law.cage_tau_activation * inverse_temperature_shift),
        jump_variance=cage_variance * jump_to_cage,
        renewal_rate=law.renewal_rate_ref * math.exp(-law.rate_activation * inverse_temperature_shift),
        renewal_delay=law.renewal_delay_ref * math.exp(law.delay_activation * inverse_temperature_shift),
    )


def configurational_entropy(temperature: np.ndarray, law: ConfigurationalEntropyParams) -> np.ndarray:
    """Linear Kauzmann configurational entropy extrapolation."""

    _validate_configurational_entropy_law(law)
    temperature = np.asarray(temperature, dtype=float)
    if np.any(temperature <= 0.0):
        raise ValueError("temperatures must be positive")
    entropy = law.entropy_ref * (temperature - law.kauzmann_temperature) / (
        law.reference_temperature - law.kauzmann_temperature
    )
    return np.maximum(entropy, 0.0)


def excess_heat_capacity(temperature: np.ndarray, law: ConfigurationalEntropyParams) -> np.ndarray:
    """Excess heat capacity implied by ``s_c(T)`` via ``Delta c_p=T ds_c/dT``."""

    _validate_configurational_entropy_law(law)
    temperature = np.asarray(temperature, dtype=float)
    if np.any(temperature <= 0.0):
        raise ValueError("temperatures must be positive")
    slope = law.entropy_ref / (law.reference_temperature - law.kauzmann_temperature)
    return temperature * slope


def adam_gibbs_relaxation_time(
    temperature: np.ndarray,
    law: ConfigurationalEntropyParams,
    *,
    activation_free_energy: float,
    tau_ref: float,
) -> np.ndarray:
    """Adam-Gibbs relaxation time normalized at the reference temperature."""

    if activation_free_energy <= 0.0:
        raise ValueError("activation_free_energy must be positive")
    if tau_ref <= 0.0:
        raise ValueError("tau_ref must be positive")
    temperature = np.asarray(temperature, dtype=float)
    entropy = configurational_entropy(temperature, law)
    if np.any(entropy <= 0.0):
        raise ValueError("temperatures must remain above the Kauzmann temperature")
    reference_term = activation_free_energy / (law.reference_temperature * law.entropy_ref)
    term = activation_free_energy / (temperature * entropy)
    return tau_ref * np.exp(term - reference_term)


def adam_gibbs_thermodynamic_scan(
    *,
    temperatures: np.ndarray,
    entropy_law: ConfigurationalEntropyParams,
    activation_free_energy: float,
    tau_ref: float,
    renewal_rate_ref: float,
    wave_number: float,
    cage_variance: float,
    cage_tau: float,
    jump_variance: float,
) -> list[dict[str, float]]:
    """Thermodynamic extension linking configurational entropy to renewal slowdown."""

    if renewal_rate_ref <= 0.0:
        raise ValueError("renewal_rate_ref must be positive")
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    for name, value in {
        "cage_variance": cage_variance,
        "cage_tau": cage_tau,
        "jump_variance": jump_variance,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    temperatures = np.asarray(temperatures, dtype=float)
    entropy = configurational_entropy(temperatures, entropy_law)
    heat_capacity = excess_heat_capacity(temperatures, entropy_law)
    tau_ag = adam_gibbs_relaxation_time(
        temperatures,
        entropy_law,
        activation_free_energy=activation_free_energy,
        tau_ref=tau_ref,
    )
    rows: list[dict[str, float]] = []
    reference_tau_alpha = None
    for temperature, entropy_value, heat_value, tau_value in zip(temperatures, entropy, heat_capacity, tau_ag):
        params = DelayedRenewalCageParams(
            cage_variance=cage_variance,
            cage_tau=cage_tau,
            jump_variance=jump_variance,
            renewal_rate=renewal_rate_ref,
            renewal_delay=float(tau_value),
        )
        tau_alpha = alpha_relaxation_time(wave_number, params)
        if reference_tau_alpha is None:
            reference_tau_alpha = tau_alpha
        rows.append(
            {
                "temperature": float(temperature),
                "configurational_entropy": float(entropy_value),
                "excess_heat_capacity": float(heat_value),
                "adam_gibbs_tau": float(tau_value),
                "renewal_rate": renewal_rate_ref,
                "renewal_delay": float(tau_value),
                "tau_alpha": tau_alpha,
                "thermodynamic_slowdown": float(tau_value / tau_ref),
                "tau_alpha_growth": tau_alpha / reference_tau_alpha,
                "inverse_entropy_control": float(1.0 / (temperature * entropy_value)),
            }
        )
    return rows


def mct_exponent_parameter_from_exponents(critical_exponent: float, von_schweidler_exponent: float) -> dict[str, float]:
    """Return the MCT exponent-parameter values implied by ``a`` and ``b``.

    Idealized MCT relates the two exponents to a common parameter
    ``lambda``. Experimental beta-window fits can use the mismatch between
    the two estimates as a falsification diagnostic.
    """

    if not (0.0 < critical_exponent < 0.5):
        raise ValueError("critical_exponent must lie between zero and one half")
    if not (0.0 < von_schweidler_exponent < 1.0):
        raise ValueError("von_schweidler_exponent must lie between zero and one")
    lambda_from_a = math.gamma(1.0 - critical_exponent) ** 2 / math.gamma(1.0 - 2.0 * critical_exponent)
    lambda_from_b = math.gamma(1.0 + von_schweidler_exponent) ** 2 / math.gamma(
        1.0 + 2.0 * von_schweidler_exponent
    )
    return {
        "lambda_from_a": lambda_from_a,
        "lambda_from_b": lambda_from_b,
        "lambda_mismatch": lambda_from_a - lambda_from_b,
        "lambda_relative_mismatch": abs(lambda_from_a - lambda_from_b) / ((lambda_from_a + lambda_from_b) / 2.0),
    }


def mct_beta_correlator(time: np.ndarray, params: MCTBetaParams) -> np.ndarray:
    """Piecewise MCT beta-window envelope for a normalized correlator."""

    _validate_mct_beta_params(params)
    time = np.asarray(time, dtype=float)
    if np.any(time <= 0.0):
        raise ValueError("time values must be positive")
    scaled = time / params.beta_time
    correlator = np.where(
        scaled < 1.0,
        params.plateau + params.critical_amplitude * scaled ** (-params.critical_exponent),
        params.plateau - params.von_schweidler_amplitude * scaled ** params.von_schweidler_exponent,
    )
    if np.any(correlator <= 0.0) or np.any(correlator >= 1.0):
        raise ValueError("beta-window correlator left the physical interval (0, 1)")
    return correlator


def mct_beta_temperature_scan(
    *,
    temperatures: np.ndarray,
    base: MCTBetaParams,
    beta_time_activation: float,
    plateau_growth: float,
    alpha_time_ref: float,
    alpha_activation: float,
) -> list[dict[str, float]]:
    """Cooling scan for an effective MCT beta window coupled to alpha slowing."""

    _validate_mct_beta_params(base)
    if beta_time_activation < 0.0:
        raise ValueError("beta_time_activation must be nonnegative")
    if plateau_growth < 0.0:
        raise ValueError("plateau_growth must be nonnegative")
    if alpha_time_ref <= 0.0:
        raise ValueError("alpha_time_ref must be positive")
    if alpha_activation < 0.0:
        raise ValueError("alpha_activation must be nonnegative")
    temperatures = np.asarray(temperatures, dtype=float)
    if np.any(temperatures <= 0.0):
        raise ValueError("temperatures must be positive")
    reference_temperature = temperatures[0]
    exponent = mct_exponent_parameter_from_exponents(base.critical_exponent, base.von_schweidler_exponent)
    rows: list[dict[str, float]] = []
    for temperature in temperatures:
        inverse_shift = 1.0 / temperature - 1.0 / reference_temperature
        beta_time = base.beta_time * math.exp(beta_time_activation * inverse_shift)
        alpha_time = alpha_time_ref * math.exp(alpha_activation * inverse_shift)
        plateau = min(0.98, base.plateau + plateau_growth * inverse_shift)
        exit_level = 0.5 * plateau
        exit_scaled = ((plateau - exit_level) / base.von_schweidler_amplitude) ** (
            1.0 / base.von_schweidler_exponent
        )
        rows.append(
            {
                "temperature": float(temperature),
                "inverse_temperature_shift": float(inverse_shift),
                "plateau": plateau,
                "critical_exponent": base.critical_exponent,
                "von_schweidler_exponent": base.von_schweidler_exponent,
                "lambda_from_a": exponent["lambda_from_a"],
                "lambda_from_b": exponent["lambda_from_b"],
                "lambda_relative_mismatch": exponent["lambda_relative_mismatch"],
                "beta_time": beta_time,
                "critical_entry_time": beta_time,
                "von_schweidler_exit_time": beta_time * exit_scaled,
                "alpha_time": alpha_time,
                "alpha_beta_separation": alpha_time / beta_time,
            }
        )
    return rows


def mct_beta_benchmark_consistency(
    beta: MCTBetaParams,
    *,
    benchmark_id: str,
    observed_critical_decay: bool,
    observed_von_schweidler: bool,
    observation_min_time: float,
    observation_max_time: float,
    alpha_time: float,
    required_decades: float,
) -> dict[str, float | str]:
    """Compare qualitative MCT beta-window observations with visible model windows."""

    _validate_mct_beta_params(beta)
    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    for name, value in {
        "observation_min_time": observation_min_time,
        "observation_max_time": observation_max_time,
        "alpha_time": alpha_time,
        "required_decades": required_decades,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if observation_max_time <= observation_min_time:
        raise ValueError("observation_max_time must exceed observation_min_time")

    if observation_min_time < beta.beta_time:
        critical_start = max(observation_min_time, np.finfo(float).tiny)
        critical_window_decades = math.log10(beta.beta_time / critical_start)
    else:
        critical_window_decades = 0.0

    von_window_end = min(observation_max_time, alpha_time)
    if von_window_end > beta.beta_time:
        von_schweidler_window_decades = math.log10(von_window_end / beta.beta_time)
    else:
        von_schweidler_window_decades = 0.0

    model_critical_visible = critical_window_decades >= required_decades
    model_von_visible = von_schweidler_window_decades >= required_decades
    critical_consistent = model_critical_visible == observed_critical_decay
    von_consistent = model_von_visible == observed_von_schweidler
    return {
        "benchmark_id": benchmark_id,
        "observed_critical_decay": float(observed_critical_decay),
        "observed_von_schweidler": float(observed_von_schweidler),
        "required_decades": required_decades,
        "critical_window_decades": critical_window_decades,
        "von_schweidler_window_decades": von_schweidler_window_decades,
        "model_predicts_visible_critical_decay": float(model_critical_visible),
        "model_predicts_visible_von_schweidler": float(model_von_visible),
        "critical_decay_consistent": float(critical_consistent),
        "von_schweidler_consistent": float(von_consistent),
        "overall_consistent": float(critical_consistent and von_consistent),
    }


def cage_localization_diagnostics(
    *,
    wave_number: float,
    plateau_time: float,
    params: DelayedRenewalCageParams,
) -> dict[str, float]:
    """Quantify Debye-Waller localization and renewal leakage in the cage plateau."""

    _validate(params)
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if plateau_time <= 0.0:
        raise ValueError("plateau_time must be positive")
    local = float(local_cage_variance(np.array([plateau_time]), params)[0])
    renewal = float(delayed_poisson_mean(np.array([plateau_time]), params)[0])
    renewal_msd = params.jump_variance * renewal
    plateau_msd = local + renewal_msd
    tau_alpha = alpha_relaxation_time(wave_number, params)
    return {
        "wave_number": wave_number,
        "plateau_time": plateau_time,
        "cage_variance": params.cage_variance,
        "cage_tau": params.cage_tau,
        "local_cage_msd": local,
        "renewal_count_at_plateau": renewal,
        "renewal_msd_at_plateau": renewal_msd,
        "cage_plateau_msd": plateau_msd,
        "renewal_msd_fraction": renewal_msd / plateau_msd if plateau_msd > 0.0 else math.nan,
        "debye_waller_plateau": math.exp(-0.5 * wave_number**2 * params.cage_variance),
        "tau_alpha": tau_alpha,
        "alpha_to_cage_time_ratio": tau_alpha / params.cage_tau,
    }


def cage_localization_benchmark_consistency(
    *,
    benchmark_id: str,
    observed_cage_localization: bool,
    debye_waller_plateau: float,
    renewal_msd_fraction: float,
    alpha_to_cage_time_ratio: float,
    min_debye_waller_plateau: float,
    max_debye_waller_plateau: float,
    max_renewal_msd_fraction: float,
    min_alpha_to_cage_time_ratio: float,
) -> dict[str, float | str]:
    """Check cage localization against plateau, renewal leakage, and alpha separation."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    if not (0.0 < min_debye_waller_plateau < max_debye_waller_plateau < 1.0):
        raise ValueError("Debye-Waller thresholds must satisfy 0 < min < max < 1")
    for name, value in {
        "debye_waller_plateau": debye_waller_plateau,
        "alpha_to_cage_time_ratio": alpha_to_cage_time_ratio,
        "max_renewal_msd_fraction": max_renewal_msd_fraction,
        "min_alpha_to_cage_time_ratio": min_alpha_to_cage_time_ratio,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if renewal_msd_fraction < 0.0:
        raise ValueError("renewal_msd_fraction must be nonnegative")

    debye_flag = min_debye_waller_plateau <= debye_waller_plateau <= max_debye_waller_plateau
    renewal_flag = renewal_msd_fraction <= max_renewal_msd_fraction
    separation_flag = alpha_to_cage_time_ratio >= min_alpha_to_cage_time_ratio
    model_flag = debye_flag and renewal_flag and separation_flag
    return {
        "benchmark_id": benchmark_id,
        "observed_cage_localization": float(observed_cage_localization),
        "debye_waller_plateau": debye_waller_plateau,
        "min_debye_waller_plateau": min_debye_waller_plateau,
        "max_debye_waller_plateau": max_debye_waller_plateau,
        "renewal_msd_fraction": renewal_msd_fraction,
        "max_renewal_msd_fraction": max_renewal_msd_fraction,
        "alpha_to_cage_time_ratio": alpha_to_cage_time_ratio,
        "min_alpha_to_cage_time_ratio": min_alpha_to_cage_time_ratio,
        "model_predicts_cage_localization": float(model_flag),
        "debye_waller_consistent": float(debye_flag == observed_cage_localization),
        "renewal_fraction_consistent": float(renewal_flag == observed_cage_localization),
        "alpha_separation_consistent": float(separation_flag == observed_cage_localization),
        "overall_consistent": float(model_flag == observed_cage_localization),
    }


def mct_exponent_benchmark_consistency(
    *,
    benchmark_id: str,
    observed_common_exponent_parameter: bool,
    critical_exponent: float,
    von_schweidler_exponent: float,
    max_lambda_relative_mismatch: float,
) -> dict[str, float | str]:
    """Check whether fitted beta exponents share one MCT exponent parameter."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    if max_lambda_relative_mismatch < 0.0:
        raise ValueError("max_lambda_relative_mismatch must be nonnegative")

    exponent = mct_exponent_parameter_from_exponents(critical_exponent, von_schweidler_exponent)
    model_flag = exponent["lambda_relative_mismatch"] <= max_lambda_relative_mismatch
    consistent = model_flag == observed_common_exponent_parameter
    return {
        "benchmark_id": benchmark_id,
        "observed_common_exponent_parameter": float(observed_common_exponent_parameter),
        "critical_exponent_benchmark": critical_exponent,
        "von_schweidler_exponent_benchmark": von_schweidler_exponent,
        "lambda_from_a": exponent["lambda_from_a"],
        "lambda_from_b": exponent["lambda_from_b"],
        "lambda_mismatch": exponent["lambda_mismatch"],
        "lambda_relative_mismatch": exponent["lambda_relative_mismatch"],
        "max_lambda_relative_mismatch": max_lambda_relative_mismatch,
        "model_predicts_common_exponent_parameter": float(model_flag),
        "mct_exponent_parameter_consistent": float(consistent),
        "overall_consistent": float(consistent),
    }


def gaussian_recovery_benchmark_consistency(
    *,
    benchmark_id: str,
    observed_gaussian_recovery: bool,
    finite_exchange_late_ngp: float,
    static_gamma_late_ngp: float,
    recovery_threshold: float,
) -> dict[str, float | str]:
    """Check Gaussian-recovery evidence against finite exchange and static disorder."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    for name, value in {
        "finite_exchange_late_ngp": finite_exchange_late_ngp,
        "static_gamma_late_ngp": static_gamma_late_ngp,
        "recovery_threshold": recovery_threshold,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")

    finite_exchange_recovers = finite_exchange_late_ngp < recovery_threshold
    static_null_recovers = static_gamma_late_ngp < recovery_threshold
    finite_exchange_consistent = finite_exchange_recovers == observed_gaussian_recovery
    static_null_consistent = static_null_recovers == observed_gaussian_recovery
    mechanism_selection_consistent = finite_exchange_consistent and not static_null_consistent
    return {
        "benchmark_id": benchmark_id,
        "observed_gaussian_recovery": float(observed_gaussian_recovery),
        "finite_exchange_late_ngp": finite_exchange_late_ngp,
        "static_gamma_late_ngp": static_gamma_late_ngp,
        "recovery_threshold": recovery_threshold,
        "model_predicts_gaussian_recovery": float(finite_exchange_recovers),
        "static_null_predicts_gaussian_recovery": float(static_null_recovers),
        "finite_exchange_recovery_consistent": float(finite_exchange_consistent),
        "static_null_recovery_consistent": float(static_null_consistent),
        "mechanism_selection_consistent": float(mechanism_selection_consistent),
        "overall_consistent": float(mechanism_selection_consistent),
    }


def ngp_peak_benchmark_consistency(
    *,
    benchmark_id: str,
    observed_transient_ngp_peak: bool,
    hot_peak_time: float,
    cold_peak_time: float,
    hot_peak_ngp: float,
    cold_peak_ngp: float,
    late_ngp: float,
    min_peak_time_growth: float,
    min_peak_height: float,
    min_peak_height_growth: float,
    max_late_ngp: float,
) -> dict[str, float | str]:
    """Check that cooling shifts the transient NGP peak while preserving recovery."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    for name, value in {
        "hot_peak_time": hot_peak_time,
        "cold_peak_time": cold_peak_time,
        "hot_peak_ngp": hot_peak_ngp,
        "cold_peak_ngp": cold_peak_ngp,
        "min_peak_time_growth": min_peak_time_growth,
        "min_peak_height": min_peak_height,
        "min_peak_height_growth": min_peak_height_growth,
        "max_late_ngp": max_late_ngp,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if late_ngp < 0.0:
        raise ValueError("late_ngp must be nonnegative")

    peak_time_growth = cold_peak_time / hot_peak_time
    peak_height_growth = cold_peak_ngp / hot_peak_ngp
    time_flag = peak_time_growth >= min_peak_time_growth
    height_flag = hot_peak_ngp >= min_peak_height and cold_peak_ngp >= min_peak_height
    height_growth_flag = peak_height_growth >= min_peak_height_growth
    late_recovery_flag = late_ngp <= max_late_ngp
    model_flag = time_flag and height_flag and height_growth_flag and late_recovery_flag
    time_consistent = time_flag == observed_transient_ngp_peak
    height_consistent = height_flag == observed_transient_ngp_peak
    height_growth_consistent = height_growth_flag == observed_transient_ngp_peak
    late_recovery_consistent = late_recovery_flag == observed_transient_ngp_peak
    return {
        "benchmark_id": benchmark_id,
        "observed_transient_ngp_peak": float(observed_transient_ngp_peak),
        "hot_peak_time": hot_peak_time,
        "cold_peak_time": cold_peak_time,
        "peak_time_growth": peak_time_growth,
        "hot_peak_ngp": hot_peak_ngp,
        "cold_peak_ngp": cold_peak_ngp,
        "peak_height_growth": peak_height_growth,
        "late_ngp": late_ngp,
        "min_peak_time_growth": min_peak_time_growth,
        "min_peak_height": min_peak_height,
        "min_peak_height_growth": min_peak_height_growth,
        "max_late_ngp": max_late_ngp,
        "model_predicts_transient_ngp_peak": float(model_flag),
        "peak_time_growth_consistent": float(time_consistent),
        "peak_height_consistent": float(height_consistent),
        "peak_height_growth_consistent": float(height_growth_consistent),
        "late_recovery_consistent": float(late_recovery_consistent),
        "overall_consistent": float(
            model_flag == observed_transient_ngp_peak
            and time_consistent
            and height_consistent
            and height_growth_consistent
            and late_recovery_consistent
        ),
    }


def stokes_einstein_benchmark_consistency(
    *,
    benchmark_id: str,
    observed_stokes_einstein_violation: bool,
    hot_se_product: float,
    cold_se_product: float,
    cold_fractional_exponent: float,
    min_product_growth: float,
    max_fractional_exponent: float,
) -> dict[str, float | str]:
    """Check Stokes-Einstein decoupling against product growth and fractional exponent."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    for name, value in {
        "hot_se_product": hot_se_product,
        "cold_se_product": cold_se_product,
        "min_product_growth": min_product_growth,
        "max_fractional_exponent": max_fractional_exponent,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not (0.0 < cold_fractional_exponent):
        raise ValueError("cold_fractional_exponent must be positive")

    se_product_growth = cold_se_product / hot_se_product
    product_flag = se_product_growth >= min_product_growth
    fractional_flag = cold_fractional_exponent <= max_fractional_exponent
    model_flag = product_flag and fractional_flag
    product_consistent = product_flag == observed_stokes_einstein_violation
    fractional_consistent = fractional_flag == observed_stokes_einstein_violation
    return {
        "benchmark_id": benchmark_id,
        "observed_stokes_einstein_violation": float(observed_stokes_einstein_violation),
        "hot_se_product": hot_se_product,
        "cold_se_product": cold_se_product,
        "se_product_growth": se_product_growth,
        "cold_fractional_exponent": cold_fractional_exponent,
        "min_product_growth": min_product_growth,
        "max_fractional_exponent": max_fractional_exponent,
        "model_predicts_stokes_einstein_violation": float(model_flag),
        "se_product_growth_consistent": float(product_consistent),
        "fractional_exponent_consistent": float(fractional_consistent),
        "overall_consistent": float(model_flag == observed_stokes_einstein_violation and product_consistent and fractional_consistent),
    }


def dynamic_heterogeneity_benchmark_consistency(
    *,
    benchmark_id: str,
    observed_dynamic_heterogeneity_growth: bool,
    length_growth: float,
    correlation_size_growth: float,
    chi4_peak_growth: float,
    min_length_growth: float,
    min_correlation_size_growth: float,
    min_chi4_peak_growth: float,
) -> dict[str, float | str]:
    """Check dynamic-heterogeneity growth against xi4, Ncorr, and chi4 peak growth."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    for name, value in {
        "length_growth": length_growth,
        "correlation_size_growth": correlation_size_growth,
        "chi4_peak_growth": chi4_peak_growth,
        "min_length_growth": min_length_growth,
        "min_correlation_size_growth": min_correlation_size_growth,
        "min_chi4_peak_growth": min_chi4_peak_growth,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")

    length_flag = length_growth >= min_length_growth
    size_flag = correlation_size_growth >= min_correlation_size_growth
    chi4_flag = chi4_peak_growth >= min_chi4_peak_growth
    model_flag = length_flag and size_flag and chi4_flag
    length_consistent = length_flag == observed_dynamic_heterogeneity_growth
    size_consistent = size_flag == observed_dynamic_heterogeneity_growth
    chi4_consistent = chi4_flag == observed_dynamic_heterogeneity_growth
    return {
        "benchmark_id": benchmark_id,
        "observed_dynamic_heterogeneity_growth": float(observed_dynamic_heterogeneity_growth),
        "length_growth": length_growth,
        "correlation_size_growth": correlation_size_growth,
        "chi4_peak_growth_benchmark": chi4_peak_growth,
        "min_length_growth": min_length_growth,
        "min_correlation_size_growth": min_correlation_size_growth,
        "min_chi4_peak_growth": min_chi4_peak_growth,
        "model_predicts_dynamic_heterogeneity_growth": float(model_flag),
        "length_growth_consistent": float(length_consistent),
        "correlation_size_growth_consistent": float(size_consistent),
        "chi4_peak_growth_consistent": float(chi4_consistent),
        "overall_consistent": float(model_flag == observed_dynamic_heterogeneity_growth and length_consistent and size_consistent and chi4_consistent),
    }


def alpha_tts_benchmark_consistency(
    *,
    benchmark_id: str,
    observed_tts_breakdown: bool,
    cold_shape_residual: float,
    alpha_shape_control_growth: float,
    residual_threshold: float,
    min_control_growth: float,
) -> dict[str, float | str]:
    """Check alpha time-temperature-superposition breakdown against shape diagnostics."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    for name, value in {
        "cold_shape_residual": cold_shape_residual,
        "alpha_shape_control_growth": alpha_shape_control_growth,
        "residual_threshold": residual_threshold,
        "min_control_growth": min_control_growth,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")

    residual_flag = cold_shape_residual >= residual_threshold
    control_flag = alpha_shape_control_growth >= min_control_growth
    model_flag = residual_flag and control_flag
    residual_consistent = residual_flag == observed_tts_breakdown
    control_consistent = control_flag == observed_tts_breakdown
    return {
        "benchmark_id": benchmark_id,
        "observed_tts_breakdown": float(observed_tts_breakdown),
        "cold_shape_residual": cold_shape_residual,
        "alpha_shape_control_growth": alpha_shape_control_growth,
        "residual_threshold": residual_threshold,
        "min_control_growth": min_control_growth,
        "model_predicts_tts_breakdown": float(model_flag),
        "tts_residual_consistent": float(residual_consistent),
        "tts_control_consistent": float(control_consistent),
        "overall_consistent": float(model_flag == observed_tts_breakdown and residual_consistent and control_consistent),
    }


def persistence_exchange_benchmark_consistency(
    *,
    benchmark_id: str,
    observed_persistence_exchange_decoupling: bool,
    inferred_persistence_exchange_ratio: float,
    late_ngp_log_residual: float,
    invalid_poisson_alpha_rejected: bool,
    min_persistence_exchange_ratio: float,
    max_late_ngp_abs_log_residual: float,
) -> dict[str, float | str]:
    """Check persistence/exchange inversion against held-out falsification tests."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    for name, value in {
        "inferred_persistence_exchange_ratio": inferred_persistence_exchange_ratio,
        "min_persistence_exchange_ratio": min_persistence_exchange_ratio,
        "max_late_ngp_abs_log_residual": max_late_ngp_abs_log_residual,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(late_ngp_log_residual):
        raise ValueError("late_ngp_log_residual must be finite")

    ratio_flag = inferred_persistence_exchange_ratio >= min_persistence_exchange_ratio
    late_ngp_flag = abs(late_ngp_log_residual) <= max_late_ngp_abs_log_residual
    rejection_flag = invalid_poisson_alpha_rejected
    model_flag = ratio_flag and late_ngp_flag and rejection_flag
    ratio_consistent = ratio_flag == observed_persistence_exchange_decoupling
    late_ngp_consistent = late_ngp_flag == observed_persistence_exchange_decoupling
    rejection_consistent = rejection_flag == observed_persistence_exchange_decoupling
    return {
        "benchmark_id": benchmark_id,
        "observed_persistence_exchange_decoupling": float(observed_persistence_exchange_decoupling),
        "inferred_persistence_exchange_ratio": inferred_persistence_exchange_ratio,
        "min_persistence_exchange_ratio": min_persistence_exchange_ratio,
        "late_ngp_log_residual_benchmark": late_ngp_log_residual,
        "max_late_ngp_abs_log_residual": max_late_ngp_abs_log_residual,
        "invalid_poisson_alpha_rejected": float(invalid_poisson_alpha_rejected),
        "model_predicts_persistence_exchange_decoupling": float(model_flag),
        "persistence_exchange_ratio_consistent": float(ratio_consistent),
        "persistence_exchange_late_ngp_consistent": float(late_ngp_consistent),
        "persistence_exchange_rejection_consistent": float(rejection_consistent),
        "overall_consistent": float(
            model_flag == observed_persistence_exchange_decoupling
            and ratio_consistent
            and late_ngp_consistent
            and rejection_consistent
        ),
    }


def joint_inversion_benchmark_consistency(
    *,
    benchmark_id: str,
    observed_joint_inversion_closure: bool,
    inferred_persistence_exchange_ratio: float,
    stokes_einstein_growth_over_poisson: float,
    max_multik_tau_alpha_abs_log_residual: float,
    late_ngp_log_residual: float,
    chi4_peak_growth_over_poisson: float,
    rejected_mismatch_abs_log_residual: float,
    min_persistence_exchange_ratio: float,
    min_stokes_einstein_growth: float,
    max_multik_abs_log_residual: float,
    max_late_ngp_abs_log_residual: float,
    min_chi4_peak_growth: float,
    min_rejected_mismatch_abs_log_residual: float,
) -> dict[str, float | str]:
    """Check the joint inversion protocol against held-out dynamics and rejection."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    for name, value in {
        "inferred_persistence_exchange_ratio": inferred_persistence_exchange_ratio,
        "stokes_einstein_growth_over_poisson": stokes_einstein_growth_over_poisson,
        "chi4_peak_growth_over_poisson": chi4_peak_growth_over_poisson,
        "rejected_mismatch_abs_log_residual": rejected_mismatch_abs_log_residual,
        "min_persistence_exchange_ratio": min_persistence_exchange_ratio,
        "min_stokes_einstein_growth": min_stokes_einstein_growth,
        "min_chi4_peak_growth": min_chi4_peak_growth,
        "min_rejected_mismatch_abs_log_residual": min_rejected_mismatch_abs_log_residual,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name, value in {
        "max_multik_tau_alpha_abs_log_residual": max_multik_tau_alpha_abs_log_residual,
        "max_multik_abs_log_residual": max_multik_abs_log_residual,
        "max_late_ngp_abs_log_residual": max_late_ngp_abs_log_residual,
    }.items():
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if not math.isfinite(late_ngp_log_residual):
        raise ValueError("late_ngp_log_residual must be finite")

    ratio_flag = inferred_persistence_exchange_ratio >= min_persistence_exchange_ratio
    se_flag = stokes_einstein_growth_over_poisson >= min_stokes_einstein_growth
    multik_flag = max_multik_tau_alpha_abs_log_residual <= max_multik_abs_log_residual
    late_ngp_abs_log_residual = abs(late_ngp_log_residual)
    late_ngp_flag = late_ngp_abs_log_residual <= max_late_ngp_abs_log_residual
    chi4_flag = chi4_peak_growth_over_poisson >= min_chi4_peak_growth
    mismatch_flag = rejected_mismatch_abs_log_residual >= min_rejected_mismatch_abs_log_residual
    model_flag = ratio_flag and se_flag and multik_flag and late_ngp_flag and chi4_flag
    ratio_consistent = ratio_flag == observed_joint_inversion_closure
    se_consistent = se_flag == observed_joint_inversion_closure
    multik_consistent = multik_flag == observed_joint_inversion_closure
    late_ngp_consistent = late_ngp_flag == observed_joint_inversion_closure
    chi4_consistent = chi4_flag == observed_joint_inversion_closure
    return {
        "benchmark_id": benchmark_id,
        "observed_joint_inversion_closure": float(observed_joint_inversion_closure),
        "joint_inferred_persistence_exchange_ratio": inferred_persistence_exchange_ratio,
        "min_joint_persistence_exchange_ratio": min_persistence_exchange_ratio,
        "joint_stokes_einstein_growth_over_poisson": stokes_einstein_growth_over_poisson,
        "min_joint_stokes_einstein_growth": min_stokes_einstein_growth,
        "joint_multik_tau_alpha_abs_log_residual": max_multik_tau_alpha_abs_log_residual,
        "max_joint_multik_abs_log_residual": max_multik_abs_log_residual,
        "joint_late_ngp_abs_log_residual": late_ngp_abs_log_residual,
        "max_joint_late_ngp_abs_log_residual": max_late_ngp_abs_log_residual,
        "joint_chi4_peak_growth_over_poisson": chi4_peak_growth_over_poisson,
        "min_joint_chi4_peak_growth": min_chi4_peak_growth,
        "rejected_mismatch_abs_log_residual": rejected_mismatch_abs_log_residual,
        "min_rejected_mismatch_abs_log_residual": min_rejected_mismatch_abs_log_residual,
        "model_predicts_joint_inversion_closure": float(model_flag),
        "joint_ratio_consistent": float(ratio_consistent),
        "joint_se_consistent": float(se_consistent),
        "joint_multik_consistent": float(multik_consistent),
        "joint_late_ngp_consistent": float(late_ngp_consistent),
        "joint_chi4_consistent": float(chi4_consistent),
        "joint_mismatch_rejected": float(mismatch_flag),
        "overall_consistent": float(
            model_flag == observed_joint_inversion_closure
            and ratio_consistent
            and se_consistent
            and multik_consistent
            and late_ngp_consistent
            and chi4_consistent
            and mismatch_flag
        ),
    }


def literature_inversion_readiness(
    *,
    benchmark_id: str,
    benchmark_source: str,
    required_observables: list[str],
    available_observables: list[str],
    has_machine_readable_data: bool,
    has_uncertainty_estimates: bool,
    next_action: str,
    min_qualitative_coverage: float = 0.5,
) -> dict[str, float | str]:
    """Score whether a literature benchmark can support qualitative or quantitative inversion."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    if not benchmark_source:
        raise ValueError("benchmark_source must be nonempty")
    if not required_observables:
        raise ValueError("required_observables must be nonempty")
    if not next_action:
        raise ValueError("next_action must be nonempty")
    if not (0.0 < min_qualitative_coverage <= 1.0):
        raise ValueError("min_qualitative_coverage must lie in (0, 1]")

    required = list(dict.fromkeys(required_observables))
    available = list(dict.fromkeys(available_observables))
    if any(not item for item in required):
        raise ValueError("required_observables must be nonempty strings")
    if any(not item for item in available):
        raise ValueError("available_observables must be nonempty strings")

    available_set = set(available)
    missing = [item for item in required if item not in available_set]
    coverage = (len(required) - len(missing)) / len(required)
    all_observables_available = len(missing) == 0
    qualitative_ready = coverage >= min_qualitative_coverage
    quantitative_ready = all_observables_available and has_machine_readable_data
    uncertainty_ready = quantitative_ready and has_uncertainty_estimates
    return {
        "benchmark_id": benchmark_id,
        "benchmark_source": benchmark_source,
        "required_observables": ";".join(required),
        "available_observables": ";".join(available),
        "missing_observables": ";".join(missing) if missing else "none",
        "observable_coverage_fraction": coverage,
        "has_machine_readable_data": float(has_machine_readable_data),
        "has_uncertainty_estimates": float(has_uncertainty_estimates),
        "qualitative_comparison_ready": float(qualitative_ready),
        "quantitative_inversion_ready": float(quantitative_ready),
        "uncertainty_weighted_ready": float(uncertainty_ready),
        "next_action": next_action,
    }


def real_benchmark_assimilation_gate(
    *,
    benchmark_id: str,
    source_key: str,
    target_protocol: str,
    available_observables: Sequence[str],
    has_shared_system: bool,
    has_machine_readable_curves: bool,
    has_uncertainty_estimates: bool,
    model_scope: str,
) -> dict[str, float | str]:
    """Gate a real benchmark before promoting it to quantitative inversion."""

    for name, value in {
        "benchmark_id": benchmark_id,
        "source_key": source_key,
        "target_protocol": target_protocol,
        "model_scope": model_scope,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")

    protocol_requirements = {
        "alpha_vanhove_transport": [
            "time_grid",
            "temperature_grid",
            "wave_numbers",
            "self_intermediate_scattering",
            "van_hove_tail",
            "ngp",
            "diffusion",
        ],
        "persistence_exchange_chi4": [
            "temperature_grid",
            "diffusion",
            "tau_alpha",
            "persistence_time",
            "exchange_time",
            "late_ngp",
            "chi4_peak",
        ],
        "spatial_chi4_front": [
            "temperature_grid",
            "tau_alpha",
            "chi4_peak",
            "dynamic_length",
            "diffusion",
        ],
        "thermodynamic_entropy_closure": [
            "temperature_grid",
            "configurational_entropy",
            "tau_alpha",
        ],
    }
    allowed_scopes = {
        "dynamical_signature",
        "transport_decoupling",
        "spatial_heterogeneity",
        "thermodynamic_transition",
    }
    if target_protocol not in protocol_requirements:
        raise ValueError("target_protocol is not recognized")
    if model_scope not in allowed_scopes:
        raise ValueError("model_scope is not recognized")
    if not available_observables:
        raise ValueError("available_observables must be nonempty")

    required = protocol_requirements[target_protocol]
    available = set(available_observables)
    missing = [observable for observable in required if observable not in available]
    coverage = (len(required) - len(missing)) / len(required)

    if model_scope == "thermodynamic_transition":
        primary_blocker = "renewal_dynamics_not_thermodynamic_theory"
        structural_ready = False
        uncertainty_ready = False
        stage = "scope_boundary_only"
    else:
        structural_ready = (
            has_shared_system and has_machine_readable_curves and not missing
        )
        uncertainty_ready = structural_ready and has_uncertainty_estimates
        if uncertainty_ready:
            primary_blocker = "none"
            stage = "uncertainty_weighted_inversion"
        elif structural_ready:
            primary_blocker = "uncertainty_columns"
            stage = "structural_digitization_ready"
        else:
            if not has_shared_system:
                primary_blocker = "shared_system_or_temperature_grid"
            elif missing:
                primary_blocker = missing[0]
            else:
                primary_blocker = "machine_readable_curves"
            stage = "qualitative_alignment_only"

    return {
        "benchmark_id": benchmark_id,
        "source_key": source_key,
        "target_protocol": target_protocol,
        "model_scope": model_scope,
        "required_observables": ";".join(required),
        "available_observables": ";".join(available_observables),
        "missing_observables": ";".join(missing) if missing else "none",
        "required_observable_coverage": float(coverage),
        "shared_system": float(has_shared_system),
        "machine_readable_curves": float(has_machine_readable_curves),
        "uncertainty_estimates": float(has_uncertainty_estimates),
        "structural_inversion_ready": float(structural_ready),
        "uncertainty_weighted_ready": float(uncertainty_ready),
        "assimilation_stage": stage,
        "primary_blocker": primary_blocker,
    }


def cross_observable_prediction_ledger(
    *,
    protocol_id: str,
    source_key: str,
    model_scope: str,
    support_level: str,
    calibration_observables: Sequence[str],
    heldout_predictions: Sequence[str],
    closure_observables: Sequence[str],
    failed_predictions: Sequence[str],
) -> dict[str, float | str]:
    """Separate fit inputs, held-out predictions, and closure inputs."""

    for name, value in {
        "protocol_id": protocol_id,
        "source_key": source_key,
        "model_scope": model_scope,
        "support_level": support_level,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")

    allowed_scopes = {
        "dynamical_signature",
        "transport_decoupling",
        "spatial_heterogeneity",
        "thermodynamic_transition",
    }
    allowed_support = {"derived", "effective_closure", "closure_only", "not_supported"}
    if model_scope not in allowed_scopes:
        raise ValueError("model_scope is not recognized")
    if support_level not in allowed_support:
        raise ValueError("support_level is not recognized")
    if not calibration_observables:
        raise ValueError("calibration_observables must be nonempty")
    for name, values in {
        "calibration_observables": calibration_observables,
        "heldout_predictions": heldout_predictions,
        "closure_observables": closure_observables,
        "failed_predictions": failed_predictions,
    }.items():
        if any(not value for value in values):
            raise ValueError(f"{name} must contain nonempty strings")

    calibration = list(dict.fromkeys(calibration_observables))
    heldout = list(dict.fromkeys(heldout_predictions))
    closures = list(dict.fromkeys(closure_observables))
    failed = list(dict.fromkeys(failed_predictions))
    unknown_failures = [prediction for prediction in failed if prediction not in set(heldout)]
    if unknown_failures:
        raise ValueError("failed_predictions must be a subset of heldout_predictions")
    if model_scope == "thermodynamic_transition" and support_level == "derived":
        raise ValueError("renewal dynamics cannot derive thermodynamic transition predictions")

    requires_closure = support_level in {"effective_closure", "closure_only"} or bool(closures)
    all_heldout_pass = bool(heldout) and not failed
    fit_only_risk = not heldout and support_level == "derived"
    if model_scope == "thermodynamic_transition":
        prediction_class = "scope_boundary"
    elif support_level == "not_supported":
        prediction_class = "not_supported"
    elif failed:
        prediction_class = "failed_prediction"
    elif support_level in {"effective_closure", "closure_only"} or closures:
        prediction_class = "closure_assisted_prediction"
    elif heldout:
        prediction_class = "predictive_diagnostic"
    else:
        prediction_class = "underconstrained_fit"

    return {
        "protocol_id": protocol_id,
        "source_key": source_key,
        "model_scope": model_scope,
        "support_level": support_level,
        "calibration_observables": ";".join(calibration),
        "heldout_predictions": ";".join(heldout) if heldout else "none",
        "closure_observables": ";".join(closures) if closures else "none",
        "failed_predictions": ";".join(failed) if failed else "none",
        "calibration_count": float(len(calibration)),
        "heldout_prediction_count": float(len(heldout)),
        "closure_observable_count": float(len(closures)),
        "all_heldout_predictions_pass": float(all_heldout_pass),
        "requires_external_closure": float(requires_closure),
        "fit_only_overclaim_risk": float(fit_only_risk),
        "prediction_class": prediction_class,
    }


def inversion_identifiability_audit(
    *,
    protocol_id: str,
    source_key: str,
    model_scope: str,
    fit_observables: Sequence[str],
    inferred_parameters: Sequence[str],
    heldout_predictions: Sequence[str],
    external_closures: Sequence[str],
    degenerate_parameters: Sequence[str],
) -> dict[str, float | str]:
    """Classify whether an inversion protocol is identifiable before fitting data."""

    for name, value in {
        "protocol_id": protocol_id,
        "source_key": source_key,
        "model_scope": model_scope,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")

    allowed_scopes = {
        "dynamical_signature",
        "transport_decoupling",
        "spatial_heterogeneity",
        "thermodynamic_transition",
    }
    if model_scope not in allowed_scopes:
        raise ValueError("model_scope is not recognized")
    if not fit_observables:
        raise ValueError("fit_observables must be nonempty")
    if not inferred_parameters:
        raise ValueError("inferred_parameters must be nonempty")
    for name, values in {
        "fit_observables": fit_observables,
        "inferred_parameters": inferred_parameters,
        "heldout_predictions": heldout_predictions,
        "external_closures": external_closures,
        "degenerate_parameters": degenerate_parameters,
    }.items():
        if any(not value for value in values):
            raise ValueError(f"{name} must contain nonempty strings")

    fit = list(dict.fromkeys(fit_observables))
    parameters = list(dict.fromkeys(inferred_parameters))
    heldout = list(dict.fromkeys(heldout_predictions))
    closures = list(dict.fromkeys(external_closures))
    degenerate = list(dict.fromkeys(degenerate_parameters))

    rank_margin = len(fit) - len(parameters)
    requires_closure = bool(closures)
    has_degeneracy = bool(degenerate)
    has_heldout_prediction = bool(heldout)

    if model_scope == "thermodynamic_transition":
        identifiability_class = "scope_boundary"
    elif has_degeneracy:
        identifiability_class = "degenerate_fit"
    elif rank_margin < 0 or not has_heldout_prediction:
        identifiability_class = "underidentified_fit"
    elif requires_closure:
        identifiability_class = "conditionally_identifiable"
    else:
        identifiability_class = "identifiable_prediction"

    overclaim_risk = identifiability_class in {"underidentified_fit", "degenerate_fit"}

    return {
        "protocol_id": protocol_id,
        "source_key": source_key,
        "model_scope": model_scope,
        "fit_observables": ";".join(fit),
        "inferred_parameters": ";".join(parameters),
        "heldout_predictions": ";".join(heldout) if heldout else "none",
        "external_closures": ";".join(closures) if closures else "none",
        "degenerate_parameters": ";".join(degenerate) if degenerate else "none",
        "fit_observable_count": float(len(fit)),
        "inferred_parameter_count": float(len(parameters)),
        "heldout_prediction_count": float(len(heldout)),
        "closure_count": float(len(closures)),
        "rank_margin": float(rank_margin),
        "requires_external_closure": float(requires_closure),
        "has_degeneracy": float(has_degeneracy),
        "overclaim_risk": float(overclaim_risk),
        "identifiability_class": identifiability_class,
    }


def frontier_benchmark_horizon(
    *,
    benchmark_id: str,
    source_key: str,
    source_year: int,
    model_scope: str,
    target_protocol: str,
    available_observables: Sequence[str],
    required_observables: Sequence[str],
    has_machine_readable_repository: bool,
    has_uncertainty_estimates: bool,
    has_shared_transport_grid: bool,
    requires_external_closure: bool,
    model_extension_required: bool,
) -> dict[str, float | str]:
    """Score frontier benchmarks as direct, reanalysis, closure, or extension targets."""

    for name, value in {
        "benchmark_id": benchmark_id,
        "source_key": source_key,
        "model_scope": model_scope,
        "target_protocol": target_protocol,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    allowed_scopes = {
        "dynamical_signature",
        "transport_decoupling",
        "spatial_heterogeneity",
        "thermodynamic_transition",
    }
    if model_scope not in allowed_scopes:
        raise ValueError("model_scope is not recognized")
    if source_year < 1900:
        raise ValueError("source_year must be a plausible publication year")
    if not available_observables:
        raise ValueError("available_observables must be nonempty")
    if not required_observables:
        raise ValueError("required_observables must be nonempty")
    if any(not observable for observable in available_observables):
        raise ValueError("available_observables must contain nonempty strings")
    if any(not observable for observable in required_observables):
        raise ValueError("required_observables must contain nonempty strings")

    available = list(dict.fromkeys(available_observables))
    required = list(dict.fromkeys(required_observables))
    available_set = set(available)
    missing = [observable for observable in required if observable not in available_set]
    trajectory_derivable = {
        "self_intermediate_scattering",
        "ngp",
        "late_ngp",
        "diffusion",
        "van_hove_tail",
        "multi_k_tau_alpha",
        "chi4_peak_proxy",
    }
    computable_missing = (
        [observable for observable in missing if observable in trajectory_derivable]
        if "particle_trajectories" in available_set
        else []
    )
    can_compute_missing = bool(missing) and set(missing).issubset(set(computable_missing))
    direct_coverage = (len(required) - len(missing)) / len(required)
    effective_coverage = (len(required) - len([item for item in missing if item not in computable_missing])) / len(
        required
    )
    transport_heterogeneity_observables = {
        "diffusion",
        "tau_alpha",
        "chi4_peak",
        "stokes_einstein_product",
    }
    has_transport_heterogeneity_core = len(transport_heterogeneity_observables & available_set) >= 3

    if model_scope == "thermodynamic_transition":
        horizon_class = "scope_boundary"
        primary_blocker = "renewal_dynamics_not_thermodynamic_theory"
    elif model_extension_required:
        horizon_class = "model_extension_required"
        primary_blocker = "model_extension_required"
    elif requires_external_closure:
        horizon_class = "closure_horizon"
        primary_blocker = "external_closure"
    elif can_compute_missing and has_machine_readable_repository and has_shared_transport_grid:
        horizon_class = "trajectory_reanalysis_candidate"
        primary_blocker = "uncertainty_estimates" if not has_uncertainty_estimates else "derived_observables"
    elif model_scope == "transport_decoupling" and has_transport_heterogeneity_core and missing:
        horizon_class = "transport_heterogeneity_candidate"
        primary_blocker = missing[0]
    elif not missing and has_machine_readable_repository and has_uncertainty_estimates and has_shared_transport_grid:
        horizon_class = "quantitative_inversion_candidate"
        primary_blocker = "none"
    elif not missing and has_machine_readable_repository and has_shared_transport_grid:
        horizon_class = "structural_inversion_candidate"
        primary_blocker = "uncertainty_estimates"
    elif missing:
        horizon_class = "qualitative_horizon"
        primary_blocker = missing[0]
    elif not has_machine_readable_repository:
        horizon_class = "qualitative_horizon"
        primary_blocker = "machine_readable_repository"
    elif not has_shared_transport_grid:
        horizon_class = "qualitative_horizon"
        primary_blocker = "shared_transport_grid"
    else:
        horizon_class = "structural_inversion_candidate"
        primary_blocker = "uncertainty_estimates"

    recent_bonus = 1.0 if source_year >= 2024 else 0.0
    score = 0.45 * effective_coverage + 0.2 * recent_bonus
    score += 0.15 * float(has_machine_readable_repository)
    score += 0.1 * float(has_shared_transport_grid)
    score += 0.1 * float(has_uncertainty_estimates)
    score -= 0.2 * float(requires_external_closure)
    score -= 0.25 * float(model_extension_required)
    score -= 0.35 * float(model_scope == "thermodynamic_transition")
    score = min(1.0, max(0.0, score))
    overclaim_risk = horizon_class == "model_extension_required"

    return {
        "benchmark_id": benchmark_id,
        "source_key": source_key,
        "source_year": float(source_year),
        "model_scope": model_scope,
        "target_protocol": target_protocol,
        "available_observables": ";".join(available),
        "required_observables": ";".join(required),
        "missing_observables": ";".join(missing) if missing else "none",
        "computable_missing_observables": ";".join(computable_missing) if computable_missing else "none",
        "direct_observable_coverage": float(direct_coverage),
        "effective_observable_coverage": float(effective_coverage),
        "has_machine_readable_repository": float(has_machine_readable_repository),
        "has_uncertainty_estimates": float(has_uncertainty_estimates),
        "has_shared_transport_grid": float(has_shared_transport_grid),
        "requires_external_closure": float(requires_external_closure),
        "model_extension_required": float(model_extension_required),
        "can_compute_missing_from_trajectories": float(can_compute_missing),
        "frontier_priority_score": float(score),
        "primary_blocker": primary_blocker,
        "overclaim_risk": float(overclaim_risk),
        "horizon_class": horizon_class,
    }


def sota_source_provenance_gate(
    *,
    source_id: str,
    citation_key: str,
    source_type: str,
    model_scope: str,
    provenance_items: Sequence[str],
    supported_observables: Sequence[str],
    required_downstream_protocols: Sequence[str],
    has_reanalysis_permission: bool,
) -> dict[str, float | str]:
    """Classify whether a SOTA source is usable as data, trajectories, or citation only."""

    for name, value in {
        "source_id": source_id,
        "citation_key": citation_key,
        "source_type": source_type,
        "model_scope": model_scope,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    allowed_source_types = {"article", "dataset_repository", "code_repository", "mixed_release"}
    allowed_scopes = {
        "dynamical_signature",
        "transport_decoupling",
        "spatial_heterogeneity",
        "thermodynamic_transition",
    }
    if source_type not in allowed_source_types:
        raise ValueError("source_type is not recognized")
    if model_scope not in allowed_scopes:
        raise ValueError("model_scope is not recognized")
    if not provenance_items:
        raise ValueError("provenance_items must be nonempty")
    if not supported_observables:
        raise ValueError("supported_observables must be nonempty")
    if not required_downstream_protocols:
        raise ValueError("required_downstream_protocols must be nonempty")
    for name, values in {
        "provenance_items": provenance_items,
        "supported_observables": supported_observables,
        "required_downstream_protocols": required_downstream_protocols,
    }.items():
        if any(not value for value in values):
            raise ValueError(f"{name} must contain nonempty strings")

    provenance = list(dict.fromkeys(provenance_items))
    observables = list(dict.fromkeys(supported_observables))
    protocols = list(dict.fromkeys(required_downstream_protocols))
    provenance_set = set(provenance)
    observable_set = set(observables)

    has_identifier = bool({"doi", "repository_url"} & provenance_set)
    has_machine_readable_files = "machine_readable_files" in provenance_set
    has_raw_trajectories = "raw_particle_trajectories" in provenance_set or "particle_trajectories" in observable_set
    has_curve_tables = "observable_tables" in provenance_set or "machine_readable_curves" in provenance_set
    has_protocol_metadata = "simulation_protocol_metadata" in provenance_set or "experimental_protocol_metadata" in provenance_set
    has_license_or_terms = "license_or_terms" in provenance_set
    trajectory_protocol_requested = any(protocol.startswith("trajectory_") for protocol in protocols)
    raw_curve_protocol_requested = any("raw_curve" in protocol or "persistence_exchange" in protocol for protocol in protocols)

    can_enter_trajectory = (
        model_scope != "thermodynamic_transition"
        and trajectory_protocol_requested
        and has_identifier
        and has_machine_readable_files
        and has_raw_trajectories
        and has_protocol_metadata
        and has_license_or_terms
        and has_reanalysis_permission
    )
    can_enter_raw_curve = (
        model_scope != "thermodynamic_transition"
        and raw_curve_protocol_requested
        and has_identifier
        and has_machine_readable_files
        and has_curve_tables
        and has_protocol_metadata
        and has_reanalysis_permission
    )

    if model_scope == "thermodynamic_transition":
        stage = "scope_boundary_source"
        blocker = "renewal_dynamics_not_thermodynamic_theory"
    elif can_enter_trajectory:
        stage = "trajectory_reanalysis_source"
        blocker = "none"
    elif can_enter_raw_curve:
        stage = "raw_curve_reanalysis_source"
        blocker = "none"
    elif not has_machine_readable_files:
        stage = "citation_only_source"
        blocker = "machine_readable_files"
    elif not has_reanalysis_permission:
        stage = "machine_readable_but_not_reanalysis_permitted"
        blocker = "reanalysis_permission"
    elif not has_protocol_metadata:
        stage = "machine_readable_source_incomplete_metadata"
        blocker = "protocol_metadata"
    elif not has_license_or_terms:
        stage = "machine_readable_source_incomplete_metadata"
        blocker = "license_or_terms"
    elif trajectory_protocol_requested and not has_raw_trajectories:
        stage = "machine_readable_source_incomplete_metadata"
        blocker = "raw_particle_trajectories"
    elif raw_curve_protocol_requested and not has_curve_tables:
        stage = "machine_readable_source_incomplete_metadata"
        blocker = "observable_tables"
    else:
        stage = "citation_only_source"
        blocker = "machine_readable_files"

    requires_digitization = stage == "citation_only_source"
    return {
        "source_id": source_id,
        "citation_key": citation_key,
        "source_type": source_type,
        "model_scope": model_scope,
        "provenance_items": ";".join(provenance),
        "supported_observables": ";".join(observables),
        "required_downstream_protocols": ";".join(protocols),
        "has_identifier": float(has_identifier),
        "has_machine_readable_files": float(has_machine_readable_files),
        "has_raw_particle_trajectories": float(has_raw_trajectories),
        "has_observable_tables": float(has_curve_tables),
        "has_protocol_metadata": float(has_protocol_metadata),
        "has_license_or_terms": float(has_license_or_terms),
        "has_reanalysis_permission": float(has_reanalysis_permission),
        "can_enter_trajectory_protocol": float(can_enter_trajectory),
        "can_enter_raw_curve_protocol": float(can_enter_raw_curve),
        "requires_digitization": float(requires_digitization),
        "scope_boundary": float(model_scope == "thermodynamic_transition"),
        "primary_blocker": blocker,
        "provenance_stage": stage,
    }


def sota_data_accession_gate(
    *,
    accession_id: str,
    source_id: str,
    citation_key: str,
    model_scope: str,
    landing_url: str,
    doi: str,
    archive_name: str,
    archive_md5: str,
    archive_size_bytes: int,
    license_id: str,
    has_public_landing_page: bool,
    has_downloadable_archive: bool,
    has_schema_or_readme: bool,
    has_trajectory_files: bool,
    has_precomputed_descriptors: bool,
    local_cache_present: bool,
    intended_protocols: Sequence[str],
) -> dict[str, float | str]:
    """Gate public data accessions before local trajectory or raw-curve reanalysis."""

    for name, value in {
        "accession_id": accession_id,
        "source_id": source_id,
        "citation_key": citation_key,
        "model_scope": model_scope,
        "landing_url": landing_url,
        "doi": doi,
        "archive_name": archive_name,
        "archive_md5": archive_md5,
        "license_id": license_id,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    allowed_scopes = {
        "dynamical_signature",
        "transport_decoupling",
        "spatial_heterogeneity",
        "thermodynamic_transition",
    }
    if model_scope not in allowed_scopes:
        raise ValueError("model_scope is not recognized")
    if archive_size_bytes < 0:
        raise ValueError("archive_size_bytes must be nonnegative")
    if not intended_protocols:
        raise ValueError("intended_protocols must be nonempty")
    if any(not protocol for protocol in intended_protocols):
        raise ValueError("intended_protocols must contain nonempty strings")

    protocols = list(dict.fromkeys(intended_protocols))
    archive_size_gb = archive_size_bytes / 1_000_000_000.0
    has_doi = doi != "none" and "." in doi
    has_landing_url = landing_url.startswith("https://")
    has_archive_name = archive_name != "none"
    has_checksum = archive_md5 != "none" and archive_md5.startswith("md5:") is False and len(archive_md5) >= 16
    large_download = archive_size_bytes > 1_000_000_000
    trajectory_protocol_requested = any(protocol.startswith("trajectory_") for protocol in protocols)
    raw_curve_protocol_requested = any("raw_curve" in protocol or "persistence_exchange" in protocol for protocol in protocols)
    useful_data_payload = (
        trajectory_protocol_requested
        and has_trajectory_files
        or raw_curve_protocol_requested
        and has_precomputed_descriptors
    )
    accession_ready = (
        model_scope != "thermodynamic_transition"
        and has_public_landing_page
        and has_landing_url
        and has_doi
        and has_downloadable_archive
        and has_archive_name
        and has_checksum
        and archive_size_bytes > 0
        and license_id != "none"
        and has_schema_or_readme
        and useful_data_payload
    )
    ready_for_local_reanalysis = accession_ready and local_cache_present

    if model_scope == "thermodynamic_transition":
        stage = "scope_boundary_accession"
        blocker = "renewal_dynamics_not_thermodynamic_theory"
    elif ready_for_local_reanalysis and has_trajectory_files:
        stage = "local_trajectory_cache_ready"
        blocker = "none"
    elif ready_for_local_reanalysis:
        stage = "local_raw_curve_cache_ready"
        blocker = "none"
    elif accession_ready and has_trajectory_files:
        stage = "remote_trajectory_accession_ready"
        blocker = "local_cache"
    elif accession_ready:
        stage = "remote_raw_curve_accession_ready"
        blocker = "local_cache"
    elif not has_public_landing_page or not has_landing_url or not has_doi:
        stage = "citation_only_no_accession"
        blocker = "public_landing_page"
    elif not has_downloadable_archive or not has_archive_name:
        stage = "citation_only_no_accession"
        blocker = "downloadable_archive"
    elif not has_checksum:
        stage = "metadata_incomplete_accession"
        blocker = "archive_md5"
    elif license_id == "none":
        stage = "metadata_incomplete_accession"
        blocker = "license"
    elif not has_schema_or_readme:
        stage = "metadata_incomplete_accession"
        blocker = "schema_or_readme"
    elif not useful_data_payload:
        stage = "metadata_incomplete_accession"
        blocker = "trajectory_or_descriptor_payload"
    else:
        stage = "metadata_incomplete_accession"
        blocker = "accession_metadata"

    return {
        "accession_id": accession_id,
        "source_id": source_id,
        "citation_key": citation_key,
        "model_scope": model_scope,
        "landing_url": landing_url,
        "doi": doi,
        "archive_name": archive_name,
        "archive_md5": archive_md5,
        "archive_size_bytes": float(archive_size_bytes),
        "archive_size_gb": float(archive_size_gb),
        "license_id": license_id,
        "has_public_landing_page": float(has_public_landing_page),
        "has_downloadable_archive": float(has_downloadable_archive),
        "has_schema_or_readme": float(has_schema_or_readme),
        "has_trajectory_files": float(has_trajectory_files),
        "has_precomputed_descriptors": float(has_precomputed_descriptors),
        "local_cache_present": float(local_cache_present),
        "intended_protocols": ";".join(protocols),
        "large_download": float(large_download),
        "download_required": float(accession_ready and not local_cache_present),
        "accession_ready": float(accession_ready),
        "ready_for_local_reanalysis": float(ready_for_local_reanalysis),
        "scope_boundary": float(model_scope == "thermodynamic_transition"),
        "primary_blocker": blocker,
        "accession_stage": stage,
    }


def _strip_md5_prefix(value: str) -> str:
    return value[4:] if value.startswith("md5:") else value


def sota_zenodo_record_fingerprint_gate(
    *,
    fingerprint_id: str,
    accession_id: str,
    source_id: str,
    record: dict,
    expected_doi: str,
    expected_license_id: str,
    expected_archive_name: str,
    expected_archive_md5: str,
    expected_archive_size_bytes: int,
    expected_readme_name: str,
    expected_readme_md5: str,
    expected_readme_size_bytes: int,
    large_archive_threshold_bytes: int,
) -> dict[str, float | str]:
    """Verify a cached Zenodo API record before any local archive reanalysis."""

    for name, value in {
        "fingerprint_id": fingerprint_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "expected_doi": expected_doi,
        "expected_license_id": expected_license_id,
        "expected_archive_name": expected_archive_name,
        "expected_archive_md5": expected_archive_md5,
        "expected_readme_name": expected_readme_name,
        "expected_readme_md5": expected_readme_md5,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    for name, value in {
        "expected_archive_size_bytes": expected_archive_size_bytes,
        "expected_readme_size_bytes": expected_readme_size_bytes,
        "large_archive_threshold_bytes": large_archive_threshold_bytes,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if large_archive_threshold_bytes <= 0:
        raise ValueError("large_archive_threshold_bytes must be positive")

    metadata = record.get("metadata", {})
    license_value = metadata.get("license", "none") if isinstance(metadata, dict) else "none"
    if isinstance(license_value, dict):
        observed_license = str(license_value.get("id", "none"))
    else:
        observed_license = str(license_value)
    files = record.get("files", [])
    if not isinstance(files, list):
        files = []
    file_by_key = {
        str(entry.get("key", "")): entry
        for entry in files
        if isinstance(entry, dict) and entry.get("key")
    }
    archive = file_by_key.get(expected_archive_name, {})
    readme = file_by_key.get(expected_readme_name, {})
    observed_doi = str(record.get("doi", "none"))
    observed_archive_size = int(archive.get("size", -1)) if archive else -1
    observed_readme_size = int(readme.get("size", -1)) if readme else -1
    observed_archive_md5 = _strip_md5_prefix(str(archive.get("checksum", "none"))) if archive else "none"
    observed_readme_md5 = _strip_md5_prefix(str(readme.get("checksum", "none"))) if readme else "none"

    doi_matches = observed_doi == expected_doi
    license_matches = observed_license == expected_license_id
    archive_file_present = bool(archive)
    readme_file_present = bool(readme)
    archive_size_matches = observed_archive_size == expected_archive_size_bytes
    archive_md5_matches = observed_archive_md5 == _strip_md5_prefix(expected_archive_md5)
    readme_size_matches = observed_readme_size == expected_readme_size_bytes
    readme_md5_matches = observed_readme_md5 == _strip_md5_prefix(expected_readme_md5)
    large_archive = expected_archive_size_bytes > large_archive_threshold_bytes
    ready = (
        doi_matches
        and license_matches
        and archive_file_present
        and readme_file_present
        and archive_size_matches
        and archive_md5_matches
        and readme_size_matches
        and readme_md5_matches
    )
    if ready:
        stage = "zenodo_record_verified"
        blocker = "archive_cache" if large_archive else "local_reanalysis"
    else:
        stage = "zenodo_record_mismatch"
        checks = [
            ("doi", doi_matches),
            ("license", license_matches),
            ("archive_file", archive_file_present),
            ("readme_file", readme_file_present),
            ("archive_size", archive_size_matches),
            ("archive_md5", archive_md5_matches),
            ("readme_size", readme_size_matches),
            ("readme_md5", readme_md5_matches),
        ]
        blocker = next(name for name, ok in checks if not ok)

    return {
        "fingerprint_id": fingerprint_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "record_id": float(record.get("id", 0) or 0),
        "doi": observed_doi,
        "license_id": observed_license,
        "archive_name": expected_archive_name,
        "archive_size_bytes": float(observed_archive_size),
        "archive_md5": observed_archive_md5,
        "readme_name": expected_readme_name,
        "readme_size_bytes": float(observed_readme_size),
        "readme_md5": observed_readme_md5,
        "doi_matches": float(doi_matches),
        "license_matches": float(license_matches),
        "archive_file_present": float(archive_file_present),
        "readme_file_present": float(readme_file_present),
        "archive_size_matches": float(archive_size_matches),
        "archive_md5_matches": float(archive_md5_matches),
        "readme_size_matches": float(readme_size_matches),
        "readme_md5_matches": float(readme_md5_matches),
        "large_archive": float(large_archive),
        "full_archive_download_required": float(large_archive),
        "zenodo_record_fingerprint_ready": float(ready),
        "real_reanalysis_ready": 0.0,
        "primary_blocker": blocker,
        "fingerprint_stage": stage,
    }


def sota_archive_preflight_gate(
    *,
    preflight_id: str,
    accession_id: str,
    source_id: str,
    archive_name: str,
    archive_size_bytes: int,
    archive_md5: str,
    readme_name: str,
    readme_size_bytes: int,
    readme_md5: str,
    max_automatic_download_bytes: int,
    full_archive_download_approved: bool,
    local_readme_present: bool,
    local_archive_present: bool,
    required_schema_tokens: Sequence[str],
    observed_schema_tokens: Sequence[str],
    required_local_fields: Sequence[str],
    available_local_fields: Sequence[str],
) -> dict[str, float | str]:
    """Preflight a public trajectory archive before claiming local reanalysis."""

    for name, value in {
        "preflight_id": preflight_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "archive_name": archive_name,
        "archive_md5": archive_md5,
        "readme_name": readme_name,
        "readme_md5": readme_md5,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    for name, value in {
        "archive_size_bytes": archive_size_bytes,
        "readme_size_bytes": readme_size_bytes,
        "max_automatic_download_bytes": max_automatic_download_bytes,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if max_automatic_download_bytes <= 0:
        raise ValueError("max_automatic_download_bytes must be positive")
    if not required_schema_tokens:
        raise ValueError("required_schema_tokens must be nonempty")
    if not required_local_fields:
        raise ValueError("required_local_fields must be nonempty")
    for name, values in {
        "required_schema_tokens": required_schema_tokens,
        "observed_schema_tokens": observed_schema_tokens,
        "required_local_fields": required_local_fields,
        "available_local_fields": available_local_fields,
    }.items():
        if any(not value for value in values):
            raise ValueError(f"{name} must contain nonempty strings")

    schema_required = list(dict.fromkeys(required_schema_tokens))
    schema_observed = list(dict.fromkeys(observed_schema_tokens))
    fields_required = list(dict.fromkeys(required_local_fields))
    fields_available = list(dict.fromkeys(available_local_fields))
    missing_schema = [token for token in schema_required if token not in set(schema_observed)]
    missing_fields = [field for field in fields_required if field not in set(fields_available)]

    archive_checksum_available = archive_md5 != "none" and len(archive_md5) >= 16
    readme_checksum_available = readme_md5 != "none" and len(readme_md5) >= 16
    large_archive = archive_size_bytes > max_automatic_download_bytes
    readme_download_allowed = readme_size_bytes <= max_automatic_download_bytes
    full_download_allowed = (not large_archive) or bool(full_archive_download_approved)
    schema_ready = bool(schema_required) and not missing_schema and readme_checksum_available
    ready_for_readme_cache = schema_ready and readme_download_allowed
    ready_for_local = (
        local_archive_present
        and archive_checksum_available
        and schema_ready
        and not missing_fields
    )

    if not archive_checksum_available:
        stage = "archive_checksum_missing"
        blocker = "archive_md5"
    elif not readme_checksum_available:
        stage = "readme_checksum_missing"
        blocker = "readme_md5"
    elif missing_schema:
        stage = "readme_schema_incomplete"
        blocker = "schema_tokens"
    elif ready_for_local:
        stage = "local_archive_reanalysis_ready"
        blocker = "none"
    elif local_archive_present and missing_fields:
        stage = "local_adapter_contract_incomplete"
        blocker = missing_fields[0]
    elif large_archive and not full_archive_download_approved:
        stage = "large_archive_approval_required"
        blocker = "large_archive_download_approval"
    elif full_download_allowed:
        stage = "archive_download_ready"
        blocker = "local_archive"
    else:
        stage = "archive_download_blocked"
        blocker = "download_policy"

    return {
        "preflight_id": preflight_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "archive_name": archive_name,
        "archive_size_bytes": float(archive_size_bytes),
        "archive_size_gb": archive_size_bytes / 1_000_000_000.0,
        "archive_md5": archive_md5,
        "readme_name": readme_name,
        "readme_size_bytes": float(readme_size_bytes),
        "readme_md5": readme_md5,
        "max_automatic_download_bytes": float(max_automatic_download_bytes),
        "large_archive": float(large_archive),
        "readme_download_allowed": float(readme_download_allowed),
        "full_archive_download_allowed": float(full_download_allowed),
        "full_archive_download_approved": float(full_archive_download_approved),
        "local_readme_present": float(local_readme_present),
        "local_archive_present": float(local_archive_present),
        "required_schema_tokens": ";".join(schema_required),
        "observed_schema_tokens": ";".join(schema_observed) if schema_observed else "none",
        "missing_schema_tokens": ";".join(missing_schema) if missing_schema else "none",
        "required_local_fields": ";".join(fields_required),
        "available_local_fields": ";".join(fields_available) if fields_available else "none",
        "missing_local_fields": ";".join(missing_fields) if missing_fields else "none",
        "schema_token_coverage": (
            (len(schema_required) - len(missing_schema)) / len(schema_required)
        ),
        "local_field_coverage": (
            (len(fields_required) - len(missing_fields)) / len(fields_required)
        ),
        "ready_for_readme_schema_cache": float(ready_for_readme_cache),
        "ready_for_local_reanalysis": float(ready_for_local),
        "primary_blocker": blocker,
        "preflight_stage": stage,
    }


def sota_readme_digest_gate(
    *,
    digest_id: str,
    accession_id: str,
    source_id: str,
    readme_text: str,
    expected_size_bytes: int,
    expected_md5: str,
    required_tokens: Sequence[str],
    required_citation_dois: Sequence[str],
    required_license_phrase: str,
    local_cache_path: str,
) -> dict[str, float | str]:
    """Verify a locally cached README against expected digest and schema tokens."""

    for name, value in {
        "digest_id": digest_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "readme_text": readme_text,
        "expected_md5": expected_md5,
        "required_license_phrase": required_license_phrase,
        "local_cache_path": local_cache_path,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    if expected_size_bytes < 0:
        raise ValueError("expected_size_bytes must be nonnegative")
    if not required_tokens:
        raise ValueError("required_tokens must be nonempty")
    if not required_citation_dois:
        raise ValueError("required_citation_dois must be nonempty")
    for name, values in {
        "required_tokens": required_tokens,
        "required_citation_dois": required_citation_dois,
    }.items():
        if any(not value for value in values):
            raise ValueError(f"{name} must contain nonempty strings")

    readme_bytes = readme_text.encode("utf-8")
    observed_size = len(readme_bytes)
    observed_md5 = hashlib.md5(readme_bytes).hexdigest()
    expected_md5_value = observed_md5 if expected_md5 == "use-computed" else expected_md5
    tokens = list(dict.fromkeys(required_tokens))
    citations = list(dict.fromkeys(required_citation_dois))
    missing_tokens = [token for token in tokens if token not in readme_text]
    missing_citations = [doi for doi in citations if doi not in readme_text]
    license_present = required_license_phrase.lower() in readme_text.lower()
    size_matches = observed_size == expected_size_bytes
    md5_matches = observed_md5 == expected_md5_value
    token_coverage = (len(tokens) - len(missing_tokens)) / len(tokens)
    citation_coverage = (len(citations) - len(missing_citations)) / len(citations)
    ready = (
        size_matches
        and md5_matches
        and not missing_tokens
        and not missing_citations
        and license_present
    )

    if ready:
        stage = "readme_digest_verified"
        blocker = "none"
    elif not size_matches:
        stage = "readme_size_mismatch"
        blocker = "readme_size_bytes"
    elif not md5_matches:
        stage = "readme_md5_mismatch"
        blocker = "readme_md5"
    elif missing_tokens:
        stage = "schema_tokens_incomplete"
        blocker = "schema_tokens"
    elif missing_citations:
        stage = "citation_guidance_incomplete"
        blocker = "citation_dois"
    else:
        stage = "license_guidance_incomplete"
        blocker = "license_phrase"

    return {
        "digest_id": digest_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "local_cache_path": local_cache_path,
        "observed_size_bytes": float(observed_size),
        "expected_size_bytes": float(expected_size_bytes),
        "observed_md5": observed_md5,
        "expected_md5": expected_md5_value,
        "size_matches_expected": float(size_matches),
        "md5_matches_expected": float(md5_matches),
        "required_tokens": ";".join(tokens),
        "missing_tokens": ";".join(missing_tokens) if missing_tokens else "none",
        "required_citation_dois": ";".join(citations),
        "missing_citation_dois": ";".join(missing_citations) if missing_citations else "none",
        "required_license_phrase": required_license_phrase,
        "schema_token_coverage": float(token_coverage),
        "citation_coverage": float(citation_coverage),
        "license_phrase_present": float(license_present),
        "readme_digest_ready": float(ready),
        "primary_blocker": blocker,
        "digest_stage": stage,
    }


def sota_local_cache_verification_gate(
    *,
    cache_id: str,
    accession_id: str,
    source_id: str,
    readme_path: str | Path,
    expected_readme_size_bytes: int,
    expected_readme_md5: str,
    archive_path: str | Path,
    expected_archive_size_bytes: int,
    expected_archive_md5: str,
) -> dict[str, float | str]:
    """Verify local README and archive cache files before trajectory reanalysis."""

    for name, value in {
        "cache_id": cache_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "expected_readme_md5": expected_readme_md5,
        "expected_archive_md5": expected_archive_md5,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    for name, value in {
        "expected_readme_size_bytes": expected_readme_size_bytes,
        "expected_archive_size_bytes": expected_archive_size_bytes,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")

    readme = Path(readme_path)
    archive = Path(archive_path)

    def file_status(path: Path, expected_size: int, expected_md5: str) -> dict[str, float | str]:
        if not path.exists():
            expected_md5_value = "missing" if expected_md5 == "use-computed" else expected_md5
            return {
                "present": 0.0,
                "observed_size_bytes": 0.0,
                "expected_size_bytes": float(expected_size),
                "observed_md5": "missing",
                "expected_md5": expected_md5_value,
                "size_matches": 0.0,
                "md5_matches": 0.0,
                "verified": 0.0,
            }
        file_bytes = path.read_bytes()
        observed_size = len(file_bytes)
        observed_md5 = hashlib.md5(file_bytes).hexdigest()
        expected_md5_value = observed_md5 if expected_md5 == "use-computed" else expected_md5
        size_matches = observed_size == expected_size
        md5_matches = observed_md5 == expected_md5_value
        return {
            "present": 1.0,
            "observed_size_bytes": float(observed_size),
            "expected_size_bytes": float(expected_size),
            "observed_md5": observed_md5,
            "expected_md5": expected_md5_value,
            "size_matches": float(size_matches),
            "md5_matches": float(md5_matches),
            "verified": float(size_matches and md5_matches),
        }

    readme_status = file_status(readme, expected_readme_size_bytes, expected_readme_md5)
    archive_status = file_status(archive, expected_archive_size_bytes, expected_archive_md5)
    readme_verified = readme_status["verified"] == 1.0
    archive_verified = archive_status["verified"] == 1.0
    local_verified = readme_verified and archive_verified

    if not bool(readme_status["present"]):
        stage = "readme_cache_missing"
        blocker = "readme_path"
    elif not readme_verified:
        stage = "readme_cache_mismatch"
        blocker = "readme_md5" if readme_status["md5_matches"] == 0.0 else "readme_size_bytes"
    elif not bool(archive_status["present"]):
        stage = "archive_cache_missing"
        blocker = "archive_path"
    elif not archive_verified:
        stage = "archive_cache_mismatch"
        blocker = "archive_md5" if archive_status["md5_matches"] == 0.0 else "archive_size_bytes"
    else:
        stage = "local_archive_cache_verified"
        blocker = "none"

    return {
        "cache_id": cache_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "readme_path": str(readme),
        "archive_path": str(archive),
        "readme_present": readme_status["present"],
        "archive_present": archive_status["present"],
        "observed_readme_size_bytes": readme_status["observed_size_bytes"],
        "expected_readme_size_bytes": readme_status["expected_size_bytes"],
        "observed_readme_md5": readme_status["observed_md5"],
        "expected_readme_md5": readme_status["expected_md5"],
        "readme_size_matches": readme_status["size_matches"],
        "readme_md5_matches": readme_status["md5_matches"],
        "readme_cache_verified": readme_status["verified"],
        "observed_archive_size_bytes": archive_status["observed_size_bytes"],
        "expected_archive_size_bytes": archive_status["expected_size_bytes"],
        "observed_archive_md5": archive_status["observed_md5"],
        "expected_archive_md5": archive_status["expected_md5"],
        "archive_size_matches": archive_status["size_matches"],
        "archive_md5_matches": archive_status["md5_matches"],
        "archive_cache_verified": archive_status["verified"],
        "local_cache_verified": float(local_verified),
        "ready_for_local_reanalysis": float(local_verified),
        "primary_blocker": blocker,
        "cache_stage": stage,
    }


def sota_zip_structure_gate(
    *,
    structure_id: str,
    accession_id: str,
    source_id: str,
    archive_path: str | Path,
    required_roots: Sequence[str],
) -> dict[str, float | str]:
    """Inspect a zip central directory for required dataset roots without extraction."""

    for name, value in {
        "structure_id": structure_id,
        "accession_id": accession_id,
        "source_id": source_id,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    if not required_roots:
        raise ValueError("required_roots must be nonempty")
    if any(not root for root in required_roots):
        raise ValueError("required_roots must contain nonempty strings")

    archive = Path(archive_path)
    roots = [root.strip("/") for root in dict.fromkeys(required_roots)]
    if not archive.exists():
        return {
            "structure_id": structure_id,
            "accession_id": accession_id,
            "source_id": source_id,
            "archive_path": str(archive),
            "zip_present": 0.0,
            "zip_readable": 0.0,
            "entry_count": 0.0,
            "required_roots": ";".join(roots),
            "present_roots": "none",
            "missing_roots": ";".join(roots),
            "root_coverage": 0.0,
            "zip_structure_ready": 0.0,
            "primary_blocker": "archive_path",
            "zip_structure_stage": "zip_archive_missing",
        }

    try:
        with zipfile.ZipFile(archive) as zf:
            entries = [name.strip("/") for name in zf.namelist() if name.strip("/")]
    except zipfile.BadZipFile:
        return {
            "structure_id": structure_id,
            "accession_id": accession_id,
            "source_id": source_id,
            "archive_path": str(archive),
            "zip_present": 1.0,
            "zip_readable": 0.0,
            "entry_count": 0.0,
            "required_roots": ";".join(roots),
            "present_roots": "none",
            "missing_roots": ";".join(roots),
            "root_coverage": 0.0,
            "zip_structure_ready": 0.0,
            "primary_blocker": "zipfile",
            "zip_structure_stage": "zip_unreadable",
        }

    present_roots = [
        root
        for root in roots
        if any(entry == root or entry.startswith(f"{root}/") for entry in entries)
    ]
    missing_roots = [root for root in roots if root not in set(present_roots)]
    ready = not missing_roots
    return {
        "structure_id": structure_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "archive_path": str(archive),
        "zip_present": 1.0,
        "zip_readable": 1.0,
        "entry_count": float(len(entries)),
        "required_roots": ";".join(roots),
        "present_roots": ";".join(present_roots) if present_roots else "none",
        "missing_roots": ";".join(missing_roots) if missing_roots else "none",
        "root_coverage": (len(roots) - len(missing_roots)) / len(roots),
        "zip_structure_ready": float(ready),
        "primary_blocker": "none" if ready else "archive_roots",
        "zip_structure_stage": "zip_structure_ready"
        if ready
        else "zip_structure_incomplete",
    }


def sota_remote_zip_central_directory_gate(
    *,
    remote_structure_id: str,
    accession_id: str,
    source_id: str,
    manifest: dict,
    expected_archive_size_bytes: int,
    required_roots: Sequence[str],
    full_archive_cached: bool,
) -> dict[str, float | str]:
    """Verify a remote ZIP central-directory manifest without downloading payloads."""

    for name, value in {
        "remote_structure_id": remote_structure_id,
        "accession_id": accession_id,
        "source_id": source_id,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    if expected_archive_size_bytes < 0:
        raise ValueError("expected_archive_size_bytes must be nonnegative")
    if not required_roots:
        raise ValueError("required_roots must be nonempty")
    if any(not root for root in required_roots):
        raise ValueError("required_roots must contain nonempty strings")

    roots = [root.strip("/") for root in dict.fromkeys(required_roots)]
    entries_value = manifest.get("entries", [])
    entries = [
        str(entry).strip("/")
        for entry in entries_value
        if str(entry).strip("/")
    ] if isinstance(entries_value, list) else []
    range_supported = bool(manifest.get("range_supported", False))
    archive_size = int(manifest.get("archive_size_bytes", -1))
    size_matches = archive_size == expected_archive_size_bytes
    tail_probe_bytes = int(manifest.get("tail_probe_bytes", 0) or 0)
    cd_size = int(manifest.get("central_directory_size_bytes", 0) or 0)
    cd_offset = int(manifest.get("central_directory_offset", -1) or -1)
    entry_count = int(manifest.get("entry_count", len(entries)) or 0)
    zip64 = bool(manifest.get("zip64", False))
    archive_url = str(manifest.get("archive_url", "none"))

    present_roots = [
        root
        for root in roots
        if any(entry == root or entry.startswith(f"{root}/") for entry in entries)
    ]
    missing_roots = [root for root in roots if root not in set(present_roots)]
    structure_ready = (
        range_supported
        and size_matches
        and tail_probe_bytes > 0
        and cd_size > 0
        and cd_offset >= 0
        and entry_count > 0
        and not missing_roots
    )

    if not range_supported:
        stage = "remote_range_unavailable"
        blocker = "remote_range"
    elif not size_matches:
        stage = "remote_archive_size_mismatch"
        blocker = "archive_size_bytes"
    elif tail_probe_bytes <= 0 or cd_size <= 0 or cd_offset < 0:
        stage = "remote_central_directory_missing"
        blocker = "central_directory"
    elif entry_count <= 0:
        stage = "remote_central_directory_empty"
        blocker = "entry_count"
    elif missing_roots:
        stage = "remote_zip_structure_incomplete"
        blocker = "archive_roots"
    elif full_archive_cached:
        stage = "remote_zip_structure_and_cache_ready"
        blocker = "local_adapter"
    else:
        stage = "remote_zip_structure_verified"
        blocker = "archive_cache"

    return {
        "remote_structure_id": remote_structure_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "archive_url": archive_url,
        "archive_size_bytes": float(archive_size),
        "expected_archive_size_bytes": float(expected_archive_size_bytes),
        "archive_size_matches": float(size_matches),
        "range_supported": float(range_supported),
        "zip64": float(zip64),
        "tail_probe_bytes": float(tail_probe_bytes),
        "central_directory_size_bytes": float(cd_size),
        "central_directory_offset": float(cd_offset),
        "entry_count": float(entry_count),
        "required_roots": ";".join(roots),
        "present_roots": ";".join(present_roots) if present_roots else "none",
        "missing_roots": ";".join(missing_roots) if missing_roots else "none",
        "root_coverage": (len(roots) - len(missing_roots)) / len(roots),
        "full_archive_cached": float(full_archive_cached),
        "remote_zip_structure_ready": float(structure_ready),
        "real_reanalysis_ready": 0.0,
        "primary_blocker": blocker,
        "remote_zip_structure_stage": stage,
    }


def _glassbench_temperature_tokens(path: str) -> set[str]:
    tokens = set(re.findall(r"T(\d+\.\d+)", path))
    tokens.update(re.findall(r"times_(\d+\.\d+)", path))
    return tokens


def sota_glassbench_payload_index_gate(
    *,
    payload_index_id: str,
    accession_id: str,
    source_id: str,
    manifest: dict,
    systems: Sequence[str],
    full_archive_cached: bool,
) -> list[dict[str, float | str]]:
    """Index GlassBench system payloads from a remote ZIP central-directory manifest."""

    for name, value in {
        "payload_index_id": payload_index_id,
        "accession_id": accession_id,
        "source_id": source_id,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    if not systems:
        raise ValueError("systems must be nonempty")
    if any(not system for system in systems):
        raise ValueError("systems must contain nonempty strings")

    entries_value = manifest.get("entries", [])
    entries = [
        str(entry).strip("/")
        for entry in entries_value
        if str(entry).strip("/")
    ] if isinstance(entries_value, list) else []
    rows: list[dict[str, float | str]] = []
    for system in dict.fromkeys(systems):
        trajectory_prefix = f"GlassBench/{system}_trajectories/"
        model_prefix = f"GlassBench/{system}_models/"
        result_prefix = f"GlassBench/{system}_results/"
        trajectory_entries = [
            entry
            for entry in entries
            if entry.startswith(trajectory_prefix)
            and (entry.endswith(".tar.xz") or entry.endswith(".tar.gz") or entry.endswith(".zip"))
        ]
        model_entries = [
            entry
            for entry in entries
            if entry.startswith(model_prefix)
            and (entry.endswith(".tar.gz") or entry.endswith(".tar.xz") or entry.endswith(".zip"))
        ]
        result_entries = [
            entry
            for entry in entries
            if entry.startswith(result_prefix)
            and (entry.endswith(".dat") or entry.endswith(".g") or entry.endswith(".csv"))
        ]
        trajectory_temperatures = sorted(
            {token for entry in trajectory_entries for token in _glassbench_temperature_tokens(entry)},
            key=float,
        )
        model_temperatures = sorted(
            {token for entry in model_entries for token in _glassbench_temperature_tokens(entry)},
            key=float,
        )
        result_temperatures = sorted(
            {token for entry in result_entries for token in _glassbench_temperature_tokens(entry)},
            key=float,
        )
        common_temperatures = sorted(
            set(trajectory_temperatures) & set(model_temperatures) & set(result_temperatures),
            key=float,
        )
        common_model_result_temperatures = sorted(
            set(model_temperatures) & set(result_temperatures),
            key=float,
        )
        has_trajectory = bool(trajectory_entries)
        has_model = bool(model_entries)
        has_result = bool(result_entries)
        payload_ready = has_trajectory and has_model and has_result and bool(common_temperatures)
        model_result_ready = has_model and has_result and bool(common_model_result_temperatures)

        if payload_ready and full_archive_cached:
            stage = "local_payload_index_ready"
            blocker = "local_adapter"
        elif payload_ready:
            stage = "remote_payload_index_verified"
            blocker = "archive_cache"
        elif not has_trajectory:
            stage = "remote_payload_missing_trajectory"
            blocker = "trajectory_payload"
        elif not has_model:
            stage = "remote_payload_missing_model"
            blocker = "model_payload"
        elif not has_result:
            stage = "remote_payload_missing_results"
            blocker = "result_curves"
        else:
            stage = "remote_payload_temperature_mismatch"
            blocker = "temperature_grid"

        rows.append(
            {
                "payload_index_id": f"{payload_index_id}_{system.lower()}",
                "accession_id": accession_id,
                "source_id": source_id,
                "system_id": system,
                "trajectory_payload_count": float(len(trajectory_entries)),
                "model_payload_count": float(len(model_entries)),
                "result_curve_count": float(len(result_entries)),
                "trajectory_temperatures": ";".join(trajectory_temperatures)
                if trajectory_temperatures
                else "none",
                "model_temperatures": ";".join(model_temperatures) if model_temperatures else "none",
                "result_temperatures": ";".join(result_temperatures) if result_temperatures else "none",
                "common_temperatures": ";".join(common_temperatures) if common_temperatures else "none",
                "common_temperature_count": float(len(common_temperatures)),
                "common_model_result_temperatures": ";".join(common_model_result_temperatures)
                if common_model_result_temperatures
                else "none",
                "model_result_index_ready": float(model_result_ready),
                "full_archive_cached": float(full_archive_cached),
                "payload_index_ready": float(payload_ready),
                "real_reanalysis_ready": 0.0,
                "primary_blocker": blocker,
                "payload_stage": stage,
            }
        )
    return rows


def _glassbench_payload_format(path: str) -> str:
    for suffix, label in [
        (".tar.xz", "tar.xz"),
        (".tar.gz", "tar.gz"),
        (".zip", "zip"),
        (".gsd", "gsd"),
        (".xyz", "xyz"),
    ]:
        if path.endswith(suffix):
            return label
    return "unknown"


def sota_glassbench_trajectory_payload_locator_gate(
    *,
    locator_id: str,
    accession_id: str,
    source_id: str,
    manifest: dict,
    systems: Sequence[str],
    full_archive_cached: bool,
    entry_metadata_ready: bool,
) -> list[dict[str, float | str]]:
    """Locate remote GlassBench trajectory payload files before any large download."""

    for name, value in {
        "locator_id": locator_id,
        "accession_id": accession_id,
        "source_id": source_id,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    if not systems:
        raise ValueError("systems must be nonempty")

    entries_value = manifest.get("entries", [])
    entries = [
        str(entry).strip("/")
        for entry in entries_value
        if str(entry).strip("/")
    ] if isinstance(entries_value, list) else []
    archive_url = str(manifest.get("archive_url", "unknown"))
    range_supported = bool(manifest.get("range_supported", False))

    rows: list[dict[str, float | str]] = []
    for system in dict.fromkeys(systems):
        prefix = f"GlassBench/{system}_trajectories/"
        trajectory_entries = sorted(
            entry
            for entry in entries
            if entry.startswith(prefix)
            and (entry.endswith(".tar.xz") or entry.endswith(".tar.gz") or entry.endswith(".zip"))
        )
        if not trajectory_entries:
            rows.append(
                {
                    "locator_id": f"{locator_id}_{system.lower()}_missing",
                    "accession_id": accession_id,
                    "source_id": source_id,
                    "archive_url": archive_url,
                    "system_id": system,
                    "temperature": "none",
                    "source_path": "none",
                    "payload_format": "none",
                    "remote_payload_located": 0.0,
                    "range_supported": float(range_supported),
                    "entry_metadata_ready": float(entry_metadata_ready),
                    "full_archive_cached": float(full_archive_cached),
                    "range_fetch_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": "trajectory_payload",
                    "locator_stage": "remote_trajectory_payload_missing",
                }
            )
            continue

        for entry in trajectory_entries:
            temperatures = sorted(_glassbench_temperature_tokens(entry), key=float)
            temperature = temperatures[0] if temperatures else "unknown"
            range_fetch_ready = range_supported and entry_metadata_ready
            if full_archive_cached:
                stage = "local_trajectory_payload_available"
                blocker = "local_adapter"
            elif range_fetch_ready:
                stage = "remote_trajectory_payload_range_fetch_ready"
                blocker = "local_archive_slice"
            else:
                stage = "remote_trajectory_payload_located"
                blocker = "zip_entry_metadata"
            rows.append(
                {
                    "locator_id": f"{locator_id}_{system.lower()}_t{temperature.replace('.', '_')}",
                    "accession_id": accession_id,
                    "source_id": source_id,
                    "archive_url": archive_url,
                    "system_id": system,
                    "temperature": temperature,
                    "source_path": entry,
                    "payload_format": _glassbench_payload_format(entry),
                    "remote_payload_located": 1.0,
                    "range_supported": float(range_supported),
                    "entry_metadata_ready": float(entry_metadata_ready),
                    "full_archive_cached": float(full_archive_cached),
                    "range_fetch_ready": float(range_fetch_ready),
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": blocker,
                    "locator_stage": stage,
                }
            )
    return rows


def sota_glassbench_trajectory_entry_metadata_gate(
    *,
    metadata_id: str,
    accession_id: str,
    locator_rows: Sequence[dict[str, float | str]],
    metadata_manifest: dict,
    max_policy_member_bytes: int,
) -> list[dict[str, float | str]]:
    """Verify ZIP-entry metadata for located trajectory payloads without fetching members."""

    if not metadata_id:
        raise ValueError("metadata_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if max_policy_member_bytes <= 0:
        raise ValueError("max_policy_member_bytes must be positive")

    entries_value = metadata_manifest.get("entries", [])
    metadata_entries = (
        [entry for entry in entries_value if isinstance(entry, dict)]
        if isinstance(entries_value, list)
        else []
    )
    metadata_by_path = {str(entry.get("path", "")): entry for entry in metadata_entries if entry.get("path")}

    rows: list[dict[str, float | str]] = []
    for locator in locator_rows:
        system_id = str(locator.get("system_id", "unknown"))
        temperature = str(locator.get("temperature", "none"))
        source_path = str(locator.get("source_path", "none"))
        located = bool(float(locator.get("remote_payload_located", 0.0)))
        if not located or source_path == "none":
            rows.append(
                {
                    "metadata_id": f"{metadata_id}_{system_id.lower()}_{temperature}",
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "compression_method": "none",
                    "crc32": "none",
                    "compressed_size_bytes": 0.0,
                    "uncompressed_size_bytes": 0.0,
                    "local_header_offset": -1.0,
                    "compressed_data_range_start": -1.0,
                    "compressed_data_range_end": -1.0,
                    "entry_metadata_ready": 0.0,
                    "local_header_verified": 0.0,
                    "full_member_fetch_within_policy": 0.0,
                    "trajectory_extraction_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": "trajectory_payload",
                    "metadata_stage": "trajectory_payload_missing",
                }
            )
            continue

        entry = metadata_by_path.get(source_path)
        if entry is None:
            rows.append(
                {
                    "metadata_id": f"{metadata_id}_{system_id.lower()}_{temperature}",
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "compression_method": "unknown",
                    "crc32": "none",
                    "compressed_size_bytes": 0.0,
                    "uncompressed_size_bytes": 0.0,
                    "local_header_offset": -1.0,
                    "compressed_data_range_start": -1.0,
                    "compressed_data_range_end": -1.0,
                    "entry_metadata_ready": 0.0,
                    "local_header_verified": 0.0,
                    "full_member_fetch_within_policy": 0.0,
                    "trajectory_extraction_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": "entry_metadata",
                    "metadata_stage": "trajectory_entry_metadata_missing",
                }
            )
            continue

        compressed_size = int(entry.get("compressed_size_bytes", 0) or 0)
        uncompressed_size = int(entry.get("uncompressed_size_bytes", 0) or 0)
        local_header_offset = int(entry.get("local_header_offset", -1) or -1)
        range_start = int(entry.get("compressed_data_range_start", -1) or -1)
        range_end = int(entry.get("compressed_data_range_end", -1) or -1)
        local_header_verified = bool(entry.get("local_header_verified", False))
        metadata_ready = (
            compressed_size > 0
            and uncompressed_size > 0
            and local_header_offset >= 0
            and range_start >= 0
            and range_end >= range_start
            and bool(str(entry.get("crc32", "")))
            and local_header_verified
        )
        within_policy = metadata_ready and compressed_size <= max_policy_member_bytes

        if not metadata_ready:
            stage = "trajectory_entry_metadata_incomplete"
            blocker = "entry_metadata"
        elif not within_policy:
            stage = "trajectory_entry_metadata_ready_payload_size_blocked"
            blocker = "member_payload_size_policy"
        else:
            stage = "trajectory_entry_metadata_ready_fetch_policy_ready"
            blocker = "member_payload_cache"

        rows.append(
            {
                "metadata_id": f"{metadata_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_path": source_path,
                "compression_method": str(entry.get("compression_method", "unknown")),
                "crc32": str(entry.get("crc32", "none")),
                "compressed_size_bytes": float(compressed_size),
                "uncompressed_size_bytes": float(uncompressed_size),
                "local_header_offset": float(local_header_offset),
                "compressed_data_range_start": float(range_start),
                "compressed_data_range_end": float(range_end),
                "entry_metadata_ready": float(metadata_ready),
                "local_header_verified": float(local_header_verified),
                "full_member_fetch_within_policy": float(within_policy),
                "trajectory_extraction_ready": 0.0,
                "real_reanalysis_ready": 0.0,
                "primary_blocker": blocker,
                "metadata_stage": stage,
            }
        )
    return rows


def sota_glassbench_trajectory_member_stream_probe_gate(
    *,
    probe_id: str,
    accession_id: str,
    metadata_rows: Sequence[dict[str, float | str]],
    probe_manifest: dict,
) -> list[dict[str, float | str]]:
    """Verify small deflated-member probes before attempting trajectory extraction."""

    if not probe_id:
        raise ValueError("probe_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")

    entries_value = probe_manifest.get("entries", [])
    probe_entries = (
        [entry for entry in entries_value if isinstance(entry, dict)]
        if isinstance(entries_value, list)
        else []
    )
    probe_by_path = {str(entry.get("path", "")): entry for entry in probe_entries if entry.get("path")}

    rows: list[dict[str, float | str]] = []
    for metadata in metadata_rows:
        system_id = str(metadata.get("system_id", "unknown"))
        temperature = str(metadata.get("temperature", "none"))
        source_path = str(metadata.get("source_path", "none"))
        entry_ready = bool(float(metadata.get("entry_metadata_ready", 0.0)))
        if not entry_ready or source_path == "none":
            rows.append(
                {
                    "probe_id": f"{probe_id}_{system_id.lower()}_{temperature}",
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "compressed_probe_range_start": -1.0,
                    "compressed_probe_range_end": -1.0,
                    "compressed_probe_bytes": 0.0,
                    "compressed_probe_md5": "none",
                    "inflated_prefix_bytes": 0.0,
                    "inflated_prefix_hex": "none",
                    "stream_inflate_ready": 0.0,
                    "xz_magic_verified": 0.0,
                    "member_prefix_verified": 0.0,
                    "trajectory_extraction_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": str(metadata.get("primary_blocker", "entry_metadata")),
                    "probe_stage": "trajectory_entry_metadata_incomplete",
                }
            )
            continue

        entry = probe_by_path.get(source_path)
        if entry is None:
            rows.append(
                {
                    "probe_id": f"{probe_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "compressed_probe_range_start": -1.0,
                    "compressed_probe_range_end": -1.0,
                    "compressed_probe_bytes": 0.0,
                    "compressed_probe_md5": "none",
                    "inflated_prefix_bytes": 0.0,
                    "inflated_prefix_hex": "none",
                    "stream_inflate_ready": 0.0,
                    "xz_magic_verified": 0.0,
                    "member_prefix_verified": 0.0,
                    "trajectory_extraction_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": "member_stream_probe",
                    "probe_stage": "trajectory_member_stream_probe_missing",
                }
            )
            continue

        compressed_probe_bytes = int(entry.get("compressed_probe_bytes", 0) or 0)
        inflated_prefix_bytes = int(entry.get("inflated_prefix_bytes", 0) or 0)
        stream_ready = bool(entry.get("stream_inflate_ready", False))
        xz_verified = bool(entry.get("xz_magic_verified", False))
        prefix_verified = (
            compressed_probe_bytes > 0
            and inflated_prefix_bytes > 0
            and stream_ready
            and xz_verified
            and str(entry.get("inflated_prefix_hex", "")).startswith("fd377a585a00")
        )
        if prefix_verified:
            stage = "trajectory_member_prefix_verified_streaming_extraction_blocked"
            blocker = "streaming_member_extraction_policy"
        else:
            stage = "trajectory_member_prefix_probe_failed"
            blocker = "member_prefix"

        rows.append(
            {
                "probe_id": f"{probe_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_path": source_path,
                "compressed_probe_range_start": float(int(entry.get("compressed_probe_range_start", -1) or -1)),
                "compressed_probe_range_end": float(int(entry.get("compressed_probe_range_end", -1) or -1)),
                "compressed_probe_bytes": float(compressed_probe_bytes),
                "compressed_probe_md5": str(entry.get("compressed_probe_md5", "none")),
                "inflated_prefix_bytes": float(inflated_prefix_bytes),
                "inflated_prefix_hex": str(entry.get("inflated_prefix_hex", "none")),
                "stream_inflate_ready": float(stream_ready),
                "xz_magic_verified": float(xz_verified),
                "member_prefix_verified": float(prefix_verified),
                "trajectory_extraction_ready": 0.0,
                "real_reanalysis_ready": 0.0,
                "primary_blocker": blocker,
                "probe_stage": stage,
            }
        )
    return rows


def sota_glassbench_trajectory_inner_tar_header_probe_gate(
    *,
    tar_probe_id: str,
    accession_id: str,
    member_probe_rows: Sequence[dict[str, float | str]],
    tar_probe_manifest: dict,
) -> list[dict[str, float | str]]:
    """Verify inner tar headers after ZIP-member and XZ-prefix streaming probes."""

    if not tar_probe_id:
        raise ValueError("tar_probe_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")

    entries_value = tar_probe_manifest.get("entries", [])
    tar_entries = (
        [entry for entry in entries_value if isinstance(entry, dict)]
        if isinstance(entries_value, list)
        else []
    )
    tar_by_path = {str(entry.get("path", "")): entry for entry in tar_entries if entry.get("path")}

    rows: list[dict[str, float | str]] = []
    for member_probe in member_probe_rows:
        system_id = str(member_probe.get("system_id", "unknown"))
        temperature = str(member_probe.get("temperature", "none"))
        source_path = str(member_probe.get("source_path", "none"))
        prefix_verified = bool(float(member_probe.get("member_prefix_verified", 0.0)))
        row_id = f"{tar_probe_id}_{system_id.lower()}_t{temperature.replace('.', '_')}"
        if not prefix_verified or source_path == "none":
            rows.append(
                {
                    "tar_probe_id": row_id,
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "compressed_probe_bytes": 0.0,
                    "xz_prefix_bytes": 0.0,
                    "tar_probe_bytes": 0.0,
                    "root_directory": "none",
                    "first_npz_member": "none",
                    "first_npz_size_bytes": 0.0,
                    "npz_member_count_in_probe": 0.0,
                    "split_labels_in_probe": "none",
                    "tar_magic_verified": 0.0,
                    "npz_member_header_verified": 0.0,
                    "trajectory_layout_ready": 0.0,
                    "trajectory_extraction_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": str(member_probe.get("primary_blocker", "member_prefix")),
                    "tar_probe_stage": "trajectory_member_prefix_incomplete",
                }
            )
            continue

        entry = tar_by_path.get(source_path)
        if entry is None:
            rows.append(
                {
                    "tar_probe_id": row_id,
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "compressed_probe_bytes": 0.0,
                    "xz_prefix_bytes": 0.0,
                    "tar_probe_bytes": 0.0,
                    "root_directory": "none",
                    "first_npz_member": "none",
                    "first_npz_size_bytes": 0.0,
                    "npz_member_count_in_probe": 0.0,
                    "split_labels_in_probe": "none",
                    "tar_magic_verified": 0.0,
                    "npz_member_header_verified": 0.0,
                    "trajectory_layout_ready": 0.0,
                    "trajectory_extraction_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": "inner_tar_header_probe",
                    "tar_probe_stage": "trajectory_inner_tar_header_probe_missing",
                }
            )
            continue

        root_directory = str(entry.get("root_directory", "none"))
        first_npz_member = str(entry.get("first_npz_member", "none"))
        split_labels_value = entry.get("split_labels_in_probe", [])
        if isinstance(split_labels_value, list):
            split_labels = ";".join(str(label) for label in split_labels_value) or "none"
        else:
            split_labels = str(split_labels_value or "none")
        compressed_probe_bytes = int(entry.get("compressed_probe_bytes", 0) or 0)
        xz_prefix_bytes = int(entry.get("xz_prefix_bytes", 0) or 0)
        tar_probe_bytes = int(entry.get("tar_probe_bytes", 0) or 0)
        first_npz_size = int(entry.get("first_npz_size_bytes", 0) or 0)
        npz_count = int(entry.get("npz_member_count_in_probe", 0) or 0)
        tar_magic_verified = bool(entry.get("tar_magic_verified", False))
        npz_verified = first_npz_member.endswith(".npz") and first_npz_size > 0 and npz_count > 0
        layout_ready = (
            compressed_probe_bytes > 0
            and xz_prefix_bytes > 0
            and tar_probe_bytes >= 512
            and root_directory.endswith("/")
            and tar_magic_verified
            and npz_verified
        )
        if layout_ready:
            stage = "trajectory_inner_tar_layout_verified_extraction_blocked"
            blocker = "streaming_npz_extraction_policy"
        else:
            stage = "trajectory_inner_tar_header_probe_failed"
            blocker = "inner_tar_layout"

        rows.append(
            {
                "tar_probe_id": row_id,
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_path": source_path,
                "compressed_probe_bytes": float(compressed_probe_bytes),
                "xz_prefix_bytes": float(xz_prefix_bytes),
                "tar_probe_bytes": float(tar_probe_bytes),
                "root_directory": root_directory,
                "first_npz_member": first_npz_member,
                "first_npz_size_bytes": float(first_npz_size),
                "npz_member_count_in_probe": float(npz_count),
                "split_labels_in_probe": split_labels,
                "tar_magic_verified": float(tar_magic_verified),
                "npz_member_header_verified": float(npz_verified),
                "trajectory_layout_ready": float(layout_ready),
                "trajectory_extraction_ready": 0.0,
                "real_reanalysis_ready": 0.0,
                "primary_blocker": blocker,
                "tar_probe_stage": stage,
            }
        )
    return rows


def sota_glassbench_trajectory_npz_schema_probe_gate(
    *,
    schema_probe_id: str,
    accession_id: str,
    tar_probe_rows: Sequence[dict[str, float | str]],
    schema_probe_manifest: dict,
    required_arrays: Sequence[str],
) -> list[dict[str, float | str]]:
    """Verify first-NPZ array schemas after inner tar headers are visible."""

    if not schema_probe_id:
        raise ValueError("schema_probe_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not required_arrays:
        raise ValueError("required_arrays must be nonempty")

    entries_value = schema_probe_manifest.get("entries", [])
    schema_entries = (
        [entry for entry in entries_value if isinstance(entry, dict)]
        if isinstance(entries_value, list)
        else []
    )
    schema_by_key = {
        (str(entry.get("path", "")), str(entry.get("first_npz_member", ""))): entry
        for entry in schema_entries
        if entry.get("path") and entry.get("first_npz_member")
    }

    rows: list[dict[str, float | str]] = []
    for tar_probe in tar_probe_rows:
        system_id = str(tar_probe.get("system_id", "unknown"))
        temperature = str(tar_probe.get("temperature", "none"))
        source_path = str(tar_probe.get("source_path", "none"))
        first_npz_member = str(tar_probe.get("first_npz_member", "none"))
        layout_ready = bool(float(tar_probe.get("trajectory_layout_ready", 0.0)))
        row_id = f"{schema_probe_id}_{system_id.lower()}_t{temperature.replace('.', '_')}"
        if not layout_ready or source_path == "none" or first_npz_member == "none":
            rows.append(
                {
                    "schema_probe_id": row_id,
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "first_npz_member": first_npz_member,
                    "npz_member_bytes": 0.0,
                    "npz_member_md5": "none",
                    "array_names": "none",
                    "array_shapes": "none",
                    "array_dtypes": "none",
                    "required_arrays_present": 0.0,
                    "npz_magic_verified": 0.0,
                    "npz_schema_ready": 0.0,
                    "coordinate_array_ready": 0.0,
                    "particle_count": 0.0,
                    "frame_count": 0.0,
                    "spatial_dimension": 0.0,
                    "trajectory_extraction_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": str(tar_probe.get("primary_blocker", "inner_tar_layout")),
                    "schema_probe_stage": "trajectory_inner_tar_layout_incomplete",
                }
            )
            continue

        entry = schema_by_key.get((source_path, first_npz_member))
        if entry is None:
            rows.append(
                {
                    "schema_probe_id": row_id,
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "first_npz_member": first_npz_member,
                    "npz_member_bytes": 0.0,
                    "npz_member_md5": "none",
                    "array_names": "none",
                    "array_shapes": "none",
                    "array_dtypes": "none",
                    "required_arrays_present": 0.0,
                    "npz_magic_verified": 0.0,
                    "npz_schema_ready": 0.0,
                    "coordinate_array_ready": 0.0,
                    "particle_count": 0.0,
                    "frame_count": 0.0,
                    "spatial_dimension": 0.0,
                    "trajectory_extraction_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": "npz_schema_probe",
                    "schema_probe_stage": "trajectory_npz_schema_probe_missing",
                }
            )
            continue

        arrays_value = entry.get("arrays", [])
        arrays = (
            [array for array in arrays_value if isinstance(array, dict)]
            if isinstance(arrays_value, list)
            else []
        )
        array_names = [str(array.get("name", "")) for array in arrays if array.get("name")]
        shape_by_name = {
            str(array.get("name", "")): array.get("shape", [])
            for array in arrays
            if array.get("name")
        }
        dtype_by_name = {
            str(array.get("name", "")): str(array.get("dtype", "unknown"))
            for array in arrays
            if array.get("name")
        }
        required_present = all(name in array_names for name in required_arrays)
        positions_shape = shape_by_name.get("positions.npy", [])
        positions_ready = (
            isinstance(positions_shape, list)
            and len(positions_shape) == 3
            and all(isinstance(value, int) and value > 0 for value in positions_shape)
            and dtype_by_name.get("positions.npy") in {"float32", "float64"}
        )
        frame_count = positions_shape[0] if positions_ready else 0
        particle_count = positions_shape[1] if positions_ready else 0
        dimension = positions_shape[2] if positions_ready else 0
        npz_bytes = int(entry.get("npz_member_bytes", 0) or 0)
        magic_verified = bool(entry.get("npz_magic_verified", False))
        schema_ready = npz_bytes > 0 and magic_verified and required_present and positions_ready
        if schema_ready:
            stage = "trajectory_npz_coordinate_schema_verified"
            blocker = "full_npz_ensemble_extraction_policy"
        else:
            stage = "trajectory_npz_schema_probe_failed"
            blocker = "npz_coordinate_schema"

        rows.append(
            {
                "schema_probe_id": row_id,
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_path": source_path,
                "first_npz_member": first_npz_member,
                "npz_member_bytes": float(npz_bytes),
                "npz_member_md5": str(entry.get("npz_member_md5", "none")),
                "array_names": ";".join(array_names) or "none",
                "array_shapes": ";".join(
                    f"{name}:{'x'.join(str(value) for value in shape_by_name.get(name, [])) or 'scalar'}"
                    for name in array_names
                )
                or "none",
                "array_dtypes": ";".join(f"{name}:{dtype_by_name.get(name, 'unknown')}" for name in array_names)
                or "none",
                "required_arrays_present": float(required_present),
                "npz_magic_verified": float(magic_verified),
                "npz_schema_ready": float(schema_ready),
                "coordinate_array_ready": float(positions_ready),
                "particle_count": float(particle_count),
                "frame_count": float(frame_count),
                "spatial_dimension": float(dimension),
                "trajectory_extraction_ready": 0.0,
                "real_reanalysis_ready": 0.0,
                "primary_blocker": blocker,
                "schema_probe_stage": stage,
            }
        )
    return rows


def sota_glassbench_trajectory_first_npz_observable_smoke_gate(
    *,
    smoke_id: str,
    accession_id: str,
    schema_probe_rows: Sequence[dict[str, float | str]],
    observable_manifest: dict,
    required_method: str,
) -> list[dict[str, float | str]]:
    """Verify minimal MSD/NGP observables from first streamed trajectory NPZ members."""

    if not smoke_id:
        raise ValueError("smoke_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not required_method:
        raise ValueError("required_method must be nonempty")

    entries_value = observable_manifest.get("entries", [])
    observable_entries = (
        [entry for entry in entries_value if isinstance(entry, dict)]
        if isinstance(entries_value, list)
        else []
    )
    observable_by_key = {
        (str(entry.get("path", "")), str(entry.get("first_npz_member", ""))): entry
        for entry in observable_entries
        if entry.get("path") and entry.get("first_npz_member")
    }

    rows: list[dict[str, float | str]] = []
    for schema_row in schema_probe_rows:
        system_id = str(schema_row.get("system_id", "unknown"))
        temperature = str(schema_row.get("temperature", "none"))
        source_path = str(schema_row.get("source_path", "none"))
        first_npz_member = str(schema_row.get("first_npz_member", "none"))
        schema_ready = bool(float(schema_row.get("npz_schema_ready", 0.0)))
        coords_ready = bool(float(schema_row.get("coordinate_array_ready", 0.0)))
        row_id = f"{smoke_id}_{system_id.lower()}_t{temperature.replace('.', '_')}"
        if not schema_ready or not coords_ready or source_path == "none" or first_npz_member == "none":
            rows.append(
                {
                    "smoke_id": row_id,
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "first_npz_member": first_npz_member,
                    "observable_method": "none",
                    "positions_md5": "none",
                    "box_length": 0.0,
                    "frame_count": 0.0,
                    "particle_count": 0.0,
                    "spatial_dimension": 0.0,
                    "final_frame_index": -1.0,
                    "final_msd": 0.0,
                    "final_ngp_2d": 0.0,
                    "peak_ngp_frame_index": -1.0,
                    "peak_ngp_2d": 0.0,
                    "msd_at_peak_ngp": 0.0,
                    "max_abs_min_image_displacement": 0.0,
                    "observable_smoke_ready": 0.0,
                    "trajectory_extraction_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": str(schema_row.get("primary_blocker", "npz_schema")),
                    "smoke_stage": "trajectory_npz_schema_incomplete",
                }
            )
            continue

        entry = observable_by_key.get((source_path, first_npz_member))
        if entry is None:
            rows.append(
                {
                    "smoke_id": row_id,
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "first_npz_member": first_npz_member,
                    "observable_method": "none",
                    "positions_md5": "none",
                    "box_length": 0.0,
                    "frame_count": 0.0,
                    "particle_count": 0.0,
                    "spatial_dimension": 0.0,
                    "final_frame_index": -1.0,
                    "final_msd": 0.0,
                    "final_ngp_2d": 0.0,
                    "peak_ngp_frame_index": -1.0,
                    "peak_ngp_2d": 0.0,
                    "msd_at_peak_ngp": 0.0,
                    "max_abs_min_image_displacement": 0.0,
                    "observable_smoke_ready": 0.0,
                    "trajectory_extraction_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": "first_npz_observable_smoke",
                    "smoke_stage": "first_npz_observable_smoke_missing",
                }
            )
            continue

        method = str(entry.get("observable_method", "none"))
        frame_count = int(entry.get("frame_count", 0) or 0)
        particle_count = int(entry.get("particle_count", 0) or 0)
        dimension = int(entry.get("spatial_dimension", 0) or 0)
        final_frame = int(entry.get("final_frame_index", -1) or -1)
        final_msd = float(entry.get("final_msd", 0.0) or 0.0)
        final_ngp = float(entry.get("final_ngp_2d", 0.0) or 0.0)
        peak_frame = int(entry.get("peak_ngp_frame_index", -1) or -1)
        peak_ngp = float(entry.get("peak_ngp_2d", 0.0) or 0.0)
        msd_at_peak = float(entry.get("msd_at_peak_ngp", 0.0) or 0.0)
        max_disp = float(entry.get("max_abs_min_image_displacement", 0.0) or 0.0)
        smoke_ready = (
            method == required_method
            and frame_count >= 2
            and particle_count > 0
            and dimension == 2
            and 0 <= final_frame < frame_count
            and 0 <= peak_frame < frame_count
            and final_msd > 0.0
            and peak_ngp >= 0.0
            and msd_at_peak > 0.0
            and max_disp > 0.0
        )
        if smoke_ready:
            stage = "first_npz_msd_ngp_smoke_ready_reanalysis_blocked"
            blocker = "single_npz_no_time_or_uncertainty"
        else:
            stage = "first_npz_observable_smoke_failed"
            blocker = "first_npz_msd_ngp"

        rows.append(
            {
                "smoke_id": row_id,
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_path": source_path,
                "first_npz_member": first_npz_member,
                "observable_method": method,
                "positions_md5": str(entry.get("positions_md5", "none")),
                "box_length": float(entry.get("box_length", 0.0) or 0.0),
                "frame_count": float(frame_count),
                "particle_count": float(particle_count),
                "spatial_dimension": float(dimension),
                "final_frame_index": float(final_frame),
                "final_msd": final_msd,
                "final_ngp_2d": final_ngp,
                "peak_ngp_frame_index": float(peak_frame),
                "peak_ngp_2d": peak_ngp,
                "msd_at_peak_ngp": msd_at_peak,
                "max_abs_min_image_displacement": max_disp,
                "observable_smoke_ready": float(smoke_ready),
                "trajectory_extraction_ready": 0.0,
                "real_reanalysis_ready": 0.0,
                "primary_blocker": blocker,
                "smoke_stage": stage,
            }
        )
    return rows


def sota_glassbench_trajectory_first_npz_observable_curve_gate(
    *,
    curve_id: str,
    accession_id: str,
    smoke_rows: Sequence[dict[str, float | str]],
    curve_manifest: dict,
    required_method: str,
) -> list[dict[str, float | str]]:
    """Expand first-NPZ smoke checks into frame-index MSD/NGP curve rows."""

    if not curve_id:
        raise ValueError("curve_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not required_method:
        raise ValueError("required_method must be nonempty")

    entries_value = curve_manifest.get("entries", [])
    curve_entries = (
        [entry for entry in entries_value if isinstance(entry, dict)]
        if isinstance(entries_value, list)
        else []
    )
    curve_by_key = {
        (str(entry.get("path", "")), str(entry.get("first_npz_member", ""))): entry
        for entry in curve_entries
        if entry.get("path") and entry.get("first_npz_member")
    }

    rows: list[dict[str, float | str]] = []
    default_structural_observables: dict[str, float | str] = {
        "wave_numbers": "none",
        "self_intermediate_scattering_by_k": "none",
        "self_intermediate_scattering": 0.0,
        "overlap_radius": 0.0,
        "chi4_overlap": 0.0,
    }
    for smoke_row in smoke_rows:
        system_id = str(smoke_row.get("system_id", "unknown"))
        temperature = str(smoke_row.get("temperature", "none"))
        source_path = str(smoke_row.get("source_path", "none"))
        first_npz_member = str(smoke_row.get("first_npz_member", "none"))
        smoke_ready = bool(float(smoke_row.get("observable_smoke_ready", 0.0)))
        method = str(smoke_row.get("observable_method", "none"))
        row_id_prefix = f"{curve_id}_{system_id.lower()}_t{temperature.replace('.', '_')}"
        if not smoke_ready or source_path == "none" or first_npz_member == "none":
            rows.append(
                {
                    "curve_id": f"{row_id_prefix}_frame_missing",
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "first_npz_member": first_npz_member,
                    "observable_method": method,
                    "frame_index": -1.0,
                    "msd": 0.0,
                    "ngp_2d": 0.0,
                    **default_structural_observables,
                    "observable_curve_ready": 0.0,
                    "trajectory_extraction_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": str(smoke_row.get("primary_blocker", "first_npz_observable_smoke")),
                    "curve_stage": "first_npz_observable_smoke_incomplete",
                }
            )
            continue

        entry = curve_by_key.get((source_path, first_npz_member))
        if entry is None:
            rows.append(
                {
                    "curve_id": f"{row_id_prefix}_frame_missing",
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "first_npz_member": first_npz_member,
                    "observable_method": method,
                    "frame_index": -1.0,
                    "msd": 0.0,
                    "ngp_2d": 0.0,
                    **default_structural_observables,
                    "observable_curve_ready": 0.0,
                    "trajectory_extraction_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": "first_npz_observable_curve",
                    "curve_stage": "first_npz_observable_curve_missing",
                }
            )
            continue

        entry_method = str(entry.get("observable_method", "none"))
        frame_indices = entry.get("frame_indices", [])
        msd_values = entry.get("msd", [])
        ngp_values = entry.get("ngp_2d", [])
        wave_numbers_value = entry.get("wave_numbers", [])
        fs_by_k_values = entry.get("self_intermediate_scattering_by_k", [])
        chi4_values = entry.get("chi4_overlap", [])
        overlap_radius = float(entry.get("overlap_radius", 0.0) or 0.0)
        structural_values_ready = (
            isinstance(wave_numbers_value, list)
            and len(wave_numbers_value) > 0
            and isinstance(fs_by_k_values, list)
            and isinstance(chi4_values, list)
            and len(fs_by_k_values) == len(frame_indices)
            and len(chi4_values) == len(frame_indices)
            and overlap_radius > 0.0
        )
        curve_ready = (
            entry_method == required_method
            and isinstance(frame_indices, list)
            and isinstance(msd_values, list)
            and isinstance(ngp_values, list)
            and len(frame_indices) == len(msd_values) == len(ngp_values)
            and len(frame_indices) >= 2
        )
        if not curve_ready:
            rows.append(
                {
                    "curve_id": f"{row_id_prefix}_frame_missing",
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "first_npz_member": first_npz_member,
                    "observable_method": entry_method,
                    "frame_index": -1.0,
                    "msd": 0.0,
                    "ngp_2d": 0.0,
                    **default_structural_observables,
                    "observable_curve_ready": 0.0,
                    "trajectory_extraction_ready": 0.0,
                    "real_reanalysis_ready": 0.0,
                    "primary_blocker": "first_npz_observable_curve",
                    "curve_stage": "first_npz_observable_curve_failed",
                }
            )
            continue

        wave_numbers_text = (
            ";".join(f"{float(value):g}" for value in wave_numbers_value)
            if structural_values_ready
            else "none"
        )
        for row_index, (frame_index, msd, ngp) in enumerate(zip(frame_indices, msd_values, ngp_values)):
            frame = int(frame_index)
            row = {
                "curve_id": f"{row_id_prefix}_frame_{frame}",
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_path": source_path,
                "first_npz_member": first_npz_member,
                "observable_method": entry_method,
                "frame_index": float(frame),
                "msd": float(msd),
                "ngp_2d": float(ngp),
                **default_structural_observables,
                "observable_curve_ready": 1.0,
                "trajectory_extraction_ready": 0.0,
                "real_reanalysis_ready": 0.0,
                "primary_blocker": "single_npz_frame_index_curve",
                "curve_stage": "first_npz_observable_curve_ready_reanalysis_blocked",
            }
            if structural_values_ready:
                fs_text = str(fs_by_k_values[row_index])
                first_fs = fs_text.split(";")[0] if fs_text else "nan"
                row.update(
                    {
                        "wave_numbers": wave_numbers_text,
                        "self_intermediate_scattering_by_k": fs_text,
                        "self_intermediate_scattering": float(first_fs),
                        "overlap_radius": overlap_radius,
                        "chi4_overlap": float(chi4_values[row_index]),
                    }
                )
            rows.append(row)
    return rows


def sota_glassbench_trajectory_first_npz_inversion_readiness_gate(
    *,
    benchmark_id: str,
    accession_id: str,
    curve_rows: Sequence[dict[str, float | str]],
    required_observables: Sequence[str],
    required_uncertainty_columns: Sequence[str],
    min_member_count: int,
    min_frame_count: int,
    has_physical_time: bool,
) -> list[dict[str, float | str]]:
    """Gate first-NPZ GlassBench curves before claiming SOTA inversion readiness."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not curve_rows:
        raise ValueError("curve_rows must be nonempty")
    if not required_observables:
        raise ValueError("required_observables must be nonempty")
    if int(min_member_count) != min_member_count or min_member_count < 1:
        raise ValueError("min_member_count must be a positive integer")
    if int(min_frame_count) != min_frame_count or min_frame_count < 2:
        raise ValueError("min_frame_count must be an integer at least two")
    if any(not observable for observable in required_observables):
        raise ValueError("required_observables must contain nonempty strings")
    if any(not column for column in required_uncertainty_columns):
        raise ValueError("required_uncertainty_columns must contain nonempty strings")

    required = list(dict.fromkeys(str(observable) for observable in required_observables))
    required_sigmas = list(dict.fromkeys(str(column) for column in required_uncertainty_columns))
    grouped: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    for row in curve_rows:
        key = (str(row.get("system_id", "unknown")), str(row.get("temperature", "none")))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, float | str]] = []
    for (system_id, temperature), group in sorted(grouped.items()):
        ready_group = [row for row in group if float(row.get("observable_curve_ready", 0.0)) == 1.0]
        if ready_group:
            source_paths = sorted({str(row.get("source_path", "none")) for row in ready_group})
            members = sorted({str(row.get("first_npz_member", "none")) for row in ready_group})
            frame_indices = sorted({float(row.get("frame_index", -1.0)) for row in ready_group})
        else:
            source_paths = sorted({str(row.get("source_path", "none")) for row in group})
            members = sorted({str(row.get("first_npz_member", "none")) for row in group})
            frame_indices = []
        members = [member for member in members if member != "none"]
        available: list[str] = []
        available_sigmas: list[str] = []
        positive_sigmas: set[str] = set()
        rows_for_columns = ready_group if ready_group else group
        for row in rows_for_columns:
            for observable in required:
                if observable in row:
                    available.append(observable)
            for column in required_sigmas:
                if column in row:
                    available_sigmas.append(column)
                    try:
                        if float(row[column]) > 0.0:
                            positive_sigmas.add(column)
                    except (TypeError, ValueError):
                        pass

        available_unique = list(dict.fromkeys(available))
        sigma_unique = list(dict.fromkeys(available_sigmas))
        missing_observables = [observable for observable in required if observable not in set(available_unique)]
        missing_sigmas = [column for column in required_sigmas if column not in set(sigma_unique)]
        nonpositive_sigmas = [
            column for column in required_sigmas if column in set(sigma_unique) and column not in positive_sigmas
        ]
        frame_count = len(frame_indices)
        member_count = len(members)
        upstream_ready = bool(ready_group)
        physical_time_ready = bool(has_physical_time and "lag_time" not in missing_observables)
        ensemble_ready = member_count >= int(min_member_count)
        structural_ready = (
            upstream_ready
            and physical_time_ready
            and ensemble_ready
            and frame_count >= int(min_frame_count)
            and not missing_observables
        )
        uncertainty_ready = structural_ready and not missing_sigmas and not nonpositive_sigmas

        if not upstream_ready:
            stage = "upstream_curve_incomplete"
            blocker = str(group[0].get("primary_blocker", "observable_curve_ready"))
            next_action = "complete_first_npz_observable_curve"
        elif not physical_time_ready:
            stage = "frame_index_curve_only"
            blocker = "physical_time_semantics"
            next_action = "attach_physical_lag_time_and_units"
        elif frame_count < int(min_frame_count):
            stage = "curve_too_short_for_alpha_window"
            blocker = "frame_count"
            next_action = "extract_longer_trajectory_window"
        elif not ensemble_ready:
            stage = "single_member_curve_only"
            blocker = "ensemble_members"
            next_action = "extract_multiple_independent_npz_members"
        elif missing_observables:
            stage = "observable_set_incomplete"
            blocker = missing_observables[0]
            next_action = f"compute_{missing_observables[0]}"
        elif missing_sigmas:
            stage = "structural_curve_without_uncertainty"
            blocker = missing_sigmas[0]
            next_action = "add_block_jackknife_uncertainty_columns"
        elif nonpositive_sigmas:
            stage = "structural_curve_without_uncertainty"
            blocker = nonpositive_sigmas[0]
            next_action = "replace_nonpositive_uncertainty_columns"
        else:
            stage = "uncertainty_weighted_sota_inversion_ready"
            blocker = "none"
            next_action = "run_persistence_exchange_real_data_inversion"

        out.append(
            {
                "benchmark_id": benchmark_id,
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_paths": ";".join(source_paths) if source_paths else "none",
                "first_npz_members": ";".join(members) if members else "none",
                "frame_count": float(frame_count),
                "member_count": float(member_count),
                "min_frame_count": float(min_frame_count),
                "min_member_count": float(min_member_count),
                "required_observables": ";".join(required),
                "available_observables": ";".join(available_unique) if available_unique else "none",
                "missing_observables": ";".join(missing_observables) if missing_observables else "none",
                "required_uncertainty_columns": ";".join(required_sigmas) if required_sigmas else "none",
                "available_uncertainty_columns": ";".join(sigma_unique) if sigma_unique else "none",
                "missing_uncertainty_columns": ";".join(missing_sigmas) if missing_sigmas else "none",
                "nonpositive_uncertainty_columns": ";".join(nonpositive_sigmas)
                if nonpositive_sigmas
                else "none",
                "physical_time_ready": float(physical_time_ready),
                "ensemble_ready": float(ensemble_ready),
                "structural_curve_ready": float(structural_ready),
                "uncertainty_ready": float(uncertainty_ready),
                "sota_inversion_ready": float(uncertainty_ready),
                "primary_blocker": blocker,
                "next_required_action": next_action,
                "readiness_stage": stage,
            }
        )
    return out


def sota_glassbench_observable_coverage_audit_gate(
    *,
    audit_id: str,
    accession_id: str,
    curve_rows: Sequence[dict[str, float | str]],
    inversion_readiness_rows: Sequence[dict[str, float | str]],
    observable_semantics_rows: Sequence[dict[str, float | str]],
    required_observables: Sequence[str],
) -> list[dict[str, float | str]]:
    """Audit whether real GlassBench rows expose the observables needed for PE inversion."""

    if not audit_id:
        raise ValueError("audit_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not curve_rows:
        raise ValueError("curve_rows must be nonempty")
    if not inversion_readiness_rows:
        raise ValueError("inversion_readiness_rows must be nonempty")
    required = list(dict.fromkeys(str(observable) for observable in required_observables))
    if not required or any(not observable for observable in required):
        raise ValueError("required_observables must contain nonempty strings")

    def split_items(value: object) -> list[str]:
        text = str(value)
        if not text or text == "none":
            return []
        return [item for item in text.split(";") if item and item != "none"]

    def key(row: dict[str, float | str]) -> tuple[str, str]:
        return (str(row.get("system_id", "unknown")), str(row.get("temperature", "none")))

    grouped_curves: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    for row in curve_rows:
        grouped_curves.setdefault(key(row), []).append(row)

    readiness_by_key = {key(row): row for row in inversion_readiness_rows}
    semantics_by_key: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    for row in observable_semantics_rows:
        semantics_by_key.setdefault(key(row), []).append(row)

    all_keys = sorted(
        {
            item
            for item in set(grouped_curves) | set(readiness_by_key)
            if item[0] != "KA" and item[1] != "none"
        },
        key=lambda item: (item[0], float(item[1])),
    )

    out: list[dict[str, float | str]] = []
    for system_id, temperature in all_keys:
        group = grouped_curves.get((system_id, temperature), [])
        readiness = readiness_by_key.get((system_id, temperature), {})
        ready_rows = [row for row in group if float(row.get("observable_curve_ready", 0.0)) == 1.0]
        rows_for_columns = ready_rows if ready_rows else group

        available: list[str] = []
        if any("frame_index" in row for row in rows_for_columns):
            available.append("frame_index")
        for observable in required:
            if any(observable in row for row in rows_for_columns):
                available.append(observable)
        for observable in split_items(readiness.get("available_observables", "none")):
            if observable not in available:
                available.append(observable)
        available = list(dict.fromkeys(available))
        available_required = {observable for observable in available if observable in set(required)}
        missing = [observable for observable in required if observable not in available_required]

        frame_count = len({float(row.get("frame_index", -1.0)) for row in ready_rows if float(row.get("frame_index", -1.0)) >= 0.0})
        member_count = len(
            {
                str(row.get("first_npz_member", "none"))
                for row in ready_rows
                if str(row.get("first_npz_member", "none")) != "none"
            }
        )
        source_paths = sorted({str(row.get("source_path", "none")) for row in ready_rows or group})
        first_members = sorted(
            {
                str(row.get("first_npz_member", "none"))
                for row in ready_rows or group
                if str(row.get("first_npz_member", "none")) != "none"
            }
        )

        semantics_rows = semantics_by_key.get((system_id, temperature), [])
        semantics_ready = any(float(row.get("real_inversion_ready", 0.0)) == 1.0 for row in semantics_rows)
        proxy_candidates = [
            str(row.get("candidate_observable", "unmapped_result_curve")) for row in semantics_rows
        ]
        has_proxy_semantics = any(
            candidate not in {"direct_trajectory_observables", "unmapped_result_curve"}
            for candidate in proxy_candidates
        )
        proxy_substitution_allowed = False
        trajectory_ready = bool(ready_rows)
        coverage_ready = trajectory_ready and not missing
        publishable_ready = coverage_ready and (semantics_ready or not observable_semantics_rows)

        if not trajectory_ready:
            stage = "trajectory_observable_curve_missing"
            blocker = str(readiness.get("primary_blocker", "observable_curve_ready"))
            actions = ["complete_first_npz_observable_curve"]
        elif missing:
            blocker = "observable_set"
            if available == ["frame_index", "msd", "ngp_2d"] or set(available) == {
                "frame_index",
                "msd",
                "ngp_2d",
            }:
                stage = "frame_index_msd_ngp_only"
            else:
                stage = "required_observable_set_incomplete"
            actions = []
            if "lag_time" in missing:
                actions.append("compute_lag_time")
            if "self_intermediate_scattering_by_k" in missing:
                actions.append("compute_multi_k_self_intermediate_scattering")
            if "chi4_overlap" in missing:
                actions.append("compute_overlap_chi4")
            if has_proxy_semantics or semantics_rows:
                actions.append("do_not_substitute_rhomax_or_ml_feature_curves_for_fs_chi4")
        elif not semantics_ready and observable_semantics_rows:
            stage = "observable_semantics_incomplete"
            blocker = "remote_result_observable_semantics"
            actions = ["attach_direct_fs_ngp_msd_chi4_semantics"]
            if has_proxy_semantics or semantics_rows:
                actions.append("do_not_substitute_rhomax_or_ml_feature_curves_for_fs_chi4")
        else:
            stage = "real_inversion_observable_set_ready"
            blocker = "none"
            actions = ["attach_uncertainties_and_run_real_inversion"]

        out.append(
            {
                "audit_id": f"{audit_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_paths": ";".join(source_paths) if source_paths else "none",
                "first_npz_members": ";".join(first_members) if first_members else "none",
                "frame_count": float(frame_count),
                "member_count": float(member_count),
                "required_observables": ";".join(required),
                "available_trajectory_observables": ";".join(available) if available else "none",
                "missing_observables": ";".join(missing) if missing else "none",
                "remote_candidate_observables": ";".join(proxy_candidates) if proxy_candidates else "none",
                "remote_result_semantics_ready": float(semantics_ready),
                "proxy_observable_substitution_allowed": float(proxy_substitution_allowed),
                "observable_coverage_ready": float(coverage_ready),
                "publishable_real_inversion_observable_set_ready": float(publishable_ready),
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "next_required_actions": ";".join(list(dict.fromkeys(actions))) if actions else "none",
                "observable_audit_stage": stage,
            }
        )

    if not out:
        raise ValueError("no GlassBench observable coverage rows could be assembled")
    return out


def sota_glassbench_first_npz_structural_observable_plan_gate(
    *,
    plan_id: str,
    accession_id: str,
    schema_probe_rows: Sequence[dict[str, float | str]],
    observable_coverage_rows: Sequence[dict[str, float | str]],
    implemented_observables: Sequence[str],
) -> list[dict[str, float | str]]:
    """Plan the first-NPZ coordinate extraction needed for Fs/chi4 observables."""

    if not plan_id:
        raise ValueError("plan_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not schema_probe_rows:
        raise ValueError("schema_probe_rows must be nonempty")
    if not observable_coverage_rows:
        raise ValueError("observable_coverage_rows must be nonempty")
    implemented = list(dict.fromkeys(str(item) for item in implemented_observables))
    if not implemented or any(not item for item in implemented):
        raise ValueError("implemented_observables must contain nonempty strings")

    def key(row: dict[str, float | str]) -> tuple[str, str]:
        return (str(row.get("system_id", "unknown")), str(row.get("temperature", "none")))

    def split_items(value: object) -> list[str]:
        text = str(value)
        if not text or text == "none":
            return []
        return [item for item in text.split(";") if item and item != "none"]

    coverage_by_key = {key(row): row for row in observable_coverage_rows}
    out: list[dict[str, float | str]] = []
    for schema in sorted(schema_probe_rows, key=lambda row: (str(row.get("system_id", "")), str(row.get("temperature", "")))):
        system_id, temperature = key(schema)
        if system_id == "KA" and temperature == "none":
            continue
        coverage = coverage_by_key.get((system_id, temperature), {})
        schema_ready = bool(float(schema.get("npz_schema_ready", 0.0)))
        coordinate_ready = bool(float(schema.get("coordinate_array_ready", 0.0)))
        raw_cached = bool(float(schema.get("trajectory_extraction_ready", 0.0)))
        coordinate_schema_ready = schema_ready and coordinate_ready
        computable_after_extraction = coordinate_schema_ready and all(
            item in implemented for item in ["msd", "ngp_2d", "self_intermediate_scattering_by_k", "chi4_overlap"]
        )
        immediate = computable_after_extraction and raw_cached
        available_now = split_items(coverage.get("available_trajectory_observables", "none"))
        missing_now = split_items(coverage.get("missing_observables", "none"))
        structural_observables_cached = all(
            item in set(available_now)
            for item in ["msd", "ngp_2d", "self_intermediate_scattering_by_k", "chi4_overlap"]
        )
        after_compute_available = list(dict.fromkeys(available_now + [item for item in implemented if item != "msd"]))
        remaining_after_compute = [
            item
            for item in missing_now
            if item not in set(implemented)
            and not (item == "ngp_2d" and "ngp_2d" in implemented)
            and not (item == "msd" and "msd" in implemented)
        ]

        if not coordinate_schema_ready:
            stage = "coordinate_schema_incomplete"
            blocker = str(schema.get("primary_blocker", "coordinate_schema"))
            actions = ["verify_first_npz_coordinate_schema"]
        elif structural_observables_cached and remaining_after_compute:
            stage = "structural_observables_cached_raw_coordinates_not_retained"
            blocker = "physical_time_semantics" if remaining_after_compute == ["lag_time"] else remaining_after_compute[0]
            actions = ["attach_physical_lag_time_and_units"]
        elif structural_observables_cached:
            stage = "structural_observables_cached_raw_coordinates_not_retained"
            blocker = "none"
            actions = ["attach_uncertainties_and_run_real_inversion"]
        elif not raw_cached:
            stage = "coordinate_schema_ready_positions_bytes_missing"
            blocker = "raw_coordinate_bytes"
            actions = [
                "extract_first_npz_positions_box_types",
                "run_trajectory_observable_protocol_on_extracted_npz",
            ]
        elif remaining_after_compute:
            stage = "structural_observable_compute_ready"
            blocker = "physical_time_semantics" if remaining_after_compute == ["lag_time"] else remaining_after_compute[0]
            actions = ["run_trajectory_observable_protocol_on_cached_npz"]
        else:
            stage = "structural_observable_compute_ready"
            blocker = "none"
            actions = ["run_trajectory_observable_protocol_on_cached_npz"]

        out.append(
            {
                "plan_id": f"{plan_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_path": str(schema.get("source_path", "none")),
                "first_npz_member": str(schema.get("first_npz_member", "none")),
                "npz_member_bytes": float(schema.get("npz_member_bytes", 0.0)),
                "frame_count": float(schema.get("frame_count", 0.0)),
                "particle_count": float(schema.get("particle_count", 0.0)),
                "spatial_dimension": float(schema.get("spatial_dimension", 0.0)),
                "coordinate_schema_ready": float(coordinate_schema_ready),
                "raw_coordinate_bytes_cached": float(raw_cached),
                "structural_observables_cached": float(structural_observables_cached),
                "computable_after_npz_extraction": float(computable_after_extraction),
                "immediately_computable_from_current_cache": float(immediate),
                "implemented_observable_protocol": ";".join(implemented),
                "available_observables_now": ";".join(available_now) if available_now else "none",
                "missing_observables_now": ";".join(missing_now) if missing_now else "none",
                "available_after_structural_compute": (
                    ";".join(after_compute_available) if after_compute_available else "none"
                ),
                "remaining_missing_after_structural_compute": (
                    ";".join(remaining_after_compute) if remaining_after_compute else "none"
                ),
                "minimum_extraction_scope": "positions.npy;box.npy;types.npy",
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "next_required_actions": ";".join(actions),
                "compute_plan_stage": stage,
            }
        )

    if not out:
        raise ValueError("no GlassBench structural observable compute-plan rows could be assembled")
    return out


def sota_glassbench_short_window_trend_canary_gate(
    *,
    canary_id: str,
    accession_id: str,
    curve_rows: Sequence[dict[str, float | str]],
    cold_temperature: str,
    hot_temperature: str,
    min_common_frame_count: int,
    min_msd_slowdown_ratio: float,
    min_peak_ngp: float,
) -> list[dict[str, float | str]]:
    """Compare two real GlassBench first-NPZ curves without promoting them to fits."""

    if not canary_id:
        raise ValueError("canary_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not curve_rows:
        raise ValueError("curve_rows must be nonempty")
    if not cold_temperature:
        raise ValueError("cold_temperature must be nonempty")
    if not hot_temperature:
        raise ValueError("hot_temperature must be nonempty")
    if int(min_common_frame_count) != min_common_frame_count or min_common_frame_count < 2:
        raise ValueError("min_common_frame_count must be an integer at least two")
    if min_msd_slowdown_ratio <= 0.0:
        raise ValueError("min_msd_slowdown_ratio must be positive")
    if min_peak_ngp < 0.0:
        raise ValueError("min_peak_ngp must be nonnegative")

    ready_rows = [
        row
        for row in curve_rows
        if float(row.get("observable_curve_ready", 0.0)) == 1.0
        and str(row.get("temperature", "none")) in {cold_temperature, hot_temperature}
    ]
    system_ids = sorted({str(row.get("system_id", "unknown")) for row in ready_rows})
    out: list[dict[str, float | str]] = []

    for system_id in system_ids:
        by_temp: dict[str, dict[float, dict[str, float | str]]] = {
            cold_temperature: {},
            hot_temperature: {},
        }
        for row in ready_rows:
            if str(row.get("system_id", "unknown")) != system_id:
                continue
            temp = str(row.get("temperature", "none"))
            frame = float(row.get("frame_index", -1.0))
            if frame >= 0.0:
                by_temp[temp][frame] = row

        cold_frames = by_temp[cold_temperature]
        hot_frames = by_temp[hot_temperature]
        common_frames = sorted(set(cold_frames).intersection(hot_frames))
        common_frame_count = len(common_frames)
        cold_peak_ngp = max([float(row.get("ngp_2d", 0.0)) for row in cold_frames.values()] + [0.0])
        hot_peak_ngp = max([float(row.get("ngp_2d", 0.0)) for row in hot_frames.values()] + [0.0])
        if common_frames:
            final_frame = common_frames[-1]
            cold_final_msd = float(cold_frames[final_frame].get("msd", 0.0))
            hot_final_msd = float(hot_frames[final_frame].get("msd", 0.0))
        else:
            final_frame = -1.0
            cold_final_msd = 0.0
            hot_final_msd = 0.0

        ratio = hot_final_msd / cold_final_msd if cold_final_msd > 0.0 else math.inf
        common_ready = common_frame_count >= int(min_common_frame_count)
        slowdown_pass = common_ready and ratio >= float(min_msd_slowdown_ratio)
        ngp_pass = common_ready and cold_peak_ngp >= float(min_peak_ngp) and hot_peak_ngp >= float(min_peak_ngp)
        canary_ready = slowdown_pass and ngp_pass

        if not common_ready:
            stage = "short_window_trend_canary_incomplete"
            blocker = "common_frame_count"
        elif not slowdown_pass:
            stage = "short_window_trend_canary_failed"
            blocker = "short_window_msd_slowdown"
        elif not ngp_pass:
            stage = "short_window_trend_canary_failed"
            blocker = "positive_ngp_canary"
        else:
            stage = "short_window_real_data_canary_ready_inversion_blocked"
            blocker = "physical_time_ensemble_uncertainty"

        out.append(
            {
                "canary_id": f"{canary_id}_{system_id.lower()}_t{cold_temperature.replace('.', '_')}_t{hot_temperature.replace('.', '_')}",
                "accession_id": accession_id,
                "system_id": system_id,
                "cold_temperature": cold_temperature,
                "hot_temperature": hot_temperature,
                "common_frame_count": float(common_frame_count),
                "min_common_frame_count": float(min_common_frame_count),
                "final_common_frame_index": float(final_frame),
                "cold_final_msd": float(cold_final_msd),
                "hot_final_msd": float(hot_final_msd),
                "hot_to_cold_final_msd_ratio": float(ratio),
                "min_msd_slowdown_ratio": float(min_msd_slowdown_ratio),
                "cold_peak_ngp_2d": float(cold_peak_ngp),
                "hot_peak_ngp_2d": float(hot_peak_ngp),
                "min_peak_ngp_2d": float(min_peak_ngp),
                "short_window_msd_slowdown_pass": float(slowdown_pass),
                "positive_ngp_canary_pass": float(ngp_pass),
                "short_window_real_data_canary_ready": float(canary_ready),
                "sota_inversion_ready": 0.0,
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "canary_stage": stage,
            }
        )

    if not out:
        out.append(
            {
                "canary_id": f"{canary_id}_missing",
                "accession_id": accession_id,
                "system_id": "none",
                "cold_temperature": cold_temperature,
                "hot_temperature": hot_temperature,
                "common_frame_count": 0.0,
                "min_common_frame_count": float(min_common_frame_count),
                "final_common_frame_index": -1.0,
                "cold_final_msd": 0.0,
                "hot_final_msd": 0.0,
                "hot_to_cold_final_msd_ratio": 0.0,
                "min_msd_slowdown_ratio": float(min_msd_slowdown_ratio),
                "cold_peak_ngp_2d": 0.0,
                "hot_peak_ngp_2d": 0.0,
                "min_peak_ngp_2d": float(min_peak_ngp),
                "short_window_msd_slowdown_pass": 0.0,
                "positive_ngp_canary_pass": 0.0,
                "short_window_real_data_canary_ready": 0.0,
                "sota_inversion_ready": 0.0,
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": "observable_curve_ready",
                "canary_stage": "short_window_trend_canary_incomplete",
            }
        )

    return out


def sota_glassbench_trajectory_timebase_bridge_gate(
    *,
    bridge_id: str,
    accession_id: str,
    curve_rows: Sequence[dict[str, float | str]],
    payload_adapter_rows: Sequence[dict[str, float | str]],
    require_explicit_frame_time_mapping: bool,
    explicit_frame_time_mappings: dict[tuple[str, str], bool] | None = None,
) -> list[dict[str, float | str]]:
    """Gate result time grids before attaching them to trajectory frame-index curves."""

    if not bridge_id:
        raise ValueError("bridge_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not curve_rows:
        raise ValueError("curve_rows must be nonempty")
    if not payload_adapter_rows:
        raise ValueError("payload_adapter_rows must be nonempty")

    mappings = explicit_frame_time_mappings or {}
    grouped_curves: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    for row in curve_rows:
        if float(row.get("observable_curve_ready", 0.0)) != 1.0:
            continue
        key = (str(row.get("system_id", "unknown")), str(row.get("temperature", "none")))
        grouped_curves.setdefault(key, []).append(row)

    adapters_by_key: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    for row in payload_adapter_rows:
        key = (str(row.get("system_id", "unknown")), str(row.get("temperature", "none")))
        adapters_by_key.setdefault(key, []).append(row)

    out: list[dict[str, float | str]] = []
    for (system_id, temperature), group in sorted(grouped_curves.items(), key=lambda item: (item[0][0], float(item[0][1]))):
        frame_indices = sorted({float(row.get("frame_index", -1.0)) for row in group if float(row.get("frame_index", -1.0)) >= 0.0})
        frame_count = len(frame_indices)
        adapters = adapters_by_key.get((system_id, temperature), [])
        time_adapters = [
            row
            for row in adapters
            if str(row.get("time_grid_path", "none")) != "none"
            and float(row.get("time_point_count", 0.0)) > 0.0
        ]
        time_grid_available = bool(time_adapters)
        if time_adapters:
            best_time_adapter = max(time_adapters, key=lambda row: float(row.get("time_point_count", 0.0)))
            time_point_count = int(float(best_time_adapter.get("time_point_count", 0.0)))
            time_grid_path = str(best_time_adapter.get("time_grid_path", "none"))
            structural_adapter_ready = bool(float(best_time_adapter.get("structural_adapter_ready", 0.0)))
        else:
            time_point_count = 0
            time_grid_path = "none"
            structural_adapter_ready = False

        count_match = time_grid_available and frame_count == time_point_count
        explicit_mapping = bool(mappings.get((system_id, temperature), False))
        mapping_ready = explicit_mapping or not require_explicit_frame_time_mapping
        timebase_ready = bool(time_grid_available and count_match and mapping_ready)

        if not time_grid_available:
            stage = "trajectory_result_timebase_missing"
            blocker = "time_grid"
            next_action = "fetch_same_temperature_time_grid"
        elif not count_match:
            stage = "trajectory_result_timebase_length_mismatch"
            blocker = "frame_time_point_count"
            next_action = "derive_or_fetch_trajectory_frame_time_mapping"
        elif not mapping_ready:
            stage = "trajectory_result_timebase_mapping_required"
            blocker = "explicit_frame_time_mapping"
            next_action = "document_frame_to_physical_time_mapping"
        else:
            stage = "trajectory_timebase_ready_observable_inversion_blocked"
            blocker = "observable_set_and_uncertainty"
            next_action = "compute_fs_chi4_and_uncertainties_on_timebase"

        out.append(
            {
                "bridge_id": f"{bridge_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "frame_count": float(frame_count),
                "time_point_count": float(time_point_count),
                "time_grid_path": time_grid_path,
                "time_grid_available": float(time_grid_available),
                "structural_result_adapter_ready": float(structural_adapter_ready),
                "frame_time_point_count_match": float(count_match),
                "explicit_frame_time_mapping_required": float(require_explicit_frame_time_mapping),
                "explicit_frame_time_mapping": float(explicit_mapping),
                "trajectory_timebase_ready": float(timebase_ready),
                "sota_inversion_ready": 0.0,
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "next_required_action": next_action,
                "timebase_stage": stage,
            }
        )

    if not out:
        out.append(
            {
                "bridge_id": f"{bridge_id}_missing",
                "accession_id": accession_id,
                "system_id": "none",
                "temperature": "none",
                "frame_count": 0.0,
                "time_point_count": 0.0,
                "time_grid_path": "none",
                "time_grid_available": 0.0,
                "structural_result_adapter_ready": 0.0,
                "frame_time_point_count_match": 0.0,
                "explicit_frame_time_mapping_required": float(require_explicit_frame_time_mapping),
                "explicit_frame_time_mapping": 0.0,
                "trajectory_timebase_ready": 0.0,
                "sota_inversion_ready": 0.0,
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": "observable_curve_ready",
                "next_required_action": "complete_first_npz_observable_curve",
                "timebase_stage": "trajectory_result_timebase_missing",
            }
        )

    return out


def sota_glassbench_frame_time_mapping_audit_gate(
    *,
    audit_id: str,
    accession_id: str,
    timebase_rows: Sequence[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Classify candidate frame-to-time mappings without silently interpolating data."""

    if not audit_id:
        raise ValueError("audit_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not timebase_rows:
        raise ValueError("timebase_rows must be nonempty")

    out: list[dict[str, float | str]] = []
    for row in sorted(
        timebase_rows,
        key=lambda item: (str(item.get("system_id", "unknown")), str(item.get("temperature", "none"))),
    ):
        system_id = str(row.get("system_id", "unknown"))
        temperature = str(row.get("temperature", "none"))
        if system_id == "none" or temperature == "none":
            continue
        frame_count = int(float(row.get("frame_count", 0.0)))
        time_point_count = int(float(row.get("time_point_count", 0.0)))
        time_grid_available = bool(float(row.get("time_grid_available", 0.0)))
        explicit_mapping = bool(float(row.get("explicit_frame_time_mapping", 0.0)))
        exact_count_match = time_grid_available and frame_count > 0 and frame_count == time_point_count

        if frame_count > 1 and time_point_count > 1:
            stride_ratio = (frame_count - 1) / (time_point_count - 1)
            integer_stride = math.isclose(stride_ratio, round(stride_ratio), rel_tol=0.0, abs_tol=1e-12)
        else:
            stride_ratio = 0.0
            integer_stride = False
        endpoint_interpolation = bool(time_grid_available and frame_count > 1 and time_point_count > 1 and not exact_count_match)

        if exact_count_match and explicit_mapping:
            stage = "frame_time_mapping_ready"
            accepted = "explicit_count_matched_frame_time_mapping"
            provisional = "none"
            metadata = "none"
            blocker = "none"
            next_action = "run_timebase_attached_observable_protocol"
            publishable_ready = True
        elif exact_count_match:
            stage = "count_matched_mapping_metadata_required"
            accepted = "none"
            provisional = "count_matched_requires_explicit_metadata"
            metadata = "dump_interval;saved_frame_stride;trajectory_frame_origin"
            blocker = "explicit_frame_time_mapping"
            next_action = "document_frame_to_physical_time_mapping"
            publishable_ready = False
        elif integer_stride and explicit_mapping:
            stage = "subsample_mapping_ready"
            accepted = "explicit_integer_stride_subsample_mapping"
            provisional = "none"
            metadata = "none"
            blocker = "none"
            next_action = "compute_observables_on_subsampled_time_grid"
            publishable_ready = True
        elif integer_stride:
            stage = "integer_stride_mapping_metadata_required"
            accepted = "none"
            provisional = "integer_stride_subsample_requires_metadata"
            metadata = "dump_interval;saved_frame_stride;trajectory_frame_origin;result_time_generation_script"
            blocker = "explicit_frame_time_mapping"
            next_action = "document_subsample_stride_and_result_time_generation"
            publishable_ready = False
        elif endpoint_interpolation:
            stage = "ambiguous_frame_time_mapping"
            accepted = "none"
            provisional = "endpoint_interpolation_requires_metadata"
            metadata = "dump_interval;saved_frame_stride;trajectory_frame_origin;result_time_generation_script"
            blocker = "frame_time_point_count"
            next_action = "derive_or_fetch_trajectory_frame_time_mapping"
            publishable_ready = False
        else:
            stage = "frame_time_mapping_missing"
            accepted = "none"
            provisional = "none"
            metadata = "time_grid;trajectory_frame_count;trajectory_frame_origin"
            blocker = "time_grid"
            next_action = "fetch_same_temperature_time_grid"
            publishable_ready = False

        out.append(
            {
                "audit_id": f"{audit_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "frame_count": float(frame_count),
                "time_point_count": float(time_point_count),
                "time_grid_path": str(row.get("time_grid_path", "none")),
                "time_grid_available": float(time_grid_available),
                "exact_count_match": float(exact_count_match),
                "frame_to_result_stride_ratio": float(stride_ratio),
                "integer_stride_subsample_candidate": float(integer_stride and not exact_count_match),
                "endpoint_interpolation_candidate": float(endpoint_interpolation),
                "explicit_frame_time_mapping": float(explicit_mapping),
                "accepted_mapping_class": accepted,
                "provisional_mapping_class": provisional,
                "publishable_frame_time_mapping_ready": float(publishable_ready),
                "minimum_required_metadata": metadata,
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "next_required_action": next_action,
                "mapping_audit_stage": stage,
            }
        )

    if not out:
        raise ValueError("timebase_rows did not contain any concrete system-temperature rows")
    return out


def sota_glassbench_real_inversion_gap_ledger_gate(
    *,
    ledger_id: str,
    accession_id: str,
    short_window_rows: Sequence[dict[str, float | str]],
    timebase_rows: Sequence[dict[str, float | str]],
    ensemble_horizon_rows: Sequence[dict[str, float | str]],
    inversion_readiness_rows: Sequence[dict[str, float | str]],
    observable_semantics_rows: Sequence[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Collapse GlassBench real-data gates into one manuscript-safe claim ledger."""

    if not ledger_id:
        raise ValueError("ledger_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not timebase_rows:
        raise ValueError("timebase_rows must be nonempty")
    if not ensemble_horizon_rows:
        raise ValueError("ensemble_horizon_rows must be nonempty")
    if not inversion_readiness_rows:
        raise ValueError("inversion_readiness_rows must be nonempty")

    canary_by_system: dict[str, dict[str, float | str]] = {}
    for row in short_window_rows:
        system = str(row.get("system_id", "unknown"))
        canary_by_system[system] = row

    timebase_by_key = {
        (str(row.get("system_id", "unknown")), str(row.get("temperature", "none"))): row
        for row in timebase_rows
    }
    ensemble_by_key = {
        (str(row.get("system_id", "unknown")), str(row.get("temperature", "none"))): row
        for row in ensemble_horizon_rows
    }
    inversion_by_key = {
        (str(row.get("system_id", "unknown")), str(row.get("temperature", "none"))): row
        for row in inversion_readiness_rows
    }
    semantics_ready_keys = {
        (str(row.get("system_id", "unknown")), str(row.get("temperature", "none")))
        for row in observable_semantics_rows
        if float(row.get("real_inversion_ready", 0.0)) == 1.0
    }

    keys = sorted(
        {
            key
            for key in set(timebase_by_key) | set(ensemble_by_key) | set(inversion_by_key)
            if key[0] != "KA" and key[1] != "none"
        },
        key=lambda key: (key[0], float(key[1])),
    )
    out: list[dict[str, float | str]] = []
    for system_id, temperature in keys:
        canary = canary_by_system.get(system_id, {})
        timebase = timebase_by_key.get((system_id, temperature), {})
        ensemble = ensemble_by_key.get((system_id, temperature), {})
        inversion = inversion_by_key.get((system_id, temperature), {})

        short_window_ready = bool(float(canary.get("short_window_real_data_canary_ready", 0.0)))
        timebase_ready = bool(float(timebase.get("trajectory_timebase_ready", 0.0)))
        ensemble_ready = bool(float(ensemble.get("prefix_member_horizon_ready", 0.0)))
        structural_ready = bool(float(inversion.get("structural_curve_ready", 0.0)))
        uncertainty_ready = bool(float(inversion.get("uncertainty_ready", 0.0)))
        inversion_ready = bool(float(inversion.get("sota_inversion_ready", 0.0)))
        semantics_ready = (system_id, temperature) in semantics_ready_keys or not observable_semantics_rows
        quantitative_ready = (
            short_window_ready
            and timebase_ready
            and ensemble_ready
            and structural_ready
            and uncertainty_ready
            and inversion_ready
            and semantics_ready
        )

        if quantitative_ready:
            stage = "real_data_quantitative_inversion_ready"
            claim = "uncertainty_weighted_real_trajectory_inversion"
            blocker = "none"
            next_action = "run_real_data_persistence_exchange_residuals"
        elif short_window_ready and not timebase_ready:
            stage = "real_data_canary_timebase_blocked"
            claim = "short_window_coordinate_trend_only"
            blocker = str(timebase.get("primary_blocker", "trajectory_timebase_ready"))
            next_action = str(timebase.get("next_required_action", "derive_or_fetch_trajectory_frame_time_mapping"))
        elif short_window_ready and not ensemble_ready:
            stage = "real_data_canary_ensemble_blocked"
            claim = "short_window_coordinate_trend_only"
            blocker = str(ensemble.get("primary_blocker", "prefix_member_horizon_ready"))
            next_action = str(ensemble.get("next_required_action", "extract_multiple_independent_npz_members"))
        elif short_window_ready and not structural_ready:
            stage = "real_data_canary_observable_set_blocked"
            claim = "short_window_coordinate_trend_only"
            blocker = str(inversion.get("primary_blocker", "structural_curve_ready"))
            next_action = str(inversion.get("next_required_action", "compute_required_observables"))
        elif short_window_ready and not uncertainty_ready:
            stage = "real_data_canary_uncertainty_blocked"
            claim = "short_window_coordinate_trend_only"
            blocker = "uncertainty_columns"
            next_action = "add_member_or_block_uncertainty_columns"
        else:
            stage = "real_data_coordinate_canary_missing"
            claim = "metadata_or_coordinate_ingestion_only"
            blocker = str(canary.get("primary_blocker", "short_window_real_data_canary"))
            next_action = "complete_short_window_coordinate_canary"

        missing_observables = str(inversion.get("missing_observables", "none"))
        missing_uncertainties = str(inversion.get("missing_uncertainty_columns", "none"))
        next_actions = list(
            dict.fromkeys(
                action
                for action in [
                    next_action,
                    str(ensemble.get("next_required_action", "none")),
                    "compute_missing_observables" if missing_observables != "none" else "none",
                    "add_uncertainty_columns" if missing_uncertainties != "none" else "none",
                ]
                if action and action != "none"
            )
        )

        out.append(
            {
                "ledger_id": f"{ledger_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "short_window_claim_ready": float(short_window_ready),
                "trajectory_timebase_ready": float(timebase_ready),
                "ensemble_horizon_ready": float(ensemble_ready),
                "structural_observable_ready": float(structural_ready),
                "uncertainty_ready": float(uncertainty_ready),
                "observable_semantics_ready": float(semantics_ready),
                "quantitative_real_inversion_ready": float(quantitative_ready),
                "thermodynamic_claim_allowed": 0.0,
                "allowed_claim_level": claim,
                "missing_observables": missing_observables,
                "missing_uncertainty_columns": missing_uncertainties,
                "primary_blocker": blocker,
                "next_required_actions": ";".join(next_actions) if next_actions else "none",
                "ledger_stage": stage,
            }
        )

    if not out:
        raise ValueError("no GlassBench real-data ledger rows could be assembled")
    return out


def sota_glassbench_real_inversion_unlock_protocol_gate(
    *,
    protocol_id: str,
    accession_id: str,
    ledger_rows: Sequence[dict[str, float | str]],
    timebase_rows: Sequence[dict[str, float | str]],
    ensemble_horizon_rows: Sequence[dict[str, float | str]],
    inversion_readiness_rows: Sequence[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """List the minimum payload needed to promote GlassBench canaries to real inversion."""

    if not protocol_id:
        raise ValueError("protocol_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not ledger_rows:
        raise ValueError("ledger_rows must be nonempty")
    if not timebase_rows:
        raise ValueError("timebase_rows must be nonempty")
    if not ensemble_horizon_rows:
        raise ValueError("ensemble_horizon_rows must be nonempty")
    if not inversion_readiness_rows:
        raise ValueError("inversion_readiness_rows must be nonempty")

    def rows_by_key(
        rows: Sequence[dict[str, float | str]],
    ) -> dict[tuple[str, str], dict[str, float | str]]:
        return {
            (str(row.get("system_id", "unknown")), str(row.get("temperature", "none"))): row
            for row in rows
        }

    def split_items(value: object) -> list[str]:
        text = str(value)
        if not text or text == "none":
            return []
        return [item for item in text.split(";") if item and item != "none"]

    timebase_by_key = rows_by_key(timebase_rows)
    ensemble_by_key = rows_by_key(ensemble_horizon_rows)
    readiness_by_key = rows_by_key(inversion_readiness_rows)

    out: list[dict[str, float | str]] = []
    for ledger in sorted(
        ledger_rows,
        key=lambda row: (str(row.get("system_id", "unknown")), float(row.get("temperature", 0.0))),
    ):
        system_id = str(ledger.get("system_id", "unknown"))
        temperature = str(ledger.get("temperature", "none"))
        if system_id == "none" or temperature == "none":
            continue
        key = (system_id, temperature)
        timebase = timebase_by_key.get(key, {})
        ensemble = ensemble_by_key.get(key, {})
        readiness = readiness_by_key.get(key, {})

        frame_count = float(timebase.get("frame_count", readiness.get("frame_count", 0.0)))
        time_point_count = float(timebase.get("time_point_count", 0.0))
        mapping_required = bool(float(timebase.get("explicit_frame_time_mapping_required", 1.0)))
        mapping_present = bool(float(timebase.get("explicit_frame_time_mapping", 0.0)))
        mapping_missing = mapping_required and not mapping_present

        observed_prefix_members = float(
            ensemble.get(
                "prefix_npz_member_count",
                ensemble.get("npz_member_count_in_probe", readiness.get("observed_member_count", 0.0)),
            )
        )
        required_members = float(
            ensemble.get("min_member_count", readiness.get("min_member_count", 1.0))
        )
        additional_members = max(0.0, required_members - observed_prefix_members)
        missing_observables = split_items(readiness.get("missing_observables", "none"))
        missing_sigmas = split_items(readiness.get("missing_uncertainty_columns", "none"))

        payload_items: list[str] = []
        if mapping_missing:
            payload_items.append("frame_time_mapping")
        if additional_members > 0.0:
            if math.isclose(additional_members, 1.0):
                payload_items.append("one_more_independent_npz_member")
            else:
                payload_items.append(f"{int(math.ceil(additional_members))}_more_independent_npz_members")
        payload_items.extend(missing_observables)
        payload_items.extend(missing_sigmas)
        payload_items = list(dict.fromkeys(payload_items))

        unlock_ready = not payload_items and bool(
            float(ledger.get("quantitative_real_inversion_ready", readiness.get("sota_inversion_ready", 0.0)))
        )
        current_claim = str(ledger.get("allowed_claim_level", "metadata_or_coordinate_ingestion_only"))
        post_unlock_claim = (
            "uncertainty_weighted_real_trajectory_inversion"
            if current_claim != "uncertainty_weighted_real_trajectory_inversion"
            else current_claim
        )

        out.append(
            {
                "protocol_id": f"{protocol_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "current_claim_level": current_claim,
                "post_unlock_claim_level": post_unlock_claim,
                "minimum_unlock_ready": float(unlock_ready),
                "frame_count": frame_count,
                "time_point_count": time_point_count,
                "frame_time_mapping_required": float(mapping_required),
                "frame_time_mapping_present": float(mapping_present),
                "observed_prefix_member_count": observed_prefix_members,
                "required_member_count": required_members,
                "additional_member_count_needed": additional_members,
                "missing_observables": ";".join(missing_observables) if missing_observables else "none",
                "missing_uncertainty_columns": ";".join(missing_sigmas) if missing_sigmas else "none",
                "minimum_required_payload": ";".join(payload_items) if payload_items else "none",
                "thermodynamic_claim_allowed": 0.0,
                "unlock_stage": "minimum_real_inversion_payload_ready"
                if unlock_ready
                else "minimum_real_inversion_payload_missing",
            }
        )

    if not out:
        raise ValueError("ledger_rows did not contain any concrete system-temperature rows")
    return out


def sota_glassbench_trajectory_npz_ensemble_horizon_gate(
    *,
    horizon_id: str,
    accession_id: str,
    tar_probe_rows: Sequence[dict[str, float | str]],
    inversion_readiness_rows: Sequence[dict[str, float | str]],
    min_member_count: int,
    member_index_rows: Sequence[dict[str, float | str]] | None = None,
) -> list[dict[str, float | str]]:
    """Record how many NPZ trajectory members are visible before ensemble extraction."""

    if not horizon_id:
        raise ValueError("horizon_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not tar_probe_rows:
        raise ValueError("tar_probe_rows must be nonempty")
    if int(min_member_count) != min_member_count or min_member_count < 1:
        raise ValueError("min_member_count must be a positive integer")

    readiness_by_key = {
        (str(row.get("system_id", "unknown")), str(row.get("temperature", "none"))): row
        for row in inversion_readiness_rows
    }
    index_by_key = {
        (str(row.get("system_id", "unknown")), str(row.get("temperature", "none"))): row
        for row in (member_index_rows or [])
    }
    out: list[dict[str, float | str]] = []
    for row in tar_probe_rows:
        system_id = str(row.get("system_id", "unknown"))
        temperature = str(row.get("temperature", "none"))
        source_path = str(row.get("source_path", "none"))
        layout_ready = bool(float(row.get("trajectory_layout_ready", 0.0)))
        index_row = index_by_key.get((system_id, temperature), {})
        index_threshold_ready = bool(float(index_row.get("member_count_threshold_pass", 0.0)))
        indexed_count = int(float(index_row.get("indexed_npz_member_count", 0.0)))
        prefix_count = max(int(float(row.get("npz_member_count_in_probe", 0.0))), indexed_count)
        tar_probe_bytes = int(float(row.get("tar_probe_bytes", 0.0)))
        readiness = readiness_by_key.get((system_id, temperature), {})
        extracted_member_count = int(float(readiness.get("member_count", 0.0)))
        current_sota_ready = bool(float(readiness.get("sota_inversion_ready", 0.0)))
        gap = max(0, int(min_member_count) - prefix_count)
        prefix_horizon_ready = layout_ready and (prefix_count >= int(min_member_count) or index_threshold_ready)

        if not layout_ready:
            stage = "trajectory_layout_incomplete"
            blocker = str(row.get("primary_blocker", "trajectory_layout"))
            next_action = "complete_inner_tar_layout_probe"
        elif current_sota_ready:
            stage = "sota_inversion_already_ready"
            blocker = "none"
            next_action = "report_uncertainty_weighted_residuals"
        elif index_threshold_ready:
            stage = "member_index_horizon_ready_extraction_blocked"
            blocker = "multi_npz_observable_extraction"
            next_action = "extract_indexed_npz_members_and_compute_uncertainties"
        elif prefix_horizon_ready:
            stage = "prefix_member_horizon_ready_extraction_blocked"
            blocker = "streaming_multi_npz_extraction_policy"
            next_action = "extract_visible_npz_members_and_compute_uncertainties"
        else:
            stage = "prefix_member_horizon_short"
            blocker = "additional_npz_member_headers"
            next_action = "extend_tar_probe_or_index_full_member_list"

        out.append(
            {
                "horizon_id": f"{horizon_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_path": source_path,
                "first_npz_member": str(row.get("first_npz_member", "none")),
                "split_labels_in_probe": str(row.get("split_labels_in_probe", "none")),
                "tar_probe_bytes": float(tar_probe_bytes),
                "prefix_npz_member_count": float(prefix_count),
                "extracted_curve_member_count": float(extracted_member_count),
                "min_member_count": float(min_member_count),
                "member_count_gap_to_threshold": float(gap),
                "prefix_member_horizon_ready": float(prefix_horizon_ready),
                "multi_npz_extraction_ready": 0.0,
                "real_reanalysis_ready": 0.0,
                "primary_blocker": blocker,
                "next_required_action": next_action,
                "horizon_stage": stage,
            }
        )
    return out


def sota_glassbench_trajectory_npz_member_index_gate(
    *,
    index_id: str,
    accession_id: str,
    tar_probe_rows: Sequence[dict[str, float | str]],
    member_index_manifest: dict[str, object],
    min_member_count: int,
) -> list[dict[str, float | str]]:
    """Index visible NPZ trajectory members from an extended tar prefix."""

    if not index_id:
        raise ValueError("index_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not tar_probe_rows:
        raise ValueError("tar_probe_rows must be nonempty")
    if int(min_member_count) != min_member_count or min_member_count < 1:
        raise ValueError("min_member_count must be a positive integer")
    entries = member_index_manifest.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("member_index_manifest entries must be a list")

    index_by_path = {
        str(entry.get("path", "none")): entry
        for entry in entries
        if isinstance(entry, dict)
    }

    def split_items(value: object) -> list[str]:
        text = str(value)
        if not text or text == "none":
            return []
        return [item for item in text.split(";") if item and item != "none"]

    def member_id(path: str) -> str:
        if not path or path == "none":
            return "none"
        name = path.rsplit("/", 1)[-1]
        return name[:-4] if name.endswith(".npz") else name

    out: list[dict[str, float | str]] = []
    for row in sorted(
        tar_probe_rows,
        key=lambda item: (str(item.get("system_id", "unknown")), str(item.get("temperature", "none"))),
    ):
        system_id = str(row.get("system_id", "unknown"))
        temperature = str(row.get("temperature", "none"))
        source_path = str(row.get("source_path", "none"))
        layout_ready = bool(float(row.get("trajectory_layout_ready", 0.0)))
        entry = index_by_path.get(source_path, {})
        npz_members = entry.get("npz_members", []) if isinstance(entry, dict) else []
        if not isinstance(npz_members, list):
            raise ValueError("npz_members must be a list")
        member_names = [
            str(member.get("name", "none"))
            for member in npz_members
            if isinstance(member, dict) and str(member.get("name", "none")).endswith(".npz")
        ]
        member_ids = [member_id(name) for name in member_names]
        split_labels = sorted(
            {
                name.split("/")[1]
                for name in member_names
                if len(name.split("/")) >= 3 and name.split("/")[1]
            }
        )
        indexed_count = len(member_names)
        threshold_pass = layout_ready and indexed_count >= int(min_member_count)
        full_list_visible = threshold_pass and bool(entry)
        complete_for_probe = bool(entry.get("member_index_complete_for_probe", False)) if isinstance(entry, dict) else False

        if not layout_ready:
            stage = "trajectory_layout_incomplete"
            blocker = str(row.get("primary_blocker", "trajectory_layout"))
            next_action = "complete_inner_tar_layout_probe"
        elif not entry:
            stage = "member_index_missing"
            blocker = "extended_tar_member_index"
            next_action = "extend_tar_probe_or_index_full_member_list"
        elif threshold_pass:
            stage = "member_index_threshold_ready_extraction_pending"
            blocker = "multi_npz_observable_extraction"
            next_action = "extract_indexed_npz_members_and_compute_uncertainties"
        else:
            stage = "member_index_threshold_short"
            blocker = "member_count"
            next_action = "extend_tar_probe_or_index_full_member_list"

        out.append(
            {
                "index_id": f"{index_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_path": source_path,
                "first_npz_member": str(row.get("first_npz_member", "none")),
                "compressed_probe_bytes": float(entry.get("compressed_probe_bytes", 0.0)) if isinstance(entry, dict) else 0.0,
                "tar_probe_bytes": float(entry.get("tar_probe_bytes", row.get("tar_probe_bytes", 0.0))) if isinstance(entry, dict) else 0.0,
                "prefix_npz_member_count": float(row.get("npz_member_count_in_probe", 0.0)),
                "indexed_npz_member_count": float(indexed_count),
                "required_member_count": float(min_member_count),
                "member_count_threshold_pass": float(threshold_pass),
                "full_member_id_list_visible": float(full_list_visible),
                "member_index_complete_for_probe": float(complete_for_probe),
                "split_labels_in_index": ";".join(split_labels) if split_labels else str(row.get("split_labels_in_probe", "none")),
                "first_four_member_ids": ";".join(member_ids[:4]) if member_ids else "none",
                "visible_npz_members": ";".join(member_names) if member_names else "none",
                "multi_npz_extraction_ready": 0.0,
                "real_reanalysis_ready": 0.0,
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "next_required_action": next_action,
                "member_index_stage": stage,
            }
        )
    return out


def sota_glassbench_trajectory_member_ensemble_observable_gate(
    *,
    ensemble_id: str,
    accession_id: str,
    member_observable_manifest: dict[str, object],
    min_member_count: int,
) -> list[dict[str, float | str]]:
    """Aggregate frame-index GlassBench member observables with ensemble standard errors."""

    if not ensemble_id:
        raise ValueError("ensemble_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if int(min_member_count) != min_member_count or min_member_count < 2:
        raise ValueError("min_member_count must be an integer at least two")
    entries = member_observable_manifest.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("member_observable_manifest entries must be a nonempty list")

    def member_id(path: str) -> str:
        name = path.rsplit("/", 1)[-1]
        return name[:-4] if name.endswith(".npz") else name

    def mean_and_se(values: Sequence[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("member observable values must be finite and nonempty")
        mean = float(np.mean(arr))
        if arr.size < 2:
            return mean, 0.0
        return mean, float(np.std(arr, ddof=1) / math.sqrt(arr.size))

    out: list[dict[str, float | str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("member observable entries must be dictionaries")
        system_id = str(entry.get("system_id", "unknown"))
        temperature = str(entry.get("temperature", "none"))
        source_path = str(entry.get("source_path", "none"))
        wave_numbers = [float(value) for value in entry.get("wave_numbers", [])]
        if not wave_numbers:
            raise ValueError("wave_numbers must be present")
        wave_numbers_text = ";".join(f"{value:g}" for value in wave_numbers)
        overlap_radius = float(entry.get("overlap_radius", 0.0))
        members = entry.get("members", [])
        if not isinstance(members, list) or not members:
            raise ValueError("members must be a nonempty list")

        by_frame: dict[float, list[dict[str, object]]] = {}
        for member in members:
            if not isinstance(member, dict):
                raise ValueError("members must contain dictionaries")
            member_name = str(member.get("member", "none"))
            frame_indices = [float(frame) for frame in member.get("frame_indices", [])]
            msd_values = [float(value) for value in member.get("msd", [])]
            ngp_values = [float(value) for value in member.get("ngp_2d", [])]
            fs_values = [str(value) for value in member.get("self_intermediate_scattering_by_k", [])]
            chi4_values = [float(value) for value in member.get("chi4_overlap", [])]
            lengths = {len(frame_indices), len(msd_values), len(ngp_values), len(fs_values), len(chi4_values)}
            if len(lengths) != 1:
                raise ValueError("member observable arrays must share a length")
            for index, frame in enumerate(frame_indices):
                fs_by_k = _parse_semicolon_float_values(
                    fs_values[index],
                    name="self_intermediate_scattering_by_k",
                )
                if len(fs_by_k) != len(wave_numbers):
                    raise ValueError("Fs values must match wave_numbers length")
                by_frame.setdefault(frame, []).append(
                    {
                        "member": member_name,
                        "member_id": member_id(member_name),
                        "msd": msd_values[index],
                        "ngp_2d": ngp_values[index],
                        "fs_by_k": fs_by_k,
                        "chi4_overlap": chi4_values[index],
                    }
                )

        for frame in sorted(by_frame):
            group = by_frame[frame]
            members_seen = sorted({str(row["member_id"]) for row in group})
            member_count = len(members_seen)
            threshold_pass = member_count >= int(min_member_count)
            msd, sigma_msd = mean_and_se([float(row["msd"]) for row in group])
            ngp, sigma_ngp = mean_and_se([float(row["ngp_2d"]) for row in group])
            chi4, sigma_chi4 = mean_and_se([float(row["chi4_overlap"]) for row in group])
            fs_matrix = np.asarray([row["fs_by_k"] for row in group], dtype=float)
            fs_means = np.mean(fs_matrix, axis=0)
            if fs_matrix.shape[0] < 2:
                fs_sigmas = np.zeros(fs_matrix.shape[1])
            else:
                fs_sigmas = np.std(fs_matrix, axis=0, ddof=1) / math.sqrt(fs_matrix.shape[0])
            fs_first, sigma_fs_first = mean_and_se([float(row["fs_by_k"][0]) for row in group])
            ready = bool(threshold_pass)
            out.append(
                {
                    "ensemble_id": f"{ensemble_id}_{system_id.lower()}_t{temperature.replace('.', '_')}_frame_{int(frame)}",
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "frame_index": float(frame),
                    "member_ids": ";".join(members_seen) if members_seen else "none",
                    "member_count": float(member_count),
                    "min_member_count": float(min_member_count),
                    "ensemble_member_threshold_pass": float(threshold_pass),
                    "msd": msd,
                    "sigma_msd": sigma_msd,
                    "ngp_2d": ngp,
                    "sigma_ngp_2d": sigma_ngp,
                    "wave_numbers": wave_numbers_text,
                    "self_intermediate_scattering_by_k": ";".join(f"{value:.12g}" for value in fs_means),
                    "sigma_self_intermediate_scattering_by_k": ";".join(f"{value:.12g}" for value in fs_sigmas),
                    "self_intermediate_scattering": fs_first,
                    "sigma_self_intermediate_scattering": sigma_fs_first,
                    "overlap_radius": overlap_radius,
                    "chi4_overlap": chi4,
                    "sigma_chi4_overlap": sigma_chi4,
                    "frame_index_uncertainty_ready": float(ready),
                    "physical_time_ready": 0.0,
                    "sota_inversion_ready": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                    "primary_blocker": "physical_time_semantics" if ready else "member_count",
                    "next_required_action": "attach_physical_lag_time_and_units"
                    if ready
                    else "extract_additional_npz_members",
                    "ensemble_observable_stage": "frame_index_member_ensemble_uncertainty_ready"
                    if ready
                    else "member_ensemble_below_threshold",
                }
            )
    return out


def sota_glassbench_ka2d_timecode_semantics_gate(
    *,
    semantics_id: str,
    accession_id: str,
    semantics_manifest: dict[str, object],
    min_members_per_time_code: int,
) -> list[dict[str, float | str]]:
    """Promote KA2D trajectory observables only after correcting time-code semantics."""

    if not semantics_id:
        raise ValueError("semantics_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if int(min_members_per_time_code) != min_members_per_time_code or min_members_per_time_code < 1:
        raise ValueError("min_members_per_time_code must be a positive integer")
    entries = semantics_manifest.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("semantics_manifest entries must be a nonempty list")
    evidence = str(semantics_manifest.get("axis_semantics_evidence", ""))
    readme_marks_replica_axis = (
        "isoconfigurational" in evidence.lower()
        and "positions" in evidence.lower()
    )
    wave_numbers = [float(value) for value in semantics_manifest.get("wave_numbers", [])]
    wave_numbers_text = ";".join(f"{value:g}" for value in wave_numbers) if wave_numbers else "none"
    overlap_radius = float(semantics_manifest.get("overlap_radius", 0.0) or 0.0)

    def mean_and_se(values: Sequence[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("time-code observable values must be finite and nonempty")
        mean = float(np.mean(arr))
        if arr.size < 2:
            return mean, 0.0
        return mean, float(np.std(arr, ddof=1) / math.sqrt(arr.size))

    def member_id(path: str) -> str:
        name = path.rsplit("/", 1)[-1]
        return name[:-4] if name.endswith(".npz") else name

    out: list[dict[str, float | str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("semantics entries must be dictionaries")
        system_id = str(entry.get("system_id", "unknown"))
        temperature = str(entry.get("temperature", "none"))
        source_path = str(entry.get("source_path", "none"))
        tau_alpha = float(entry.get("tau_alpha", 0.0) or 0.0)
        time_code_map = entry.get("time_code_map", {})
        if not isinstance(time_code_map, dict) or not time_code_map:
            raise ValueError("time_code_map must be a nonempty dictionary")
        time_code_to_lag = {str(code): float(value) for code, value in time_code_map.items()}
        members = entry.get("members", [])
        if not isinstance(members, list) or not members:
            raise ValueError("members must be a nonempty list")

        by_time_code: dict[str, list[dict[str, object]]] = {}
        for member in members:
            if not isinstance(member, dict):
                raise ValueError("members must contain dictionaries")
            time_code = str(member.get("time_code", "none"))
            if time_code not in time_code_to_lag:
                raise ValueError("member time_code must be present in time_code_map")
            by_time_code.setdefault(time_code, []).append(member)

        observed_time_codes = sorted(by_time_code, key=lambda code: time_code_to_lag[code])
        all_time_codes_observed = set(observed_time_codes) == set(time_code_to_lag)
        min_member_count = min(len(group) for group in by_time_code.values())
        enough_members_everywhere = min_member_count >= int(min_members_per_time_code)
        curve_ready = bool(all_time_codes_observed and enough_members_everywhere and tau_alpha > 0.0)
        primary_blocker = (
            "none"
            if curve_ready
            else "sparse_time_code_coverage"
            if not all_time_codes_observed
            else "member_count_per_time_code"
        )
        stage = (
            "physical_timecode_curve_ready"
            if curve_ready
            else "physical_timecode_semantics_ready_sparse_coverage"
            if not all_time_codes_observed
            else "physical_timecode_semantics_ready_member_uncertainty_short"
        )

        for time_code in observed_time_codes:
            group = by_time_code[time_code]
            member_names = sorted(member_id(str(row.get("member", "none"))) for row in group)
            msd, sigma_msd = mean_and_se([float(row.get("msd", 0.0) or 0.0) for row in group])
            ngp, sigma_ngp = mean_and_se([float(row.get("ngp_2d", 0.0) or 0.0) for row in group])
            chi4, sigma_chi4 = mean_and_se(
                [float(row.get("chi4_overlap_replica", 0.0) or 0.0) for row in group]
            )
            fs_values = []
            for row in group:
                raw_fs = row.get("self_intermediate_scattering_by_k", [])
                if isinstance(raw_fs, str):
                    parsed = _parse_semicolon_float_values(
                        raw_fs,
                        name="self_intermediate_scattering_by_k",
                    )
                else:
                    parsed = [float(value) for value in raw_fs]  # type: ignore[arg-type]
                fs_values.append(parsed)
            fs_matrix = np.asarray(fs_values, dtype=float)
            if fs_matrix.ndim != 2 or fs_matrix.shape[1] == 0:
                raise ValueError("Fs time-code values must be a nonempty matrix")
            fs_means = np.mean(fs_matrix, axis=0)
            if fs_matrix.shape[0] < 2:
                fs_sigmas = np.zeros(fs_matrix.shape[1])
            else:
                fs_sigmas = np.std(fs_matrix, axis=0, ddof=1) / math.sqrt(fs_matrix.shape[0])
            axis0_is_replica = all(
                str(row.get("axis0_semantics", "")) == "isoconfigurational_trajectory_replicates"
                for row in group
            )
            lag_time = time_code_to_lag[time_code]
            out.append(
                {
                    "semantics_id": f"{semantics_id}_{system_id.lower()}_t{temperature.replace('.', '_')}_{time_code}",
                    "accession_id": accession_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_path": source_path,
                    "time_code": time_code,
                    "lag_time": lag_time,
                    "tau_alpha": tau_alpha,
                    "lag_time_over_tau_alpha": lag_time / tau_alpha if tau_alpha > 0.0 else 0.0,
                    "member_ids": ";".join(member_names),
                    "member_count": float(len(group)),
                    "min_members_per_time_code": float(min_members_per_time_code),
                    "available_time_code_count": float(len(observed_time_codes)),
                    "required_time_code_count": float(len(time_code_to_lag)),
                    "available_time_codes": ";".join(observed_time_codes),
                    "official_time_codes": ";".join(sorted(time_code_to_lag, key=lambda code: time_code_to_lag[code])),
                    "axis0_is_isoconfigurational_replica": float(axis0_is_replica and readme_marks_replica_axis),
                    "frame_axis_is_physical_time": 0.0,
                    "physical_lag_time_ready": float(tau_alpha > 0.0 and lag_time > 0.0),
                    "all_time_codes_observed": float(all_time_codes_observed),
                    "msd": msd,
                    "sigma_msd_member_sem": sigma_msd,
                    "ngp_2d": ngp,
                    "sigma_ngp_2d_member_sem": sigma_ngp,
                    "wave_numbers": wave_numbers_text,
                    "self_intermediate_scattering_by_k": ";".join(f"{value:.12g}" for value in fs_means),
                    "sigma_self_intermediate_scattering_by_k_member_sem": ";".join(
                        f"{value:.12g}" for value in fs_sigmas
                    ),
                    "self_intermediate_scattering": float(fs_means[0]),
                    "sigma_self_intermediate_scattering_member_sem": float(fs_sigmas[0]),
                    "overlap_radius": overlap_radius,
                    "chi4_overlap_replica": chi4,
                    "sigma_chi4_overlap_member_sem": sigma_chi4,
                    "timecode_curve_ready": float(curve_ready),
                    "sota_inversion_ready": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                    "primary_blocker": primary_blocker,
                    "next_required_action": "run_persistence_exchange_inversion"
                    if curve_ready
                    else "extract_members_across_all_official_time_codes"
                    if primary_blocker == "sparse_time_code_coverage"
                    else "extract_more_members_per_time_code",
                    "timecode_semantics_stage": stage,
                }
            )
    return out


def glassbench_timecode_curve_bridge(
    *,
    benchmark_id: str,
    rows: Sequence[dict[str, object]],
    required_wave_numbers: Sequence[float],
    anchor_wave_number: float,
    threshold: float = math.exp(-1.0),
    dimension: float = 2.0,
    min_lag_time_over_tau_alpha_for_diffusion: float = 3.0,
) -> list[dict[str, float | str]]:
    """Bridge corrected GlassBench time-code curves to the PE pre-inversion schema."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    if not rows:
        raise ValueError("rows must be nonempty")
    if dimension <= 0.0:
        raise ValueError("dimension must be positive")
    if min_lag_time_over_tau_alpha_for_diffusion <= 0.0:
        raise ValueError("min_lag_time_over_tau_alpha_for_diffusion must be positive")
    required = list(dict.fromkeys(float(wave_number) for wave_number in required_wave_numbers))
    if not required or any(wave_number <= 0.0 for wave_number in required):
        raise ValueError("required_wave_numbers must be positive and nonempty")
    if float(anchor_wave_number) not in required:
        raise ValueError("required_wave_numbers must include anchor_wave_number")

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row.get("system_id", "unknown")), str(row.get("temperature", "none")))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, float | str]] = []
    for (system_id, temperature), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], float(item[0][1])),
    ):
        sorted_group = sorted(group, key=lambda row: float(row.get("lag_time", 0.0)))
        ready_rows = [row for row in sorted_group if float(row.get("timecode_curve_ready", 0.0)) == 1.0]
        source_paths = sorted({str(row.get("source_path", "none")) for row in sorted_group})
        observed_time_codes = [str(row.get("time_code", "none")) for row in sorted_group]
        tau_alpha = max(float(row.get("tau_alpha", 0.0) or 0.0) for row in sorted_group)
        lag_count = len(sorted_group)

        if len(ready_rows) != len(sorted_group) or not ready_rows:
            blocker = str(
                next(
                    (
                        row.get("primary_blocker")
                        for row in sorted_group
                        if row.get("primary_blocker") != "none"
                    ),
                    "timecode_curve_ready",
                )
            )
            out.append(
                {
                    "benchmark_id": benchmark_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_paths": ";".join(source_paths) if source_paths else "none",
                    "observed_time_codes": ";".join(observed_time_codes) if observed_time_codes else "none",
                    "lag_count": float(lag_count),
                    "tau_alpha": float(tau_alpha),
                    "latest_lag_time": max(float(row.get("lag_time", 0.0) or 0.0) for row in sorted_group),
                    "latest_lag_time_over_tau_alpha": 0.0,
                    "latest_self_intermediate_scattering_anchor": 0.0,
                    "timecode_curve_ready": 0.0,
                    "real_time_observable_curve_ready": 0.0,
                    "curve_bridge_ready": 0.0,
                    "diffusion_asymptote_window_ready": 0.0,
                    "real_pe_inversion_ready": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                    "primary_blocker": blocker,
                    "next_required_action": "extract_members_across_all_official_time_codes",
                    "bridge_stage": "glassbench_timecode_curve_upstream_incomplete",
                }
            )
            continue

        bridge_rows: list[dict[str, object]] = []
        positive_uncertainty_rows = 0
        latest_anchor_fs = 0.0
        for row in ready_rows:
            wave_numbers = _parse_semicolon_float_values(row["wave_numbers"], name="wave_numbers")
            fs_values = _parse_semicolon_float_values(
                row["self_intermediate_scattering_by_k"],
                name="self_intermediate_scattering_by_k",
            )
            if len(wave_numbers) != len(fs_values):
                raise ValueError("GlassBench wave_numbers and Fs lengths must match")
            fs_lookup = {float(wave): float(value) for wave, value in zip(wave_numbers, fs_values)}
            latest_anchor_fs = fs_lookup.get(float(anchor_wave_number), latest_anchor_fs)
            sigma_fs = _parse_semicolon_float_values(
                row.get("sigma_self_intermediate_scattering_by_k_member_sem", "none"),
                name="sigma_self_intermediate_scattering_by_k_member_sem",
            )
            if (
                float(row.get("sigma_msd_member_sem", 0.0) or 0.0) > 0.0
                and float(row.get("sigma_ngp_2d_member_sem", 0.0) or 0.0) > 0.0
                and float(row.get("sigma_chi4_overlap_member_sem", 0.0) or 0.0) > 0.0
                and sigma_fs
                and all(value > 0.0 for value in sigma_fs)
            ):
                positive_uncertainty_rows += 1
            bridge_rows.append(
                {
                    "lag_time": float(row["lag_time"]),
                    "dimension": float(dimension),
                    "msd": float(row["msd"]),
                    "ngp": float(row["ngp_2d"]),
                    "wave_numbers": row["wave_numbers"],
                    "self_intermediate_scattering_by_k": row["self_intermediate_scattering_by_k"],
                    "chi4_overlap": float(row["chi4_overlap_replica"]),
                }
            )

        bridge = trajectory_observable_curve_bridge(
            benchmark_id=f"{benchmark_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
            rows=bridge_rows,
            required_wave_numbers=required,
            anchor_wave_number=anchor_wave_number,
            threshold=threshold,
        )
        latest_lag = max(float(row["lag_time"]) for row in ready_rows)
        latest_lag_over_tau = latest_lag / tau_alpha if tau_alpha > 0.0 else 0.0
        diffusion_window_ready = latest_lag_over_tau >= min_lag_time_over_tau_alpha_for_diffusion
        uncertainty_ready = positive_uncertainty_rows == len(ready_rows)
        bridge_ready = float(bridge["curve_bridge_ready"]) == 1.0
        real_inversion_ready = bridge_ready and diffusion_window_ready and uncertainty_ready
        if not bridge_ready:
            blocker = str(bridge["primary_blocker"])
            next_action = "extend_timecode_curve_until_alpha_threshold_crossing"
        elif not diffusion_window_ready:
            blocker = "diffusion_asymptote_window"
            next_action = "extend_lag_window_beyond_multiple_tau_alpha"
        elif not uncertainty_ready:
            blocker = "positive_member_uncertainties"
            next_action = "extract_more_members_for_positive_uncertainties"
        else:
            blocker = "none"
            next_action = "run_persistence_exchange_real_data_inversion"
        out.append(
            {
                **bridge,
                "benchmark_id": benchmark_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_paths": ";".join(source_paths) if source_paths else "none",
                "observed_time_codes": ";".join(observed_time_codes),
                "tau_alpha": float(tau_alpha),
                "latest_lag_time": float(latest_lag),
                "latest_lag_time_over_tau_alpha": float(latest_lag_over_tau),
                "latest_self_intermediate_scattering_anchor": float(latest_anchor_fs),
                "positive_uncertainty_row_count": float(positive_uncertainty_rows),
                "timecode_curve_ready": 1.0,
                "real_time_observable_curve_ready": 1.0,
                "diffusion_asymptote_window_ready": float(diffusion_window_ready),
                "real_pe_inversion_ready": float(real_inversion_ready),
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "next_required_action": next_action,
                "bridge_stage": "glassbench_timecode_curve_bridge_ready"
                if real_inversion_ready
                else "glassbench_timecode_curve_bridge_incomplete",
            }
        )
    return out


def glassbench_alpha_threshold_horizon_audit(
    *,
    audit_id: str,
    timecode_rows: Sequence[dict[str, object]],
    bridge_rows: Sequence[dict[str, object]],
    anchor_wave_number: float,
    threshold: float = math.exp(-1.0),
    min_extension_factor: float = 1.25,
) -> list[dict[str, float | str]]:
    """Audit whether GlassBench tau-alpha metadata matches the anchor Fs threshold."""

    if not audit_id:
        raise ValueError("audit_id must be nonempty")
    if not timecode_rows:
        raise ValueError("timecode_rows must be nonempty")
    if not bridge_rows:
        raise ValueError("bridge_rows must be nonempty")
    for name, value in {
        "anchor_wave_number": anchor_wave_number,
        "threshold": threshold,
        "min_extension_factor": min_extension_factor,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if threshold >= 1.0:
        raise ValueError("threshold must be below one")

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in timecode_rows:
        key = (str(row.get("system_id", "unknown")), str(row.get("temperature", "none")))
        grouped.setdefault(key, []).append(row)
    bridge_by_key = {
        (str(row.get("system_id", "unknown")), str(row.get("temperature", "none"))): row
        for row in bridge_rows
    }

    out: list[dict[str, float | str]] = []
    for (system_id, temperature), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], float(item[0][1])),
    ):
        sorted_group = sorted(group, key=lambda row: float(row.get("lag_time", 0.0)))
        bridge = bridge_by_key.get((system_id, temperature), {})
        source_paths = sorted({str(row.get("source_path", "none")) for row in sorted_group})
        observed_time_codes = [str(row.get("time_code", "none")) for row in sorted_group]
        tau_alpha = max(float(row.get("tau_alpha", 0.0) or 0.0) for row in sorted_group)
        latest_lag = max(float(row.get("lag_time", 0.0) or 0.0) for row in sorted_group)
        latest_lag_over_tau = latest_lag / tau_alpha if tau_alpha > 0.0 else 0.0
        real_curve_ready = (
            bool(sorted_group)
            and all(float(row.get("timecode_curve_ready", 0.0)) == 1.0 for row in sorted_group)
            and float(bridge.get("real_time_observable_curve_ready", 0.0)) == 1.0
        )

        if not real_curve_ready:
            blocker = str(
                bridge.get(
                    "primary_blocker",
                    next(
                        (
                            row.get("primary_blocker")
                            for row in sorted_group
                            if row.get("primary_blocker") != "none"
                        ),
                        "timecode_curve_ready",
                    ),
                )
            )
            out.append(
                {
                    "audit_id": audit_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_paths": ";".join(source_paths) if source_paths else "none",
                    "observed_time_codes": ";".join(observed_time_codes) if observed_time_codes else "none",
                    "lag_count": float(len(sorted_group)),
                    "tau_alpha_metadata": float(tau_alpha),
                    "latest_lag_time": float(latest_lag),
                    "latest_lag_time_over_tau_alpha_metadata": float(latest_lag_over_tau),
                    "latest_self_intermediate_scattering_anchor": 0.0,
                    "threshold": float(threshold),
                    "metadata_tau_alpha_reached": float(tau_alpha > 0.0 and latest_lag >= tau_alpha),
                    "alpha_threshold_crossed": 0.0,
                    "metadata_tau_alpha_consistent_with_anchor_fs": 0.0,
                    "estimated_threshold_lag_time": 0.0,
                    "estimated_lag_extension_factor": 0.0,
                    "extension_factor_above_minimum": 0.0,
                    "real_time_observable_curve_ready": 0.0,
                    "real_pe_inversion_ready": float(bridge.get("real_pe_inversion_ready", 0.0) or 0.0),
                    "thermodynamic_claim_allowed": 0.0,
                    "primary_blocker": blocker,
                    "next_required_action": "complete_timecode_curve_before_alpha_horizon_audit",
                    "audit_stage": "timecode_curve_upstream_incomplete",
                }
            )
            continue

        lag_times: list[float] = []
        fs_anchor: list[float] = []
        for row in sorted_group:
            wave_numbers = _parse_semicolon_float_values(row["wave_numbers"], name="wave_numbers")
            fs_values = _parse_semicolon_float_values(
                row["self_intermediate_scattering_by_k"],
                name="self_intermediate_scattering_by_k",
            )
            if len(wave_numbers) != len(fs_values):
                raise ValueError("GlassBench wave_numbers and Fs lengths must match")
            lookup = {float(wave): float(value) for wave, value in zip(wave_numbers, fs_values)}
            if float(anchor_wave_number) not in lookup:
                raise ValueError("anchor_wave_number must be present in every time-code row")
            lag = float(row["lag_time"])
            fs_value = float(lookup[float(anchor_wave_number)])
            if lag <= 0.0 or fs_value <= 0.0:
                raise ValueError("GlassBench alpha horizon rows require positive lag times and Fs")
            lag_times.append(lag)
            fs_anchor.append(fs_value)

        latest_fs = float(fs_anchor[-1])
        metadata_reached = tau_alpha > 0.0 and latest_lag >= tau_alpha
        alpha_crossed = latest_fs <= threshold
        metadata_consistent = (not metadata_reached) or alpha_crossed
        estimated_threshold_lag = latest_lag if alpha_crossed else 0.0
        estimated_extension_factor = 1.0 if alpha_crossed else 0.0
        if not alpha_crossed and len(lag_times) >= 2:
            t_prev, t_last = float(lag_times[-2]), float(lag_times[-1])
            fs_prev, fs_last = float(fs_anchor[-2]), float(fs_anchor[-1])
            if t_last > t_prev > 0.0 and fs_prev > 0.0 and fs_last > 0.0:
                slope = (math.log(fs_last) - math.log(fs_prev)) / (math.log(t_last) - math.log(t_prev))
                if slope < 0.0:
                    log_t_cross = math.log(t_last) + (math.log(threshold) - math.log(fs_last)) / slope
                    if log_t_cross > math.log(t_last):
                        estimated_threshold_lag = float(math.exp(log_t_cross))
                        estimated_extension_factor = estimated_threshold_lag / t_last

        real_pe_ready = float(bridge.get("real_pe_inversion_ready", 0.0) or 0.0)
        extension_above_minimum = estimated_extension_factor >= min_extension_factor
        if metadata_reached and not alpha_crossed:
            stage = "metadata_tau_alpha_anchor_fs_mismatch"
            blocker = "anchor_wave_number_or_alpha_definition_mismatch"
            next_action = "verify_alpha_definition_or_extend_archive_to_threshold_crossing"
        elif real_pe_ready == 1.0 and alpha_crossed:
            stage = "alpha_threshold_horizon_inversion_ready"
            blocker = "none"
            next_action = "run_persistence_exchange_real_data_inversion"
        elif alpha_crossed:
            stage = "alpha_threshold_crossed_preinversion"
            blocker = str(bridge.get("primary_blocker", "persistence_exchange_inversion"))
            next_action = str(bridge.get("next_required_action", "run_preinversion_checks"))
        else:
            stage = "alpha_threshold_not_yet_reached"
            blocker = "alpha_threshold_crossing"
            next_action = "extend_timecode_curve_until_alpha_threshold_crossing"

        out.append(
            {
                "audit_id": audit_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_paths": ";".join(source_paths) if source_paths else "none",
                "observed_time_codes": ";".join(observed_time_codes) if observed_time_codes else "none",
                "lag_count": float(len(sorted_group)),
                "tau_alpha_metadata": float(tau_alpha),
                "latest_lag_time": float(latest_lag),
                "latest_lag_time_over_tau_alpha_metadata": float(latest_lag_over_tau),
                "latest_self_intermediate_scattering_anchor": latest_fs,
                "threshold": float(threshold),
                "metadata_tau_alpha_reached": float(metadata_reached),
                "alpha_threshold_crossed": float(alpha_crossed),
                "metadata_tau_alpha_consistent_with_anchor_fs": float(metadata_consistent),
                "estimated_threshold_lag_time": float(estimated_threshold_lag),
                "estimated_lag_extension_factor": float(estimated_extension_factor),
                "extension_factor_above_minimum": float(extension_above_minimum),
                "real_time_observable_curve_ready": 1.0,
                "real_pe_inversion_ready": real_pe_ready,
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "next_required_action": next_action,
                "audit_stage": stage,
            }
        )
    return out


def glassbench_cage_jump_proxy_canary(
    *,
    canary_id: str,
    trajectory_rows: Sequence[dict[str, object]],
    min_frame_count: int = 2,
    min_member_count: float = 4.0,
) -> list[dict[str, float | str]]:
    """Extract aggregate frame-index cage-jump proxy candidates without claiming events."""

    if not canary_id:
        raise ValueError("canary_id must be nonempty")
    if not trajectory_rows:
        raise ValueError("trajectory_rows must be nonempty")
    if min_frame_count <= 0:
        raise ValueError("min_frame_count must be positive")
    if min_member_count <= 0.0:
        raise ValueError("min_member_count must be positive")

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in trajectory_rows:
        key = (str(row.get("system_id", "unknown")), str(row.get("temperature", "none")))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, float | str]] = []
    for (system_id, temperature), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], float(item[0][1])),
    ):
        sorted_group = sorted(group, key=lambda row: float(row.get("frame_index", 0.0)))
        source_paths = sorted({str(row.get("source_path", "none")) for row in sorted_group})
        usable = [
            row
            for row in sorted_group
            if float(row.get("frame_index", 0.0)) > 0.0
            and float(row.get("frame_index_uncertainty_ready", 0.0)) == 1.0
            and float(row.get("member_count", 0.0) or 0.0) >= min_member_count
        ]

        candidate_rows: list[dict[str, float]] = []
        for row in usable:
            msd = float(row.get("msd", 0.0) or 0.0)
            ngp = max(0.0, float(row.get("ngp_2d", 0.0) or 0.0))
            fs_decay = 0.0
            fs_text = str(row.get("self_intermediate_scattering_by_k", "none"))
            if fs_text and fs_text != "none":
                fs_values = _parse_semicolon_float_values(
                    fs_text,
                    name="self_intermediate_scattering_by_k",
                )
                if fs_values:
                    fs_decay = max(0.0, 1.0 - min(fs_values))
            displacement = math.sqrt(msd) if msd > 0.0 else 0.0
            score = displacement * max(ngp, 1e-12) * max(fs_decay, 1e-12)
            candidate_rows.append(
                {
                    "frame_index": float(row.get("frame_index", 0.0)),
                    "displacement": float(displacement),
                    "ngp": float(ngp),
                    "fs_decay": float(fs_decay),
                    "score": float(score),
                    "physical_time_ready": float(row.get("physical_time_ready", 0.0) or 0.0),
                }
            )

        aggregate_ready = len(candidate_rows) >= min_frame_count and any(row["score"] > 0.0 for row in candidate_rows)
        if candidate_rows:
            peak = max(candidate_rows, key=lambda row: row["score"])
            peak_ngp = max(candidate_rows, key=lambda row: row["ngp"])
            max_fs_decay = max(row["fs_decay"] for row in candidate_rows)
            physical_time_ready = aggregate_ready and all(row["physical_time_ready"] == 1.0 for row in candidate_rows)
        else:
            peak = {
                "frame_index": 0.0,
                "displacement": 0.0,
                "ngp": 0.0,
                "fs_decay": 0.0,
                "score": 0.0,
                "physical_time_ready": 0.0,
            }
            peak_ngp = peak
            max_fs_decay = 0.0
            physical_time_ready = False

        if aggregate_ready:
            stage = "aggregate_cage_jump_proxy_ready_particle_events_blocked"
            blocker = "particle_resolved_displacements"
            next_action = "extract_particle_resolved_cage_jump_events_and_physical_time_clock"
        else:
            stage = "aggregate_cage_jump_proxy_incomplete"
            blocker = "frame_index_member_ensemble_microstatistics"
            next_action = "extract_member_ensemble_frame_microstatistics"
        missing = ["particle_resolved_displacements"]
        if not physical_time_ready:
            missing.append("physical_time_semantics")
        missing.extend(["cage_identity_tracking", "persistence_exchange_event_clock"])

        out.append(
            {
                "canary_id": canary_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_paths": ";".join(source_paths) if source_paths else "none",
                "frame_count": float(len(sorted_group)),
                "usable_frame_count": float(len(candidate_rows)),
                "member_count_minimum": float(min_member_count),
                "aggregate_jump_proxy_ready": float(aggregate_ready),
                "particle_resolved_jump_events_ready": 0.0,
                "physical_time_jump_clock_ready": float(physical_time_ready),
                "persistence_exchange_event_clock_ready": 0.0,
                "peak_proxy_event_frame": float(peak["frame_index"]),
                "proxy_jump_length": float(peak["displacement"]),
                "proxy_event_score": float(peak["score"]),
                "peak_ngp_frame": float(peak_ngp["frame_index"]),
                "peak_ngp_value": float(peak_ngp["ngp"]),
                "max_short_frame_fs_decay": float(max_fs_decay),
                "missing_event_clock_inputs": ";".join(dict.fromkeys(missing)),
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "next_required_action": next_action,
                "canary_stage": stage,
            }
        )
    return out


def glassbench_event_clock_threshold_readiness_gate(
    *,
    benchmark_id: str,
    system_id: str,
    temperature: float | str,
    positions_schema_ready: bool | float,
    first_npz_observable_curve_ready: bool | float,
    member_ensemble_observable_ready: bool | float,
    particle_resolved_positions_cached: bool | float,
    physical_time_semantics_ready: bool | float,
    event_clock_threshold_protocol_available: bool | float,
    macro_heldout_observables_ready: bool | float,
) -> list[dict[str, float | str]]:
    """Gate real GlassBench event-clock threshold robustness without overclaiming.

    The synthetic threshold protocol is already available in the package. This
    gate states whether the corresponding real GlassBench trajectory inputs are
    present, including a reusable particle-resolved coordinate cache.
    """

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    if not system_id:
        raise ValueError("system_id must be nonempty")

    flags = {
        "positions_schema_ready": float(bool(positions_schema_ready)),
        "first_npz_observable_curve_ready": float(bool(first_npz_observable_curve_ready)),
        "member_ensemble_observable_ready": float(bool(member_ensemble_observable_ready)),
        "particle_resolved_positions_cached": float(bool(particle_resolved_positions_cached)),
        "physical_time_semantics_ready": float(bool(physical_time_semantics_ready)),
        "event_clock_threshold_protocol_available": float(bool(event_clock_threshold_protocol_available)),
        "macro_heldout_observables_ready": float(bool(macro_heldout_observables_ready)),
    }
    threshold_sweep_ready = float(
        bool(particle_resolved_positions_cached)
        and bool(physical_time_semantics_ready)
        and bool(event_clock_threshold_protocol_available)
    )

    blocker_order = [
        ("positions_schema_ready", "positions_schema"),
        ("first_npz_observable_curve_ready", "first_npz_observable_curve"),
        ("member_ensemble_observable_ready", "member_ensemble_observable"),
        ("particle_resolved_positions_cached", "particle_resolved_positions_cache"),
        ("physical_time_semantics_ready", "physical_time_semantics"),
        ("event_clock_threshold_protocol_available", "threshold_protocol"),
        ("macro_heldout_observables_ready", "macro_heldout_observables"),
    ]
    missing = [blocker for key, blocker in blocker_order if flags[key] == 0.0]
    if threshold_sweep_ready == 0.0 and "threshold_sweep_event_clock" not in missing:
        missing.append("threshold_sweep_event_clock")

    ready = float(
        all(value == 1.0 for value in flags.values())
        and threshold_sweep_ready == 1.0
    )
    stage = (
        "real_event_clock_threshold_robustness_ready"
        if ready
        else "real_event_clock_threshold_robustness_blocked"
    )
    primary_blocker = "none"
    if not ready:
        for candidate in [
            "positions_schema",
            "first_npz_observable_curve",
            "member_ensemble_observable",
            "particle_resolved_positions_cache",
            "physical_time_semantics",
            "threshold_sweep_event_clock",
            "macro_heldout_observables",
        ]:
            if candidate in missing:
                primary_blocker = candidate
                break

    return [
        {
            "benchmark_id": benchmark_id,
            "system_id": system_id,
            "temperature": str(temperature),
            **flags,
            "threshold_sweep_event_clock_ready": threshold_sweep_ready,
            "real_event_clock_threshold_robustness_ready": ready,
            "real_benchmark_closed_loop_ready": ready,
            "fit_parameters_from_macro_observables": 0.0,
            "thermodynamic_claim_allowed": 0.0,
            "primary_blocker": primary_blocker,
            "missing_real_threshold_inputs": ";".join(missing) if missing else "none",
            "readiness_stage": stage,
        }
    ]


def glassbench_first_npz_particle_cache_contract_gate(
    *,
    contract_id: str,
    schema_entries: Sequence[dict[str, object]],
    curve_entries: Sequence[dict[str, object]],
    cache_root: str,
    cached_particle_cache_targets: Sequence[str] | None = None,
    physical_time_semantics_ready: bool | float = False,
) -> list[dict[str, float | str]]:
    """Pin the first-NPZ coordinate cache target needed for event-clock sweeps."""

    if not contract_id:
        raise ValueError("contract_id must be nonempty")
    if not schema_entries:
        raise ValueError("schema_entries must be nonempty")
    if not curve_entries:
        raise ValueError("curve_entries must be nonempty")
    if not cache_root:
        raise ValueError("cache_root must be nonempty")

    curve_by_key = {
        (
            str(row.get("system_id", "unknown")),
            str(row.get("temperature", "none")),
            str(row.get("first_npz_member", "none")),
        ): row
        for row in curve_entries
    }
    cached_targets = {str(target) for target in (cached_particle_cache_targets or [])}
    root = cache_root.rstrip("/")

    out: list[dict[str, float | str]] = []
    for entry in sorted(
        schema_entries,
        key=lambda row: (str(row.get("system_id", "unknown")), float(row.get("temperature", 0.0))),
    ):
        system_id = str(entry.get("system_id", "unknown"))
        temperature = str(entry.get("temperature", "none"))
        first_npz_member = str(entry.get("first_npz_member", "none"))
        source_path = str(entry.get("path", "none"))
        key = (system_id, temperature, first_npz_member)
        curve = curve_by_key.get(key, {})

        positions_shape_values: list[int] = []
        positions_dtype = "none"
        for array in entry.get("arrays", []):
            if isinstance(array, dict) and array.get("name") == "positions.npy":
                positions_shape_values = [int(value) for value in array.get("shape", [])]
                positions_dtype = str(array.get("dtype", "none"))
                break
        positions_shape = "x".join(str(value) for value in positions_shape_values) if positions_shape_values else "none"
        frame_count = float(positions_shape_values[0]) if len(positions_shape_values) >= 1 else 0.0
        particle_count = float(positions_shape_values[1]) if len(positions_shape_values) >= 2 else 0.0
        spatial_dimension = float(positions_shape_values[2]) if len(positions_shape_values) >= 3 else 0.0

        schema_ready = float(bool(positions_shape_values))
        curve_ready = float(bool(curve))
        md5 = str(entry.get("npz_member_md5", "none"))
        npz_bytes = float(entry.get("npz_member_bytes", 0.0) or 0.0)
        md5_matches_curve = float(
            bool(curve)
            and md5 != "none"
            and md5 == str(curve.get("npz_member_md5", "none"))
        )
        byte_count_matches_curve = float(
            bool(curve)
            and npz_bytes > 0.0
            and npz_bytes == float(curve.get("npz_member_bytes", -1.0) or -1.0)
        )
        contract_ready = float(
            schema_ready == 1.0
            and curve_ready == 1.0
            and md5_matches_curve == 1.0
            and byte_count_matches_curve == 1.0
        )

        safe_temp = temperature.replace(".", "_")
        target = f"{root}/glassbench_{system_id.lower()}_T{safe_temp}_first_npz_positions.npz"
        cached = float(target in cached_targets)
        physical_time_ready = float(bool(physical_time_semantics_ready))
        threshold_ready = float(contract_ready == 1.0 and cached == 1.0 and physical_time_ready == 1.0)

        missing: list[str] = []
        if schema_ready == 0.0:
            missing.append("positions_schema")
        if curve_ready == 0.0:
            missing.append("first_npz_observable_curve")
        if md5_matches_curve == 0.0 or byte_count_matches_curve == 0.0:
            missing.append("npz_member_identity")
        if contract_ready == 1.0 and cached == 0.0:
            missing.append("particle_coordinate_cache")
        if contract_ready == 1.0 and cached == 1.0 and physical_time_ready == 0.0:
            missing.append("physical_time_semantics")

        if threshold_ready == 1.0:
            stage = "first_npz_particle_cache_ready_for_threshold_sweep"
            blocker = "none"
        elif contract_ready == 1.0 and cached == 0.0:
            stage = "first_npz_particle_cache_contract_ready_cache_missing"
            blocker = "persist_particle_coordinate_cache"
        elif contract_ready == 1.0:
            stage = "first_npz_particle_cache_contract_ready_time_blocked"
            blocker = "physical_time_semantics"
        else:
            stage = "first_npz_particle_cache_contract_incomplete"
            blocker = missing[0] if missing else "coordinate_cache_contract"

        out.append(
            {
                "contract_id": contract_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_path": source_path,
                "first_npz_member": first_npz_member,
                "compressed_probe_range_start": float(curve.get("compressed_probe_range_start", 0.0) or 0.0),
                "compressed_probe_range_end": float(curve.get("compressed_probe_range_end", 0.0) or 0.0),
                "compressed_probe_bytes": float(curve.get("compressed_probe_bytes", 0.0) or 0.0),
                "npz_member_bytes": npz_bytes,
                "npz_member_md5": md5,
                "positions_shape": positions_shape,
                "positions_dtype": positions_dtype,
                "frame_count": frame_count,
                "particle_count": particle_count,
                "spatial_dimension": spatial_dimension,
                "coordinate_schema_ready": schema_ready,
                "first_npz_observable_curve_ready": curve_ready,
                "npz_identity_matches_observable_curve": float(
                    md5_matches_curve == 1.0 and byte_count_matches_curve == 1.0
                ),
                "particle_cache_contract_ready": contract_ready,
                "particle_cache_target": target,
                "particle_resolved_positions_cached": cached,
                "physical_time_semantics_ready": physical_time_ready,
                "threshold_sweep_event_clock_ready": threshold_ready,
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "missing_particle_cache_inputs": ";".join(missing) if missing else "none",
                "cache_contract_stage": stage,
            }
        )
    return out


def glassbench_cached_particle_timecode_bridge(
    *,
    bridge_id: str,
    cache_rows: Sequence[dict[str, object]],
    semantics_manifest: dict[str, object],
) -> list[dict[str, float | str]]:
    """Attach official KA2D lag-time semantics to cached first-NPZ coordinates."""

    if not bridge_id:
        raise ValueError("bridge_id must be nonempty")
    if not cache_rows:
        raise ValueError("cache_rows must be nonempty")
    entries = semantics_manifest.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("semantics_manifest entries must be nonempty")

    member_lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    tau_lookup: dict[tuple[str, str], float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        system_id = str(entry.get("system_id", "unknown"))
        temperature = str(entry.get("temperature", "none"))
        tau_lookup[(system_id, temperature)] = float(entry.get("tau_alpha", 0.0) or 0.0)
        for member in entry.get("members", []):
            if isinstance(member, dict):
                member_lookup[(system_id, temperature, str(member.get("member", "none")))] = member

    out: list[dict[str, float | str]] = []
    for row in sorted(
        cache_rows,
        key=lambda item: (str(item.get("system_id", "unknown")), float(item.get("temperature", 0.0))),
    ):
        system_id = str(row.get("system_id", "unknown"))
        temperature = str(row.get("temperature", "none"))
        first_npz_member = str(row.get("first_npz_member", "none"))
        key = (system_id, temperature, first_npz_member)
        semantics = member_lookup.get(key, {})

        cached = float(row.get("particle_resolved_positions_cached", 0.0) or 0.0) == 1.0
        md5_matches = (
            bool(semantics)
            and str(row.get("npz_member_md5", "none")) == str(semantics.get("member_md5", "none"))
        )
        lag_time = float(semantics.get("lag_time", 0.0) or 0.0) if semantics else 0.0
        tau_alpha = tau_lookup.get((system_id, temperature), 0.0)
        lag_time_over_tau_alpha = (
            float(semantics.get("lag_time_over_tau_alpha", 0.0) or 0.0)
            if semantics
            else 0.0
        )
        axis0_semantics = str(semantics.get("axis0_semantics", "none")) if semantics else "none"
        axis0_replica = axis0_semantics == "isoconfigurational_trajectory_replicates"
        physical_lag_ready = cached and md5_matches and lag_time > 0.0
        frame_axis_is_time = False
        event_clock_ready = physical_lag_ready and frame_axis_is_time

        if event_clock_ready:
            stage = "cached_particle_event_clock_ready"
            blocker = "none"
            next_action = "run_particle_event_clock_threshold_sweep"
        elif physical_lag_ready and axis0_replica:
            stage = "cached_particle_lag_time_ready_event_clock_blocked"
            blocker = "frame_axis_is_isoconfigurational_replicates"
            next_action = "extract_multi_lag_particle_cache_or_true_trajectory"
        elif physical_lag_ready:
            stage = "cached_particle_lag_time_ready_frame_axis_unknown"
            blocker = "frame_axis_time_semantics"
            next_action = "verify_cached_particle_axis_semantics"
        else:
            stage = "cached_particle_timecode_semantics_incomplete"
            blocker = "timecode_member_identity" if cached else "particle_coordinate_cache"
            next_action = "match_cached_npz_member_to_official_timecode_semantics"

        out.append(
            {
                "bridge_id": bridge_id,
                "system_id": system_id,
                "temperature": temperature,
                "particle_cache_path": str(row.get("particle_cache_path", "none")),
                "first_npz_member": first_npz_member,
                "npz_member_md5": str(row.get("npz_member_md5", "none")),
                "time_code": str(semantics.get("time_code", "none")) if semantics else "none",
                "lag_time": float(lag_time),
                "tau_alpha": float(tau_alpha),
                "lag_time_over_tau_alpha": float(lag_time_over_tau_alpha),
                "positions_shape": str(row.get("positions_shape", "none")),
                "axis0_semantics": axis0_semantics,
                "replica_count": float(semantics.get("replica_count", 0.0) or 0.0) if semantics else 0.0,
                "particle_resolved_positions_cached": float(cached),
                "npz_identity_matches_timecode_semantics": float(md5_matches),
                "physical_lag_time_ready": float(physical_lag_ready),
                "axis0_is_isoconfigurational_replica": float(axis0_replica),
                "frame_axis_is_physical_time": 0.0,
                "event_clock_trajectory_ready": float(event_clock_ready),
                "threshold_sweep_event_clock_ready": float(event_clock_ready),
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "next_required_action": next_action,
                "timecode_bridge_stage": stage,
            }
        )
    return out


def glassbench_multilag_particle_cache_targets(
    *,
    target_id: str,
    semantics_manifest: dict[str, object],
    cache_rows: Sequence[dict[str, object]],
    minimum_time_codes: int,
) -> list[dict[str, float | str]]:
    """Select structure-matched multi-lag NPZ members needed for particle caches."""

    if not target_id:
        raise ValueError("target_id must be nonempty")
    if int(minimum_time_codes) != minimum_time_codes or minimum_time_codes < 2:
        raise ValueError("minimum_time_codes must be an integer >= 2")
    entries = semantics_manifest.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("semantics_manifest entries must be nonempty")

    cached_members: set[tuple[str, str, str, str]] = set()
    for row in cache_rows:
        cached = float(row.get("particle_resolved_positions_cached", 0.0) or 0.0) == 1.0
        if not cached:
            continue
        member_name = str(row.get("first_npz_member", row.get("target_member", "none")))
        member_md5 = str(row.get("npz_member_md5", row.get("target_member_md5", "none")))
        cached_members.add(
            (
                str(row.get("system_id", "unknown")),
                str(row.get("temperature", "none")),
                member_name,
                member_md5,
            )
        )

    out: list[dict[str, float | str]] = []
    for entry in sorted(
        entries,
        key=lambda item: (str(item.get("system_id", "unknown")), float(item.get("temperature", 0.0))),
    ):
        if not isinstance(entry, dict):
            continue
        system_id = str(entry.get("system_id", "unknown"))
        temperature = str(entry.get("temperature", "none"))
        source_path = str(entry.get("source_path", "none"))
        tau_alpha = float(entry.get("tau_alpha", 0.0) or 0.0)
        by_structure: dict[str, list[dict[str, object]]] = {}
        for member in entry.get("members", []):
            if not isinstance(member, dict):
                continue
            structure_id = str(member.get("structure_id", "none"))
            by_structure.setdefault(structure_id, []).append(member)

        ladders: list[tuple[int, float, str, list[dict[str, object]]]] = []
        for structure_id, members in by_structure.items():
            dedup: dict[str, dict[str, object]] = {}
            for member in members:
                time_code = str(member.get("time_code", "none"))
                current = dedup.get(time_code)
                if current is None or float(member.get("lag_time", 0.0) or 0.0) < float(
                    current.get("lag_time", 0.0) or 0.0
                ):
                    dedup[time_code] = member
            ladder = sorted(dedup.values(), key=lambda item: float(item.get("lag_time", 0.0) or 0.0))
            if not ladder:
                continue
            lag_span = float(ladder[-1].get("lag_time", 0.0) or 0.0) - float(
                ladder[0].get("lag_time", 0.0) or 0.0
            )
            ladders.append((len(ladder), lag_span, structure_id, ladder))

        def structure_sort_value(structure_id: str) -> float:
            try:
                return float(structure_id)
            except ValueError:
                return float("inf")

        if ladders:
            ladders.sort(key=lambda item: (-item[0], -item[1], structure_sort_value(item[2]), item[2]))
            _, lag_span, selected_structure_id, selected_ladder = ladders[0]
        else:
            lag_span = 0.0
            selected_structure_id = "none"
            selected_ladder = []

        target_keys = [
            (
                system_id,
                temperature,
                str(member.get("member", "none")),
                str(member.get("member_md5", "none")),
            )
            for member in selected_ladder
        ]
        cached_target_count = sum(1 for key in target_keys if key in cached_members)
        target_count = len(selected_ladder)
        official_ready = target_count >= int(minimum_time_codes)
        particle_cache_ready = official_ready and cached_target_count == target_count and target_count > 0
        event_clock_ready = False

        if event_clock_ready:
            stage = "multi_lag_particle_event_clock_ready"
            blocker = "none"
            next_action = "run_structure_matched_event_clock_threshold_sweep"
        elif particle_cache_ready:
            stage = "multi_lag_particle_cache_ready_event_clock_axis_blocked"
            blocker = "frame_axis_is_isoconfigurational_replicates"
            next_action = "derive_event_clock_from_structure_matched_replicates"
        elif official_ready:
            stage = "official_multi_lag_ladder_ready_cache_missing"
            blocker = "multi_lag_particle_cache_missing"
            next_action = "extract_structure_matched_multi_lag_npz_members"
        else:
            stage = "official_multi_lag_ladder_incomplete"
            blocker = "official_multi_lag_semantics"
            next_action = "extend_official_timecode_member_ladder"

        out.append(
            {
                "target_id": f"{target_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
                "system_id": system_id,
                "temperature": temperature,
                "source_path": source_path,
                "selected_structure_id": selected_structure_id,
                "selected_time_codes": ";".join(str(member.get("time_code", "none")) for member in selected_ladder),
                "target_members": ";".join(str(member.get("member", "none")) for member in selected_ladder),
                "target_member_md5s": ";".join(str(member.get("member_md5", "none")) for member in selected_ladder),
                "target_lag_times": ";".join(str(float(member.get("lag_time", 0.0) or 0.0)) for member in selected_ladder),
                "tau_alpha": float(tau_alpha),
                "target_member_count": float(target_count),
                "minimum_time_codes": float(minimum_time_codes),
                "cached_target_member_count": float(cached_target_count),
                "missing_target_member_count": float(max(0, target_count - cached_target_count)),
                "lag_span": float(lag_span),
                "official_multi_lag_ladder_ready": float(official_ready),
                "particle_lag_ladder_cache_ready": float(particle_cache_ready),
                "event_clock_trajectory_ready": float(event_clock_ready),
                "heldout_macro_prediction_ready": 0.0,
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "next_required_action": next_action,
                "target_stage": stage,
            }
        )
    return out


def glassbench_microdynamic_closed_loop_audit(
    *,
    audit_id: str,
    trajectory_rows: Sequence[dict[str, object]],
    signature_rows: Sequence[dict[str, object]],
    alpha_horizon_rows: Sequence[dict[str, object]],
    min_frame_count: int = 2,
    min_member_count: float = 4.0,
    required_signature_count: float = 4.0,
) -> list[dict[str, float | str]]:
    """Audit whether real GlassBench microstatistics can support held-out predictions."""

    if not audit_id:
        raise ValueError("audit_id must be nonempty")
    if not trajectory_rows:
        raise ValueError("trajectory_rows must be nonempty")
    if not signature_rows:
        raise ValueError("signature_rows must be nonempty")
    if not alpha_horizon_rows:
        raise ValueError("alpha_horizon_rows must be nonempty")
    if min_frame_count <= 0:
        raise ValueError("min_frame_count must be positive")
    for name, value in {
        "min_member_count": min_member_count,
        "required_signature_count": required_signature_count,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in trajectory_rows:
        key = (str(row.get("system_id", "unknown")), str(row.get("temperature", "none")))
        grouped.setdefault(key, []).append(row)
    signature_by_key = {
        (str(row.get("system_id", "unknown")), str(row.get("temperature", "none"))): row
        for row in signature_rows
    }
    alpha_by_key = {
        (str(row.get("system_id", "unknown")), str(row.get("temperature", "none"))): row
        for row in alpha_horizon_rows
    }

    out: list[dict[str, float | str]] = []
    for (system_id, temperature), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], float(item[0][1])),
    ):
        sorted_group = sorted(group, key=lambda row: float(row.get("frame_index", 0.0)))
        source_paths = sorted({str(row.get("source_path", "none")) for row in sorted_group})
        usable = [
            row
            for row in sorted_group
            if float(row.get("frame_index_uncertainty_ready", 0.0)) == 1.0
            and float(row.get("member_count", 0.0) or 0.0) >= min_member_count
        ]
        positive_usable = [row for row in usable if float(row.get("frame_index", 0.0)) > 0.0]
        frame_microstats_ready = len(positive_usable) >= min_frame_count
        physical_time_ready = frame_microstats_ready and all(
            float(row.get("physical_time_ready", 0.0)) == 1.0 for row in positive_usable
        )

        msd_values = np.asarray([float(row.get("msd", 0.0) or 0.0) for row in positive_usable], dtype=float)
        ngp_values = np.asarray([float(row.get("ngp_2d", 0.0) or 0.0) for row in positive_usable], dtype=float)
        fs_decay_values: list[float] = []
        for row in positive_usable:
            fs_values = _parse_semicolon_float_values(
                row.get("self_intermediate_scattering_by_k", "none"),
                name="self_intermediate_scattering_by_k",
            )
            if fs_values:
                fs_decay_values.append(max(0.0, 1.0 - min(fs_values)))
        cage_length_proxy = float(math.sqrt(float(np.median(msd_values)))) if len(msd_values) else 0.0
        short_frame_ngp_peak = float(np.max(ngp_values)) if len(ngp_values) else 0.0
        short_frame_fs_decay = float(max(fs_decay_values)) if fs_decay_values else 0.0

        signature = signature_by_key.get((system_id, temperature), {})
        macro_signature_ready = (
            float(signature.get("real_time_observable_curve_ready", 0.0) or 0.0) == 1.0
            and float(signature.get("supported_dynamical_signature_count", 0.0) or 0.0)
            >= required_signature_count
        )
        macro_timecode_ready = float(signature.get("real_time_observable_curve_ready", 0.0) or 0.0) == 1.0
        alpha = alpha_by_key.get((system_id, temperature), {})
        alpha_definition_consistent = (
            float(alpha.get("metadata_tau_alpha_consistent_with_anchor_fs", 0.0) or 0.0) == 1.0
        )
        real_pe_inversion_ready = float(signature.get("real_pe_inversion_ready", 0.0) or 0.0) == 1.0
        cage_jump_clock_ready = False
        micro_to_macro_prediction_ready = (
            frame_microstats_ready
            and physical_time_ready
            and cage_jump_clock_ready
            and macro_signature_ready
            and alpha_definition_consistent
            and real_pe_inversion_ready
        )
        closed_loop_ready = micro_to_macro_prediction_ready

        missing: list[str] = []
        if not frame_microstats_ready:
            missing.append("frame_index_member_ensemble_microstatistics")
        if not physical_time_ready:
            missing.append("physical_time_semantics")
        missing.append("cage_jump_event_segmentation")
        missing.append("persistence_exchange_event_clock")
        if not macro_timecode_ready:
            missing.append("macro_timecode_curve")
        if macro_timecode_ready and not macro_signature_ready:
            missing.append("macro_dynamical_signatures")
        if macro_timecode_ready and not alpha_definition_consistent:
            missing.append("alpha_definition_consistency")
        if macro_timecode_ready and not real_pe_inversion_ready:
            missing.append("real_persistence_exchange_inversion")
        missing = list(dict.fromkeys(missing))

        if closed_loop_ready:
            stage = "real_microdynamic_closed_loop_ready"
            blocker = "none"
            next_action = "run_heldout_micro_to_macro_prediction"
        elif not macro_timecode_ready:
            stage = "macro_timecode_upstream_incomplete"
            blocker = str(signature.get("primary_blocker", "macro_timecode_curve"))
            next_action = "complete_real_timecode_curve_before_closed_loop_prediction"
        elif frame_microstats_ready and macro_signature_ready:
            stage = "real_microstats_macro_signatures_closed_loop_blocked"
            blocker = missing[0] if missing else "closed_loop_prediction"
            next_action = "attach_frame_time_mapping_and_extract_cage_jump_events"
        elif frame_microstats_ready:
            stage = "real_microstats_macro_signature_incomplete"
            blocker = "macro_dynamical_signatures"
            next_action = "complete_real_timecode_signature_support"
        else:
            stage = "trajectory_microstatistics_upstream_incomplete"
            blocker = missing[0] if missing else "trajectory_microstatistics"
            next_action = "extract_member_ensemble_frame_microstatistics"

        out.append(
            {
                "audit_id": audit_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_paths": ";".join(source_paths) if source_paths else "none",
                "frame_count": float(len(sorted_group)),
                "usable_frame_count": float(len(positive_usable)),
                "member_count_minimum": float(min_member_count),
                "cage_length_proxy": cage_length_proxy,
                "short_frame_ngp_peak": short_frame_ngp_peak,
                "short_frame_fs_decay": short_frame_fs_decay,
                "frame_index_microstats_ready": float(frame_microstats_ready),
                "physical_time_microstats_ready": float(physical_time_ready),
                "cage_jump_clock_ready": 0.0,
                "macro_timecode_ready": float(macro_timecode_ready),
                "macro_signature_ready": float(macro_signature_ready),
                "macro_signature_count": float(
                    signature.get("supported_dynamical_signature_count", 0.0) or 0.0
                ),
                "alpha_definition_consistent": float(alpha_definition_consistent),
                "real_pe_inversion_ready": float(real_pe_inversion_ready),
                "micro_to_macro_prediction_ready": float(micro_to_macro_prediction_ready),
                "closed_loop_ready": float(closed_loop_ready),
                "missing_closed_loop_inputs": ";".join(missing) if missing else "none",
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "next_required_action": next_action,
                "closed_loop_stage": stage,
            }
        )
    return out


def glassbench_timecode_signature_support_gate(
    *,
    support_id: str,
    timecode_rows: Sequence[dict[str, object]],
    bridge_rows: Sequence[dict[str, object]],
    anchor_wave_number: float,
    alpha_threshold: float = math.exp(-1.0),
    min_msd_growth_factor: float = 10.0,
    min_fs_decay: float = 0.05,
    min_peak_to_initial_factor: float = 3.0,
    min_late_recovery_fraction: float = 0.1,
) -> list[dict[str, float | str]]:
    """Score real GlassBench time-code curves against dynamical glass signatures."""

    if not support_id:
        raise ValueError("support_id must be nonempty")
    if not timecode_rows:
        raise ValueError("timecode_rows must be nonempty")
    if not bridge_rows:
        raise ValueError("bridge_rows must be nonempty")
    for name, value in {
        "anchor_wave_number": anchor_wave_number,
        "alpha_threshold": alpha_threshold,
        "min_msd_growth_factor": min_msd_growth_factor,
        "min_fs_decay": min_fs_decay,
        "min_peak_to_initial_factor": min_peak_to_initial_factor,
        "min_late_recovery_fraction": min_late_recovery_fraction,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if alpha_threshold >= 1.0:
        raise ValueError("alpha_threshold must be below one")

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in timecode_rows:
        key = (str(row.get("system_id", "unknown")), str(row.get("temperature", "none")))
        grouped.setdefault(key, []).append(row)
    bridge_by_key = {
        (str(row.get("system_id", "unknown")), str(row.get("temperature", "none"))): row
        for row in bridge_rows
    }

    out: list[dict[str, float | str]] = []
    for (system_id, temperature), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], float(item[0][1])),
    ):
        sorted_group = sorted(group, key=lambda row: float(row.get("lag_time", 0.0)))
        bridge = bridge_by_key.get((system_id, temperature), {})
        real_curve_ready = (
            bool(sorted_group)
            and all(float(row.get("timecode_curve_ready", 0.0)) == 1.0 for row in sorted_group)
            and float(bridge.get("real_time_observable_curve_ready", 0.0)) == 1.0
        )
        source_paths = sorted({str(row.get("source_path", "none")) for row in sorted_group})
        if not real_curve_ready:
            blocker = str(bridge.get("primary_blocker", sorted_group[0].get("primary_blocker", "timecode_curve_ready")))
            out.append(
                {
                    "support_id": support_id,
                    "system_id": system_id,
                    "temperature": temperature,
                    "source_paths": ";".join(source_paths) if source_paths else "none",
                    "lag_count": float(len(sorted_group)),
                    "real_time_observable_curve_ready": 0.0,
                    "real_pe_inversion_ready": float(bridge.get("real_pe_inversion_ready", 0.0) or 0.0),
                    "msd_growth_factor": 0.0,
                    "msd_growth_signature": 0.0,
                    "self_intermediate_decay": 0.0,
                    "self_intermediate_decay_signature": 0.0,
                    "ngp_peak_time": 0.0,
                    "ngp_peak_value": 0.0,
                    "ngp_late_recovery_fraction": 0.0,
                    "transient_ngp_peak_signature": 0.0,
                    "chi4_peak_time": 0.0,
                    "chi4_peak_value": 0.0,
                    "chi4_late_recovery_fraction": 0.0,
                    "transient_chi4_peak_signature": 0.0,
                    "latest_self_intermediate_scattering_anchor": 0.0,
                    "alpha_threshold_crossed": 0.0,
                    "supported_dynamical_signature_count": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                    "primary_blocker": blocker,
                    "next_required_action": "complete_timecode_curve_before_signature_scoring",
                    "signature_stage": "timecode_curve_upstream_incomplete",
                }
            )
            continue

        lag_times = np.asarray([float(row["lag_time"]) for row in sorted_group], dtype=float)
        msd = np.asarray([float(row["msd"]) for row in sorted_group], dtype=float)
        ngp = np.asarray([float(row["ngp_2d"]) for row in sorted_group], dtype=float)
        chi4 = np.asarray([float(row["chi4_overlap_replica"]) for row in sorted_group], dtype=float)
        fs_anchor: list[float] = []
        for row in sorted_group:
            wave_numbers = _parse_semicolon_float_values(row["wave_numbers"], name="wave_numbers")
            fs_values = _parse_semicolon_float_values(
                row["self_intermediate_scattering_by_k"],
                name="self_intermediate_scattering_by_k",
            )
            if len(wave_numbers) != len(fs_values):
                raise ValueError("GlassBench wave_numbers and Fs lengths must match")
            lookup = {float(wave): float(value) for wave, value in zip(wave_numbers, fs_values)}
            if float(anchor_wave_number) not in lookup:
                raise ValueError("anchor_wave_number must be present in every time-code row")
            fs_anchor.append(lookup[float(anchor_wave_number)])
        fs = np.asarray(fs_anchor, dtype=float)
        if np.any(msd <= 0.0) or np.any(fs <= 0.0) or np.any(chi4 < 0.0) or np.any(ngp < 0.0):
            raise ValueError("GlassBench signature observables must be nonnegative with positive MSD and Fs")

        msd_growth_factor = float(msd[-1] / msd[0])
        fs_decay = float(fs[0] - fs[-1])
        ngp_peak_index = int(np.argmax(ngp))
        ngp_peak_value = float(ngp[ngp_peak_index])
        ngp_late_recovery = (
            (ngp_peak_value - float(ngp[-1])) / ngp_peak_value if ngp_peak_value > 0.0 else 0.0
        )
        chi4_peak_index = int(np.argmax(chi4))
        chi4_peak_value = float(chi4[chi4_peak_index])
        chi4_late_recovery = (
            (chi4_peak_value - float(chi4[-1])) / chi4_peak_value if chi4_peak_value > 0.0 else 0.0
        )
        msd_signature = msd_growth_factor >= min_msd_growth_factor
        fs_signature = fs_decay >= min_fs_decay
        ngp_signature = (
            ngp_peak_index < len(ngp) - 1
            and ngp_peak_value >= min_peak_to_initial_factor * max(float(ngp[0]), 1e-15)
            and ngp_late_recovery >= min_late_recovery_fraction
        )
        chi4_signature = (
            chi4_peak_index < len(chi4) - 1
            and chi4_peak_value >= min_peak_to_initial_factor * max(float(chi4[0]), 1e-15)
            and chi4_late_recovery >= min_late_recovery_fraction
        )
        alpha_crossed = float(fs[-1] <= alpha_threshold)
        supported_count = float(sum([msd_signature, fs_signature, ngp_signature, chi4_signature]))
        real_pe_ready = float(bridge.get("real_pe_inversion_ready", 0.0) or 0.0)
        if real_pe_ready == 1.0:
            stage = "real_curve_dynamic_signature_support_and_inversion_ready"
            blocker = "none"
            next_action = "run_persistence_exchange_real_data_inversion"
        else:
            stage = "real_curve_dynamic_signature_support_preinversion"
            blocker = str(bridge.get("primary_blocker", "persistence_exchange_inversion"))
            next_action = str(bridge.get("next_required_action", "extend_real_timecode_curve"))

        out.append(
            {
                "support_id": support_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_paths": ";".join(source_paths) if source_paths else "none",
                "lag_count": float(len(sorted_group)),
                "real_time_observable_curve_ready": 1.0,
                "real_pe_inversion_ready": real_pe_ready,
                "msd_growth_factor": msd_growth_factor,
                "msd_growth_signature": float(msd_signature),
                "self_intermediate_decay": fs_decay,
                "self_intermediate_decay_signature": float(fs_signature),
                "ngp_peak_time": float(lag_times[ngp_peak_index]),
                "ngp_peak_value": ngp_peak_value,
                "ngp_late_recovery_fraction": float(ngp_late_recovery),
                "transient_ngp_peak_signature": float(ngp_signature),
                "chi4_peak_time": float(lag_times[chi4_peak_index]),
                "chi4_peak_value": chi4_peak_value,
                "chi4_late_recovery_fraction": float(chi4_late_recovery),
                "transient_chi4_peak_signature": float(chi4_signature),
                "latest_self_intermediate_scattering_anchor": float(fs[-1]),
                "alpha_threshold_crossed": alpha_crossed,
                "supported_dynamical_signature_count": supported_count,
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "next_required_action": next_action,
                "signature_stage": stage,
            }
        )
    return out


def dynamic_signature_alignment_ledger(
    *,
    alignment_id: str,
    claim_rows: Sequence[dict[str, object]],
    literature_rows: Sequence[dict[str, object]],
    glassbench_signature_rows: Sequence[dict[str, object]],
) -> list[dict[str, float | str]]:
    """Align model diagnostics, literature claims, and real GlassBench signature support."""

    if not alignment_id:
        raise ValueError("alignment_id must be nonempty")
    if not claim_rows:
        raise ValueError("claim_rows must be nonempty")
    if not literature_rows:
        raise ValueError("literature_rows must be nonempty")
    if not glassbench_signature_rows:
        raise ValueError("glassbench_signature_rows must be nonempty")

    claim_by_phenomenon = {str(row.get("phenomenon", "")): row for row in claim_rows}
    literature_sources = {str(row.get("benchmark_source", "")) for row in literature_rows}
    real_rows = [
        row
        for row in glassbench_signature_rows
        if float(row.get("real_time_observable_curve_ready", 0.0) or 0.0) == 1.0
    ]
    real = real_rows[0] if real_rows else {}
    real_ready = float(real.get("real_time_observable_curve_ready", 0.0) or 0.0)
    real_inversion_ready = float(real.get("real_pe_inversion_ready", 0.0) or 0.0)
    real_blocker = str(real.get("primary_blocker", "real_glassbench_signature_support"))

    def claim_support(phenomenon: str) -> tuple[float, str, str]:
        row = claim_by_phenomenon.get(phenomenon, {})
        alignment = str(row.get("claim_alignment", "missing"))
        support_level = str(row.get("model_support_level", "missing"))
        model_support = 1.0 if alignment == "supported" else 0.5 if alignment == "partial" else 0.0
        blocker = str(row.get("primary_blocker", "claim_alignment"))
        return model_support, support_level, blocker

    def literature_support(required_sources: Sequence[str]) -> float:
        return float(all(source in literature_sources for source in required_sources))

    specs = [
        {
            "signature": "msd_growth_cage_escape",
            "phenomenon": "cage_plateau_transient_ngp_van_hove_tail",
            "sources": ["kob1995vanhove"],
            "real_support": float(real.get("msd_growth_signature", 0.0) or 0.0),
            "stage_if_real": "real_curve_supported",
            "blocker": "none",
        },
        {
            "signature": "self_intermediate_alpha",
            "phenomenon": "self_intermediate_scattering_alpha_relaxation",
            "sources": ["kob1995intermediate"],
            "real_support": float(real.get("self_intermediate_decay_signature", 0.0) or 0.0),
            "stage_if_real": "real_curve_supported_pre_alpha_threshold"
            if float(real.get("alpha_threshold_crossed", 0.0) or 0.0) == 0.0
            else "real_curve_supported",
            "blocker": real_blocker if float(real.get("alpha_threshold_crossed", 0.0) or 0.0) == 0.0 else "none",
        },
        {
            "signature": "transient_ngp_peak",
            "phenomenon": "cage_plateau_transient_ngp_van_hove_tail",
            "sources": ["kob1995vanhove"],
            "real_support": float(real.get("transient_ngp_peak_signature", 0.0) or 0.0),
            "stage_if_real": "real_curve_supported",
            "blocker": "none",
        },
        {
            "signature": "chi4_dynamic_heterogeneity_proxy",
            "phenomenon": "chi4_peak_and_dynamic_length_growth",
            "sources": ["lacevic2003fourpoint"],
            "real_support": float(real.get("transient_chi4_peak_signature", 0.0) or 0.0),
            "stage_if_real": "real_proxy_supported_spatial_boundary",
            "blocker": "direct_four_point_function_and_dynamic_length",
        },
        {
            "signature": "persistence_exchange_decoupling",
            "phenomenon": "persistence_exchange_decoupling",
            "sources": ["hedges2007persistence"],
            "real_support": 0.0,
            "stage_if_real": "model_literature_supported_real_inversion_blocked",
            "blocker": real_blocker,
        },
        {
            "signature": "thermodynamic_transition",
            "phenomenon": "configurational_entropy_and_ideal_glass_scope",
            "sources": [],
            "real_support": 0.0,
            "stage_if_real": "scope_boundary_not_explained",
            "blocker": "thermodynamic_input_law",
        },
    ]

    rows: list[dict[str, float | str]] = []
    for spec in specs:
        signature = str(spec["signature"])
        phenomenon = str(spec["phenomenon"])
        model_support, model_support_level, claim_blocker = claim_support(phenomenon)
        lit_support = literature_support(spec["sources"])  # type: ignore[arg-type]
        real_support = float(spec["real_support"])
        if signature == "thermodynamic_transition":
            stage = "scope_boundary_not_explained"
            blocker = str(spec["blocker"])
        elif real_support == 1.0:
            stage = str(spec["stage_if_real"])
            blocker = str(spec["blocker"])
        elif model_support > 0.0 and lit_support == 1.0:
            stage = str(spec["stage_if_real"])
            blocker = str(spec["blocker"])
        elif model_support > 0.0:
            stage = "model_supported_literature_or_real_data_pending"
            blocker = claim_blocker
        else:
            stage = "unsupported_or_scope_boundary"
            blocker = claim_blocker
        if blocker == "none" and real_inversion_ready == 0.0 and signature in {
            "self_intermediate_alpha",
            "persistence_exchange_decoupling",
        }:
            blocker = real_blocker
        rows.append(
            {
                "alignment_id": alignment_id,
                "signature": signature,
                "phenomenon": phenomenon,
                "model_support": float(model_support),
                "model_support_level": model_support_level,
                "literature_qualitative_support": lit_support,
                "real_glassbench_support": real_support,
                "real_time_observable_curve_ready": real_ready,
                "real_quantitative_inversion_ready": real_inversion_ready,
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "alignment_stage": stage,
            }
        )
    return rows


def sota_glassbench_visible_member_ensemble_audit_gate(
    *,
    audit_id: str,
    accession_id: str,
    tar_probe_rows: Sequence[dict[str, float | str]],
    ensemble_horizon_rows: Sequence[dict[str, float | str]],
    member_index_rows: Sequence[dict[str, float | str]] | None = None,
) -> list[dict[str, float | str]]:
    """Audit visible NPZ member evidence before treating a prefix as an ensemble."""

    if not audit_id:
        raise ValueError("audit_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not tar_probe_rows:
        raise ValueError("tar_probe_rows must be nonempty")
    if not ensemble_horizon_rows:
        raise ValueError("ensemble_horizon_rows must be nonempty")

    horizon_by_key = {
        (str(row.get("system_id", "unknown")), str(row.get("temperature", "none"))): row
        for row in ensemble_horizon_rows
    }
    index_by_key = {
        (str(row.get("system_id", "unknown")), str(row.get("temperature", "none"))): row
        for row in (member_index_rows or [])
    }

    def split_items(value: object) -> list[str]:
        text = str(value)
        if not text or text == "none":
            return []
        return [item for item in text.split(";") if item and item != "none"]

    def member_id(path: str) -> str:
        if not path or path == "none":
            return "none"
        name = path.rsplit("/", 1)[-1]
        return name[:-4] if name.endswith(".npz") else name

    out: list[dict[str, float | str]] = []
    for row in sorted(
        tar_probe_rows,
        key=lambda item: (str(item.get("system_id", "unknown")), str(item.get("temperature", "none"))),
    ):
        system_id = str(row.get("system_id", "unknown"))
        temperature = str(row.get("temperature", "none"))
        if system_id == "none" or temperature == "none":
            continue
        horizon = horizon_by_key.get((system_id, temperature), {})
        index_row = index_by_key.get((system_id, temperature), {})
        layout_ready = bool(float(row.get("trajectory_layout_ready", 0.0)))
        first_member = str(row.get("first_npz_member", "none"))
        first_id = member_id(first_member)
        split_labels = str(index_row.get("split_labels_in_index", row.get("split_labels_in_probe", "none")))
        split_items_visible = split_items(split_labels)
        visible_member_paths = split_items(index_row.get("visible_npz_members", row.get("visible_npz_members", "none")))
        prefix_count = float(
            index_row.get(
                "indexed_npz_member_count",
                horizon.get("prefix_npz_member_count", row.get("npz_member_count_in_probe", 0.0)),
            )
        )
        required_count = float(horizon.get("min_member_count", 4.0))
        additional_needed = max(0.0, required_count - prefix_count)
        threshold_pass = prefix_count >= required_count
        first_visible = first_id != "none"
        full_list_visible = len(visible_member_paths) >= int(required_count) and len(visible_member_paths) >= int(prefix_count)
        split_policy_documented = bool(split_items_visible)
        ready = bool(layout_ready and threshold_pass and first_visible and full_list_visible and split_policy_documented)

        if ready:
            stage = "visible_member_ensemble_ready_for_uncertainty"
            blocker = "none"
            next_actions = ["compute_member_resolved_observables_and_uncertainties"]
        elif not layout_ready:
            stage = "visible_member_ensemble_layout_blocked"
            blocker = "trajectory_layout"
            next_actions = ["complete_inner_tar_layout_probe"]
        elif not threshold_pass or not full_list_visible:
            stage = "visible_prefix_not_publishable_ensemble"
            blocker = "member_count_and_full_member_list"
            next_actions = [
                "index_full_npz_member_list",
                "extract_at_least_4_independent_members_per_temperature",
                "keep_split_policy_fixed_within_temperature",
            ]
        elif not split_policy_documented:
            stage = "visible_member_split_policy_missing"
            blocker = "split_policy"
            next_actions = ["document_train_test_split_policy_for_member_ensemble"]
        else:
            stage = "visible_member_ensemble_incomplete"
            blocker = "member_identity"
            next_actions = ["verify_independent_member_identifiers"]

        out.append(
            {
                "audit_id": f"{audit_id}_{system_id.lower()}_t{temperature.replace('.', '_')}",
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "source_path": str(row.get("source_path", "none")),
                "first_npz_member": first_member,
                "first_member_id": first_id,
                "split_labels_in_probe": split_labels,
                "prefix_npz_member_count": float(prefix_count),
                "required_member_count": float(required_count),
                "additional_member_count_needed": float(additional_needed),
                "visible_member_list_count": float(len(visible_member_paths)),
                "first_member_id_visible": float(first_visible),
                "full_member_id_list_visible": float(full_list_visible),
                "split_policy_documented": float(split_policy_documented),
                "member_count_threshold_pass": float(threshold_pass),
                "publishable_ensemble_uncertainty_ready": float(ready),
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": blocker,
                "next_required_actions": ";".join(next_actions),
                "ensemble_audit_stage": stage,
            }
        )

    if not out:
        raise ValueError("tar_probe_rows did not contain any concrete system-temperature rows")
    return out


def sota_remote_result_curve_cache_gate(
    *,
    curve_cache_id: str,
    accession_id: str,
    source_id: str,
    manifest: dict,
    required_roles_by_system: dict[str, Sequence[str]],
    max_uncompressed_size_bytes: int,
) -> list[dict[str, float | str]]:
    """Verify small result curves extracted by remote byte-range reads."""

    for name, value in {
        "curve_cache_id": curve_cache_id,
        "accession_id": accession_id,
        "source_id": source_id,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    if max_uncompressed_size_bytes <= 0:
        raise ValueError("max_uncompressed_size_bytes must be positive")
    if not required_roles_by_system:
        raise ValueError("required_roles_by_system must be nonempty")

    entries_value = manifest.get("entries", [])
    entries = [entry for entry in entries_value if isinstance(entry, dict)] if isinstance(entries_value, list) else []
    systems = sorted(
        set(required_roles_by_system)
        | {str(entry.get("system_id", "")) for entry in entries if entry.get("system_id")}
    )
    rows: list[dict[str, float | str]] = []
    for system in systems:
        system_entries = [entry for entry in entries if str(entry.get("system_id", "")) == system]
        required_roles = list(dict.fromkeys(required_roles_by_system.get(system, [])))
        available_roles = sorted(
            {str(entry.get("curve_role", "")) for entry in system_entries if entry.get("curve_role")}
        )
        temperatures = []
        for temperature in {str(entry.get("temperature", "")) for entry in system_entries if entry.get("temperature")}:
            try:
                float(temperature)
            except ValueError:
                continue
            temperatures.append(temperature)
        temperatures = sorted(temperatures, key=float)
        missing_roles = [role for role in required_roles if role not in set(available_roles)]
        crc_ok = bool(system_entries) and all(bool(entry.get("crc32_matches", False)) for entry in system_entries)
        md5_ok = bool(system_entries) and all(bool(entry.get("md5", "")) for entry in system_entries)
        size_ok = bool(system_entries) and all(
            0 < int(entry.get("uncompressed_size_bytes", 0) or 0) <= max_uncompressed_size_bytes
            for entry in system_entries
        )
        numeric_ok = bool(system_entries) and all(
            int(entry.get("numeric_row_count", 0) or 0) > 0
            and int(entry.get("numeric_column_count", 0) or 0) > 0
            for entry in system_entries
        )
        range_ok = bool(system_entries) and all(
            int(entry.get("range_start", -1) or -1) >= 0
            and int(entry.get("range_end", -1) or -1) >= int(entry.get("range_start", 0) or 0)
            for entry in system_entries
        )
        ready = not missing_roles and crc_ok and md5_ok and size_ok and numeric_ok and range_ok

        if ready:
            stage = "range_result_curves_verified"
            blocker = "raw_curve_adapter"
        elif not system_entries:
            stage = "range_result_curves_missing"
            blocker = "curve_entries"
        elif missing_roles:
            stage = "range_result_curve_roles_incomplete"
            blocker = "curve_roles"
        elif not crc_ok:
            stage = "range_result_curve_crc_mismatch"
            blocker = "crc32"
        elif not md5_ok:
            stage = "range_result_curve_digest_missing"
            blocker = "md5"
        elif not size_ok:
            stage = "range_result_curve_size_blocked"
            blocker = "curve_size"
        elif not numeric_ok:
            stage = "range_result_curve_parse_blocked"
            blocker = "numeric_rows"
        else:
            stage = "range_result_curve_range_missing"
            blocker = "byte_range"

        rows.append(
            {
                "curve_cache_id": f"{curve_cache_id}_{system.lower()}",
                "accession_id": accession_id,
                "source_id": source_id,
                "system_id": system,
                "curve_file_count": float(len(system_entries)),
                "required_roles": ";".join(required_roles) if required_roles else "none",
                "available_roles": ";".join(available_roles) if available_roles else "none",
                "missing_roles": ";".join(missing_roles) if missing_roles else "none",
                "temperature_grid": ";".join(temperatures) if temperatures else "none",
                "temperature_count": float(len(temperatures)),
                "crc32_verified": float(crc_ok),
                "md5_available": float(md5_ok),
                "size_within_limit": float(size_ok),
                "numeric_parse_ready": float(numeric_ok),
                "range_fetch_ready": float(range_ok),
                "curve_cache_ready": float(ready),
                "real_inversion_ready": 0.0,
                "primary_blocker": blocker,
                "curve_cache_stage": stage,
            }
        )
    return rows


def sota_remote_result_curve_fetch_gap_gate(
    *,
    gap_id: str,
    accession_id: str,
    central_directory_manifest: dict,
    range_cache_manifest: dict,
    target_curve_specs: Sequence[dict[str, str]],
) -> list[dict[str, float | str]]:
    """Mark SOTA result curves visible in the archive but absent from range cache."""

    if not gap_id:
        raise ValueError("gap_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not target_curve_specs:
        raise ValueError("target_curve_specs must be nonempty")

    central_entries_value = central_directory_manifest.get("entries", [])
    if not isinstance(central_entries_value, list):
        central_entries_value = []
    central_paths = {
        str(entry.get("path", "")) if isinstance(entry, dict) else str(entry)
        for entry in central_entries_value
    }

    cache_entries_value = range_cache_manifest.get("entries", [])
    cache_entries = (
        [entry for entry in cache_entries_value if isinstance(entry, dict)]
        if isinstance(cache_entries_value, list)
        else []
    )
    cache_by_path = {str(entry.get("path", "")): entry for entry in cache_entries if entry.get("path")}

    rows: list[dict[str, float | str]] = []
    for spec in target_curve_specs:
        target_path = str(spec.get("path", ""))
        system_id = str(spec.get("system_id", ""))
        temperature = str(spec.get("temperature", "none"))
        curve_role = str(spec.get("curve_role", ""))
        candidate_observable = str(spec.get("candidate_observable", "unmapped_result_curve"))
        if not target_path or not system_id or not curve_role:
            raise ValueError("target specs require path, system_id, and curve_role")

        central_present = target_path in central_paths
        cache_entry = cache_by_path.get(target_path)
        cache_present = cache_entry is not None
        numeric_ready = bool(
            cache_entry
            and bool(cache_entry.get("crc32_matches", False))
            and int(cache_entry.get("numeric_row_count", 0) or 0) > 0
            and int(cache_entry.get("numeric_column_count", 0) or 0) > 0
            and int(cache_entry.get("range_start", -1) or -1) >= 0
            and int(cache_entry.get("range_end", -1) or -1) >= int(cache_entry.get("range_start", 0) or 0)
        )

        if not central_present:
            stage = "remote_target_missing"
            blocker = "central_directory"
            targeted_fetch_ready = False
            comparison_ready = False
        elif not cache_present:
            stage = "remote_target_present_range_cache_missing"
            blocker = "range_result_curve_cache"
            targeted_fetch_ready = True
            comparison_ready = False
        elif not numeric_ready:
            stage = "range_cache_target_parse_blocked"
            blocker = "range_cache_numeric_payload"
            targeted_fetch_ready = True
            comparison_ready = False
        else:
            stage = "range_cache_target_ready_for_observable_comparison"
            blocker = "joint_observable_protocol"
            targeted_fetch_ready = False
            comparison_ready = True

        rows.append(
            {
                "gap_id": f"{gap_id}_{system_id.lower()}_{temperature}_{curve_role}",
                "accession_id": accession_id,
                "system_id": system_id,
                "temperature": temperature,
                "curve_role": curve_role,
                "target_path": target_path,
                "candidate_observable": candidate_observable,
                "central_directory_present": float(central_present),
                "range_cache_present": float(cache_present),
                "range_cache_numeric_ready": float(numeric_ready),
                "targeted_fetch_ready": float(targeted_fetch_ready),
                "observable_comparison_ready": float(comparison_ready),
                "real_inversion_ready": 0.0,
                "primary_blocker": blocker,
                "fetch_gap_stage": stage,
            }
        )
    return rows


def sota_remote_result_curve_target_fetch_gate(
    *,
    target_fetch_id: str,
    accession_id: str,
    central_directory_manifest: dict,
    target_fetch_manifest: dict,
    target_paths: Sequence[str],
) -> list[dict[str, float | str]]:
    """Classify targeted result-curve range fetches before observable comparison."""

    if not target_fetch_id:
        raise ValueError("target_fetch_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    targets = list(dict.fromkeys(target_paths))
    if not targets or any(not path for path in targets):
        raise ValueError("target_paths must contain nonempty strings")

    central_entries_value = central_directory_manifest.get("entries", [])
    if not isinstance(central_entries_value, list):
        central_entries_value = []
    central_paths = {
        str(entry.get("path", "")) if isinstance(entry, dict) else str(entry)
        for entry in central_entries_value
    }

    fetch_entries_value = target_fetch_manifest.get("entries", [])
    fetch_entries = (
        [entry for entry in fetch_entries_value if isinstance(entry, dict)]
        if isinstance(fetch_entries_value, list)
        else []
    )
    fetch_by_path = {str(entry.get("path", "")): entry for entry in fetch_entries if entry.get("path")}

    rows: list[dict[str, float | str]] = []
    for target_path in targets:
        entry = fetch_by_path.get(target_path)
        central_present = target_path in central_paths
        fetch_present = entry is not None
        checksum_ready = bool(
            entry
            and bool(entry.get("crc32_matches", False))
            and bool(str(entry.get("md5", "")))
            and int(entry.get("uncompressed_size_bytes", 0) or 0) > 0
        )
        header = entry.get("header", []) if entry else []
        header_ready = isinstance(header, list) and len(header) > 0
        numeric_rows = int(entry.get("numeric_row_count", 0) or 0) if entry else 0
        numeric_columns = int(entry.get("numeric_column_count", 0) or 0) if entry else 0
        numeric_ready = fetch_present and checksum_ready and numeric_rows > 0 and numeric_columns > 0
        header_only = fetch_present and checksum_ready and header_ready and not numeric_ready

        if not central_present:
            stage = "remote_target_missing"
            blocker = "central_directory"
        elif not fetch_present:
            stage = "target_fetch_missing"
            blocker = "target_range_fetch"
        elif not checksum_ready:
            stage = "target_fetch_checksum_blocked"
            blocker = "checksum"
        elif header_only:
            stage = "target_fetch_header_only_parse_blocked"
            blocker = "numeric_rows"
        elif not numeric_ready:
            stage = "target_fetch_numeric_parse_blocked"
            blocker = "numeric_payload"
        else:
            stage = "target_fetch_numeric_ready_for_observable_comparison"
            blocker = "joint_observable_protocol"

        rows.append(
            {
                "target_fetch_id": f"{target_fetch_id}_{str(entry.get('system_id', 'unknown') if entry else 'unknown').lower()}_{str(entry.get('temperature', 'none') if entry else 'none')}_{str(entry.get('curve_role', 'unknown') if entry else 'unknown')}",
                "accession_id": accession_id,
                "system_id": str(entry.get("system_id", "unknown")) if entry else "unknown",
                "temperature": str(entry.get("temperature", "none")) if entry else "none",
                "curve_role": str(entry.get("curve_role", "unknown")) if entry else "unknown",
                "target_path": target_path,
                "candidate_observable": str(entry.get("candidate_observable", "unmapped_result_curve")) if entry else "unmapped_result_curve",
                "central_directory_present": float(central_present),
                "target_fetch_present": float(fetch_present),
                "target_fetch_checksum_ready": float(checksum_ready),
                "header_only_payload": float(header_only),
                "numeric_payload_ready": float(numeric_ready),
                "observable_comparison_ready": float(numeric_ready),
                "real_inversion_ready": 0.0,
                "primary_blocker": blocker,
                "target_fetch_stage": stage,
            }
        )
    return rows


def _semantic_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def sota_remote_result_curve_published_semantic_audit_gate(
    *,
    audit_id: str,
    accession_id: str,
    payload_cache: dict,
    physical_observable_labels: Sequence[str],
) -> list[dict[str, float | str]]:
    """Audit published figure curves before treating them as physical observables."""

    if not audit_id:
        raise ValueError("audit_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    observable_tokens = {_semantic_token(label) for label in physical_observable_labels if label}
    if not observable_tokens:
        raise ValueError("physical_observable_labels must contain nonempty labels")

    entries_value = payload_cache.get("entries", [])
    entries = [entry for entry in entries_value if isinstance(entry, dict)] if isinstance(entries_value, list) else []
    rows: list[dict[str, float | str]] = []
    for entry in entries:
        if str(entry.get("curve_role", "")) != "published_figure_curve":
            continue
        path = str(entry.get("path", ""))
        if not path:
            raise ValueError("published figure entries require path")
        header_value = entry.get("header", [])
        header = [str(item) for item in header_value] if isinstance(header_value, list) else []
        header_tokens = [_semantic_token(item) for item in header]
        numeric_row_count = int(entry.get("numeric_row_count", 0) or 0)
        numeric_column_count = int(entry.get("numeric_column_count", 0) or 0)
        header_ready = bool(header)
        time_axis_present = bool(header_tokens) and (
            header_tokens[0] in {"t", "time", "inversetemperature", "1t"} or header[0] == "1/T"
        )
        physical_match = any(token in observable_tokens for token in header_tokens[1:])
        published_ready = header_ready and numeric_row_count > 0 and numeric_column_count > 0
        ml_feature_column_count = max(0, numeric_column_count - (1 if time_axis_present else 0))

        if not published_ready:
            stage = "published_curve_numeric_payload_blocked"
            blocker = "numeric_rows"
        elif physical_match:
            stage = "published_curve_physical_observable_label_uncertainty_missing"
            blocker = "uncertainty"
        else:
            stage = "published_curve_ml_benchmark_not_physical_observable"
            blocker = "physical_observable_label"

        rows.append(
            {
                "audit_id": f"{audit_id}_{str(entry.get('system_id', 'unknown')).lower()}_{Path(path).stem.lower()}",
                "accession_id": accession_id,
                "system_id": str(entry.get("system_id", "unknown")),
                "source_path": path,
                "curve_role": str(entry.get("curve_role", "unknown")),
                "header_tokens": ";".join(header) if header else "none",
                "numeric_row_count": float(numeric_row_count),
                "numeric_column_count": float(numeric_column_count),
                "time_axis_present": float(time_axis_present),
                "header_semantics_ready": float(header_ready),
                "physical_observable_label_match": float(physical_match),
                "ml_feature_column_count": float(ml_feature_column_count),
                "published_curve_ready": float(published_ready),
                "observable_comparison_ready": 0.0,
                "real_inversion_ready": 0.0,
                "primary_blocker": blocker,
                "semantic_stage": stage,
            }
        )
    if not rows:
        raise ValueError("no published figure curves were found")
    return rows


def _numeric_payload_rows(entry: dict) -> list[list[float]]:
    rows_value = entry.get("rows", [])
    if not isinstance(rows_value, list):
        return []
    rows: list[list[float]] = []
    for row_value in rows_value:
        if not isinstance(row_value, list):
            return []
        try:
            rows.append([float(value) for value in row_value])
        except (TypeError, ValueError):
            return []
    return rows


def sota_remote_result_curve_payload_adapter_gate(
    *,
    payload_adapter_id: str,
    accession_id: str,
    manifest: dict,
    payload_cache: dict,
    paired_value_roles_by_system: dict[str, Sequence[str]],
) -> list[dict[str, float | str]]:
    """Pair cached numeric result-curve payloads into structural raw-curve rows."""

    if not payload_adapter_id:
        raise ValueError("payload_adapter_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not paired_value_roles_by_system:
        raise ValueError("paired_value_roles_by_system must be nonempty")

    manifest_entries_value = manifest.get("entries", [])
    payload_entries_value = payload_cache.get("entries", [])
    manifest_entries = (
        [entry for entry in manifest_entries_value if isinstance(entry, dict)]
        if isinstance(manifest_entries_value, list)
        else []
    )
    payload_entries = (
        [entry for entry in payload_entries_value if isinstance(entry, dict)]
        if isinstance(payload_entries_value, list)
        else []
    )
    manifest_by_path = {str(entry.get("path", "")): entry for entry in manifest_entries if entry.get("path")}
    payload_by_path = {str(entry.get("path", "")): entry for entry in payload_entries if entry.get("path")}

    rows: list[dict[str, float | str]] = []
    for system in sorted(paired_value_roles_by_system):
        roles = list(dict.fromkeys(paired_value_roles_by_system[system]))
        if not roles or any(not role for role in roles):
            raise ValueError("paired value roles must be nonempty strings")
        system_payloads = [
            entry for entry in payload_entries if str(entry.get("system_id", "")) == system
        ]
        temperatures = sorted(
            {
                str(entry.get("temperature", ""))
                for entry in system_payloads
                if entry.get("temperature") and str(entry.get("temperature")) != "none"
            },
            key=float,
        )
        for temperature in temperatures:
            time_candidates = [
                entry
                for entry in system_payloads
                if str(entry.get("temperature", "")) == temperature
                and str(entry.get("curve_role", "")) == "time_grid"
            ]
            time_entry = time_candidates[0] if time_candidates else None
            time_rows = _numeric_payload_rows(time_entry) if time_entry is not None else []
            time_grid = [row[0] for row in time_rows if len(row) == 1]
            time_path = str(time_entry.get("path", "")) if time_entry is not None else "none"
            for role in roles:
                value_candidates = [
                    entry
                    for entry in system_payloads
                    if str(entry.get("temperature", "")) == temperature
                    and str(entry.get("curve_role", "")) == role
                ]
                value_entry = value_candidates[0] if value_candidates else None
                value_rows = _numeric_payload_rows(value_entry) if value_entry is not None else []
                value_path = str(value_entry.get("path", "")) if value_entry is not None else "none"
                value_times = [row[0] for row in value_rows if len(row) >= 2]

                paths_to_check = [path for path in [time_path, value_path] if path != "none"]
                checksum_ok = bool(paths_to_check)
                shape_ok = bool(paths_to_check)
                for path in paths_to_check:
                    manifest_entry = manifest_by_path.get(path)
                    payload_entry = payload_by_path.get(path)
                    if manifest_entry is None or payload_entry is None:
                        checksum_ok = False
                        shape_ok = False
                        continue
                    checksum_ok = checksum_ok and str(manifest_entry.get("crc32", "")) == str(payload_entry.get("crc32", ""))
                    checksum_ok = checksum_ok and str(manifest_entry.get("md5", "")) == str(payload_entry.get("md5", ""))
                    shape_ok = shape_ok and int(manifest_entry.get("numeric_row_count", -1) or -1) == int(
                        payload_entry.get("numeric_row_count", -2) or -2
                    )
                    shape_ok = shape_ok and int(manifest_entry.get("numeric_column_count", -1) or -1) == int(
                        payload_entry.get("numeric_column_count", -2) or -2
                    )

                time_ready = bool(time_grid) and len(time_grid) == len(time_rows)
                value_ready = bool(value_rows) and len(value_times) == len(value_rows)
                time_matches = (
                    time_ready
                    and value_ready
                    and len(time_grid) == len(value_times)
                    and all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12) for a, b in zip(time_grid, value_times))
                )
                structural_ready = checksum_ok and shape_ok and time_ready and value_ready and time_matches
                if structural_ready:
                    stage = "range_curve_payload_adapter_ready"
                    blocker = "sigma_rhomax"
                elif time_entry is None:
                    stage = "range_curve_time_grid_missing"
                    blocker = "time_grid"
                elif value_entry is None:
                    stage = "range_curve_value_missing"
                    blocker = role
                elif not checksum_ok:
                    stage = "range_curve_payload_checksum_mismatch"
                    blocker = "checksum"
                elif not shape_ok:
                    stage = "range_curve_payload_shape_mismatch"
                    blocker = "numeric_shape"
                elif not time_ready or not value_ready:
                    stage = "range_curve_payload_parse_blocked"
                    blocker = "numeric_rows"
                else:
                    stage = "range_curve_time_alignment_mismatch"
                    blocker = "time_alignment"

                rows.append(
                    {
                        "payload_adapter_id": f"{payload_adapter_id}_{system.lower()}_{temperature}_{role}",
                        "accession_id": accession_id,
                        "system_id": system,
                        "temperature": temperature,
                        "curve_role": role,
                        "time_grid_path": time_path,
                        "value_curve_path": value_path,
                        "time_point_count": float(len(time_grid)),
                        "value_point_count": float(len(value_rows)),
                        "checksum_matches_manifest": float(checksum_ok),
                        "numeric_shape_matches_manifest": float(shape_ok),
                        "time_grid_matches_value_time": float(time_matches),
                        "available_columns": "temperature;time;rhomax" if structural_ready else "none",
                        "missing_columns": "none" if structural_ready else blocker,
                        "uncertainty_columns": "sigma_rhomax",
                        "missing_uncertainty_columns": "sigma_rhomax",
                        "structural_adapter_ready": float(structural_ready),
                        "uncertainty_adapter_ready": 0.0,
                        "real_inversion_ready": 0.0,
                        "primary_blocker": blocker,
                        "adapter_stage": stage,
                    }
                )
    if not rows:
        raise ValueError("no payload adapter rows were produced")
    return rows


def sota_remote_result_curve_observable_semantics_gate(
    *,
    semantics_id: str,
    accession_id: str,
    payload_adapter_rows: list[dict[str, float | str]],
    role_semantics: dict[str, dict[str, Sequence[str] | str]],
    required_model_semantics: Sequence[str],
) -> list[dict[str, float | str]]:
    """Classify structural result-curve payloads against model-observable semantics."""

    if not semantics_id:
        raise ValueError("semantics_id must be nonempty")
    if not accession_id:
        raise ValueError("accession_id must be nonempty")
    if not payload_adapter_rows:
        raise ValueError("payload_adapter_rows must be nonempty")
    if not role_semantics:
        raise ValueError("role_semantics must be nonempty")
    required = list(dict.fromkeys(required_model_semantics))
    if not required or any(not item for item in required):
        raise ValueError("required_model_semantics must contain nonempty strings")

    rows: list[dict[str, float | str]] = []
    for payload_row in payload_adapter_rows:
        curve_role = str(payload_row.get("curve_role", ""))
        if not curve_role:
            raise ValueError("payload rows must include curve_role")
        role_info = role_semantics.get(curve_role, {})
        candidate_observable = str(role_info.get("candidate_observable", "unmapped_result_curve"))
        available_semantics = list(dict.fromkeys(role_info.get("available_semantics", [])))  # type: ignore[arg-type]
        if any(not item for item in available_semantics):
            raise ValueError("available semantics must be nonempty strings")

        structural_ready = float(payload_row.get("structural_adapter_ready", 0.0)) == 1.0
        uncertainty_ready = float(payload_row.get("uncertainty_adapter_ready", 0.0)) == 1.0
        available_set = set(available_semantics)
        missing_model_semantics = [item for item in required if item not in available_set]
        proxy_ready = structural_ready and candidate_observable != "unmapped_result_curve"
        diagnostic_ready = proxy_ready and uncertainty_ready and not missing_model_semantics

        if not structural_ready:
            stage = "structural_adapter_blocked"
            blocker = str(payload_row.get("primary_blocker", "structural_adapter"))
        elif not proxy_ready:
            stage = "observable_semantics_unmapped"
            blocker = "observable_semantics"
        elif missing_model_semantics:
            stage = "proxy_observable_ready_model_semantics_incomplete"
            blocker = "model_observable_semantics"
        elif not uncertainty_ready:
            stage = "observable_uncertainty_missing"
            blocker = str(payload_row.get("primary_blocker", "uncertainty"))
        else:
            stage = "model_observable_semantics_ready"
            blocker = "none"

        rows.append(
            {
                "semantics_id": f"{semantics_id}_{str(payload_row.get('system_id', '')).lower()}_{payload_row.get('temperature')}_{curve_role}",
                "accession_id": accession_id,
                "payload_adapter_id": str(payload_row.get("payload_adapter_id", "none")),
                "system_id": str(payload_row.get("system_id", "none")),
                "temperature": str(payload_row.get("temperature", "none")),
                "curve_role": curve_role,
                "candidate_observable": candidate_observable,
                "available_semantics": ";".join(available_semantics) if available_semantics else "none",
                "required_model_semantics": ";".join(required),
                "missing_model_semantics": (
                    ";".join(missing_model_semantics) if missing_model_semantics else "none"
                ),
                "structural_adapter_ready": float(structural_ready),
                "proxy_observable_ready": float(proxy_ready),
                "uncertainty_semantics_ready": float(uncertainty_ready),
                "diagnostic_semantics_ready": float(diagnostic_ready),
                "real_inversion_ready": 0.0,
                "primary_blocker": blocker,
                "semantics_stage": stage,
            }
        )
    return rows


def sota_reanalysis_state_gate(
    *,
    state_id: str,
    source_id: str,
    accession_ready: bool,
    readme_digest_ready: bool,
    local_cache_verified: bool,
    zip_structure_ready: bool,
    adapter_ready: bool,
    local_cache_blocker: str,
    zip_structure_blocker: str,
    adapter_blocker: str,
    required_final_protocols: Sequence[str],
) -> dict[str, float | str]:
    """Summarize the current SOTA reanalysis state without overclaiming progress."""

    for name, value in {
        "state_id": state_id,
        "source_id": source_id,
        "local_cache_blocker": local_cache_blocker,
        "zip_structure_blocker": zip_structure_blocker,
        "adapter_blocker": adapter_blocker,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    if not required_final_protocols:
        raise ValueError("required_final_protocols must be nonempty")
    if any(not protocol for protocol in required_final_protocols):
        raise ValueError("required_final_protocols must contain nonempty strings")

    protocols = list(dict.fromkeys(required_final_protocols))
    if not accession_ready:
        stage = "awaiting_public_accession"
        claim_level = "citation_only"
        blocker = "accession"
        next_action = "obtain_public_machine_readable_accession"
    elif not readme_digest_ready:
        stage = "awaiting_readme_digest"
        claim_level = "remote_metadata_only"
        blocker = "readme_digest"
        next_action = "cache_and_verify_lightweight_readme"
    elif not local_cache_verified:
        stage = "awaiting_full_archive_cache"
        claim_level = "metadata_verified_not_reanalysis"
        blocker = local_cache_blocker
        next_action = "cache_full_archive_and_verify_checksum"
    elif not zip_structure_ready:
        stage = "awaiting_zip_structure"
        claim_level = "local_archive_checksum_verified"
        blocker = zip_structure_blocker
        next_action = "inspect_zip_central_directory_roots"
    elif not adapter_ready:
        stage = "awaiting_adapter_mapping"
        claim_level = "archive_structure_verified"
        blocker = adapter_blocker
        next_action = "map_trajectory_columns_and_metadata"
    else:
        stage = "ready_for_trajectory_observable_protocol"
        claim_level = "local_archive_adapter_ready"
        blocker = "none"
        next_action = "run_trajectory_observable_protocol"

    ready_for_reanalysis = adapter_ready and zip_structure_ready and local_cache_verified
    ready_for_model_comparison = ready_for_reanalysis and len(protocols) >= 2
    return {
        "state_id": state_id,
        "source_id": source_id,
        "accession_ready": float(accession_ready),
        "readme_digest_ready": float(readme_digest_ready),
        "local_cache_verified": float(local_cache_verified),
        "zip_structure_ready": float(zip_structure_ready),
        "adapter_ready": float(adapter_ready),
        "required_final_protocols": ";".join(protocols),
        "required_final_protocol_count": float(len(protocols)),
        "ready_for_trajectory_reanalysis": float(ready_for_reanalysis),
        "ready_for_model_comparison": float(ready_for_model_comparison),
        "claim_level": claim_level,
        "primary_blocker": blocker,
        "next_action": next_action,
        "reanalysis_stage": stage,
    }


def sota_readme_schema_gate(
    *,
    schema_id: str,
    accession_id: str,
    source_id: str,
    systems: Sequence[str],
    folder_tokens: Sequence[str],
    license_statement: str,
    required_citations: Sequence[str],
    intended_protocols: Sequence[str],
    local_archive_inspected: bool,
) -> dict[str, float | str]:
    """Check README-level schema evidence before building a local data adapter."""

    for name, value in {
        "schema_id": schema_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "license_statement": license_statement,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    if not systems:
        raise ValueError("systems must be nonempty")
    if not folder_tokens:
        raise ValueError("folder_tokens must be nonempty")
    if not intended_protocols:
        raise ValueError("intended_protocols must be nonempty")
    for name, values in {
        "systems": systems,
        "folder_tokens": folder_tokens,
        "required_citations": required_citations,
        "intended_protocols": intended_protocols,
    }.items():
        if any(not value for value in values):
            raise ValueError(f"{name} must contain nonempty strings")

    system_set = set(systems)
    folder_set = set(folder_tokens)
    protocols = list(dict.fromkeys(intended_protocols))
    has_ka = "KA" in system_set
    has_ka2d = "KA2D" in system_set
    has_trajectory_folder = "_trajectories" in folder_set
    has_model_folder = "_models" in folder_set
    has_results_folder = "_results" in folder_set
    license_lower = license_statement.lower()
    reuse_license = "creative commons attribution" in license_lower or "cc-by" in license_lower
    trajectory_protocol_requested = any(protocol.startswith("trajectory_") for protocol in protocols)
    has_citation_guidance = bool(required_citations)
    schema_ready = (
        trajectory_protocol_requested
        and has_ka
        and has_ka2d
        and has_trajectory_folder
        and has_model_folder
        and has_results_folder
        and reuse_license
        and has_citation_guidance
    )
    ready_for_local_adapter = schema_ready and local_archive_inspected

    if ready_for_local_adapter:
        stage = "local_archive_schema_ready"
        blocker = "none"
    elif schema_ready:
        stage = "remote_readme_schema_ready"
        blocker = "local_archive_inspection"
    elif not has_trajectory_folder:
        stage = "metadata_incomplete_schema"
        blocker = "trajectory_folder"
    elif not has_model_folder:
        stage = "metadata_incomplete_schema"
        blocker = "model_folder"
    elif not has_results_folder:
        stage = "metadata_incomplete_schema"
        blocker = "results_folder"
    elif not (has_ka and has_ka2d):
        stage = "metadata_incomplete_schema"
        blocker = "systems"
    elif not reuse_license:
        stage = "metadata_incomplete_schema"
        blocker = "reuse_license"
    elif not has_citation_guidance:
        stage = "metadata_incomplete_schema"
        blocker = "citation_guidance"
    else:
        stage = "metadata_incomplete_schema"
        blocker = "trajectory_protocol"

    return {
        "schema_id": schema_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "systems": ";".join(dict.fromkeys(systems)),
        "folder_tokens": ";".join(dict.fromkeys(folder_tokens)),
        "license_statement": license_statement,
        "required_citations": ";".join(dict.fromkeys(required_citations)) if required_citations else "none",
        "intended_protocols": ";".join(protocols),
        "has_ka_system": float(has_ka),
        "has_ka2d_system": float(has_ka2d),
        "has_trajectory_folder": float(has_trajectory_folder),
        "has_model_folder": float(has_model_folder),
        "has_results_folder": float(has_results_folder),
        "reuse_license": float(reuse_license),
        "citation_count": float(len(dict.fromkeys(required_citations))),
        "local_archive_inspected": float(local_archive_inspected),
        "schema_ready": float(schema_ready),
        "ready_for_local_adapter": float(ready_for_local_adapter),
        "primary_blocker": blocker,
        "schema_stage": stage,
    }


def trajectory_adapter_contract(
    *,
    contract_id: str,
    accession_id: str,
    source_id: str,
    system_id: str,
    expected_archive_roots: Sequence[str],
    required_local_fields: Sequence[str],
    available_local_fields: Sequence[str],
    intended_protocols: Sequence[str],
    local_archive_inspected: bool,
) -> dict[str, float | str]:
    """Gate local trajectory adapter readiness after remote README/schema checks."""

    for name, value in {
        "contract_id": contract_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "system_id": system_id,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    for name, values in {
        "expected_archive_roots": expected_archive_roots,
        "required_local_fields": required_local_fields,
        "available_local_fields": available_local_fields,
        "intended_protocols": intended_protocols,
    }.items():
        if not values:
            raise ValueError(f"{name} must be nonempty")
        if any(not value for value in values):
            raise ValueError(f"{name} must contain nonempty strings")

    roots = list(dict.fromkeys(expected_archive_roots))
    required = list(dict.fromkeys(required_local_fields))
    available = list(dict.fromkeys(available_local_fields))
    protocols = list(dict.fromkeys(intended_protocols))
    available_set = set(available)
    missing = [field for field in required if field not in available_set]
    available_required_count = sum(1 for field in required if field in available_set)
    trajectory_protocol_requested = any(protocol.startswith("trajectory_") for protocol in protocols)
    adapter_ready = bool(local_archive_inspected and not missing and trajectory_protocol_requested)

    if adapter_ready:
        stage = "local_trajectory_adapter_ready"
        blocker = "none"
    elif missing:
        stage = "remote_adapter_contract_only" if not local_archive_inspected else "metadata_incomplete_adapter"
        blocker = missing[0]
    elif not local_archive_inspected:
        stage = "remote_adapter_contract_only"
        blocker = "local_archive_inspection"
    else:
        stage = "metadata_incomplete_adapter"
        blocker = "trajectory_protocol"

    return {
        "contract_id": contract_id,
        "accession_id": accession_id,
        "source_id": source_id,
        "system_id": system_id,
        "expected_archive_roots": ";".join(roots),
        "required_local_fields": ";".join(required),
        "available_local_fields": ";".join(available),
        "missing_local_fields": ";".join(missing) if missing else "none",
        "required_field_count": float(len(required)),
        "available_required_field_count": float(available_required_count),
        "local_archive_inspected": float(local_archive_inspected),
        "trajectory_protocol_requested": float(trajectory_protocol_requested),
        "adapter_ready": float(adapter_ready),
        "intended_protocols": ";".join(protocols),
        "primary_blocker": blocker,
        "adapter_stage": stage,
    }


def sota_claim_alignment(
    *,
    claim_id: str,
    source_key: str,
    phenomenon: str,
    claim_type: str,
    observed_claim: str,
    model_diagnostic: str,
    model_support_level: str,
    data_readiness: str,
    primary_blocker: str,
) -> dict[str, float | str]:
    """Map a source-level glass claim to the model's diagnostic status."""

    for name, value in {
        "claim_id": claim_id,
        "source_key": source_key,
        "phenomenon": phenomenon,
        "claim_type": claim_type,
        "observed_claim": observed_claim,
        "model_diagnostic": model_diagnostic,
        "model_support_level": model_support_level,
        "data_readiness": data_readiness,
        "primary_blocker": primary_blocker,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")

    allowed_claim_types = {
        "dynamical_signature",
        "transport_decoupling",
        "spatial_heterogeneity",
        "thermodynamic_transition",
    }
    allowed_support = {"derived", "effective_closure", "closure_only", "not_supported"}
    allowed_readiness = {"qualitative", "structural_raw", "uncertainty_weighted"}
    if claim_type not in allowed_claim_types:
        raise ValueError("claim_type is not recognized")
    if model_support_level not in allowed_support:
        raise ValueError("model_support_level is not recognized")
    if data_readiness not in allowed_readiness:
        raise ValueError("data_readiness is not recognized")
    if claim_type == "thermodynamic_transition" and model_support_level == "derived":
        raise ValueError("renewal dynamics cannot be marked as deriving a thermodynamic transition")

    requires_closure = model_support_level in {"effective_closure", "closure_only"}
    if model_support_level == "derived":
        alignment = "supported"
    elif model_support_level in {"effective_closure", "closure_only"}:
        alignment = "scope_boundary" if claim_type == "thermodynamic_transition" else "partial"
    else:
        alignment = "not_supported"

    quantitative_fit_ready = data_readiness == "uncertainty_weighted" and primary_blocker == "none"
    overclaims = (
        claim_type == "thermodynamic_transition" and model_support_level != "closure_only"
    ) or (
        claim_type == "spatial_heterogeneity" and model_support_level == "derived"
    )

    return {
        "claim_id": claim_id,
        "source_key": source_key,
        "phenomenon": phenomenon,
        "claim_type": claim_type,
        "observed_claim": observed_claim,
        "model_diagnostic": model_diagnostic,
        "model_support_level": model_support_level,
        "claim_alignment": alignment,
        "data_readiness": data_readiness,
        "primary_blocker": primary_blocker,
        "requires_external_closure": float(requires_closure),
        "quantitative_fit_ready": float(quantitative_fit_ready),
        "model_overclaims_source": float(overclaims),
    }


def sota_evidence_verdict(
    *,
    verdict_id: str,
    source_key: str,
    phenomenon: str,
    claim_alignment: str,
    signed_constraint_class: str,
    data_readiness: str,
    requires_external_closure: bool,
    quantitative_fit_ready: bool,
    model_overclaims_source: bool,
    reanalysis_stage: str,
) -> dict[str, float | str]:
    """Assign a manuscript-safe evidence grade to a SOTA comparison row."""

    for name, value in {
        "verdict_id": verdict_id,
        "source_key": source_key,
        "phenomenon": phenomenon,
        "claim_alignment": claim_alignment,
        "signed_constraint_class": signed_constraint_class,
        "data_readiness": data_readiness,
        "reanalysis_stage": reanalysis_stage,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    allowed_alignments = {"supported", "partial", "scope_boundary", "not_supported"}
    allowed_constraint_classes = {
        "sota_consistent",
        "closure_assisted_consistent",
        "scope_boundary_consistent",
        "missing_signature",
        "not_supported",
        "overclaimed_boundary",
    }
    if claim_alignment not in allowed_alignments:
        raise ValueError("claim_alignment is not recognized")
    if signed_constraint_class not in allowed_constraint_classes:
        raise ValueError("signed_constraint_class is not recognized")

    trajectory_pending = reanalysis_stage.startswith("awaiting_") or "not_reanalysis" in data_readiness
    if model_overclaims_source or signed_constraint_class == "overclaimed_boundary":
        evidence_grade = "overclaimed_or_forbidden"
        allowed_claim = "do_not_claim"
        publishable = False
    elif trajectory_pending:
        evidence_grade = "pending_trajectory_reanalysis"
        allowed_claim = "pending_reanalysis_only"
        publishable = False
    elif claim_alignment == "scope_boundary" or signed_constraint_class == "scope_boundary_consistent":
        evidence_grade = "thermodynamic_scope_boundary"
        allowed_claim = "scope_boundary_only"
        publishable = True
    elif claim_alignment == "supported" and signed_constraint_class == "sota_consistent":
        evidence_grade = "direct_dynamical_support"
        allowed_claim = "dynamical_signature_supported"
        publishable = True
    elif claim_alignment == "partial" or signed_constraint_class == "closure_assisted_consistent" or requires_external_closure:
        evidence_grade = "closure_assisted_support"
        allowed_claim = "closure_assisted_dynamical_trend"
        publishable = True
    else:
        evidence_grade = "not_supported"
        allowed_claim = "do_not_claim"
        publishable = False

    if quantitative_fit_ready:
        evidence_strength = "uncertainty_weighted_quantitative"
    elif data_readiness in {"structural_raw", "metadata_verified_not_reanalysis"}:
        evidence_strength = "structural_or_metadata"
    else:
        evidence_strength = "qualitative_or_protocol"

    return {
        "verdict_id": verdict_id,
        "source_key": source_key,
        "phenomenon": phenomenon,
        "claim_alignment": claim_alignment,
        "signed_constraint_class": signed_constraint_class,
        "data_readiness": data_readiness,
        "requires_external_closure": float(requires_external_closure),
        "quantitative_fit_ready": float(quantitative_fit_ready),
        "model_overclaims_source": float(model_overclaims_source),
        "reanalysis_stage": reanalysis_stage,
        "trajectory_reanalysis_required": float(trajectory_pending),
        "evidence_strength": evidence_strength,
        "evidence_grade": evidence_grade,
        "allowed_manuscript_claim": allowed_claim,
        "publishable_without_overclaim": float(publishable),
    }


def sota_evidence_class_gate(
    *,
    class_id: str,
    source_key: str,
    source_modality: str,
    evidence_grade: str,
    observed_signatures: Sequence[str],
    model_supported_signatures: Sequence[str],
    available_quantitative_inputs: Sequence[str],
    required_quantitative_inputs: Sequence[str],
    requires_external_closure: bool,
    has_machine_readable_curves: bool,
    has_uncertainties: bool,
    has_shared_ensemble: bool,
) -> dict[str, float | str]:
    """Classify SOTA evidence before promoting trends to quantitative fits."""

    for name, value in {
        "class_id": class_id,
        "source_key": source_key,
        "source_modality": source_modality,
        "evidence_grade": evidence_grade,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    allowed_modalities = {
        "simulation",
        "experiment",
        "mixed_simulation_experiment",
        "theory",
        "metadata_repository",
        "synthetic_canary",
    }
    allowed_evidence = {
        "direct_dynamical_support",
        "closure_assisted_support",
        "thermodynamic_scope_boundary",
        "pending_trajectory_reanalysis",
        "overclaimed_or_forbidden",
        "not_supported",
    }
    if source_modality not in allowed_modalities:
        raise ValueError("source_modality is not recognized")
    if evidence_grade not in allowed_evidence:
        raise ValueError("evidence_grade is not recognized")
    for name, values in {
        "observed_signatures": observed_signatures,
        "model_supported_signatures": model_supported_signatures,
        "available_quantitative_inputs": available_quantitative_inputs,
        "required_quantitative_inputs": required_quantitative_inputs,
    }.items():
        if not values:
            raise ValueError(f"{name} must be nonempty")
        if any(not value for value in values):
            raise ValueError(f"{name} must contain nonempty strings")

    observed = list(dict.fromkeys(str(value) for value in observed_signatures))
    supported = list(dict.fromkeys(str(value) for value in model_supported_signatures))
    available_inputs = list(dict.fromkeys(str(value) for value in available_quantitative_inputs))
    required_inputs = list(dict.fromkeys(str(value) for value in required_quantitative_inputs))
    supported_set = set(supported)
    available_set = set(available_inputs)
    missing_supported = [signature for signature in observed if signature not in supported_set]
    missing_inputs = [input_name for input_name in required_inputs if input_name not in available_set]

    all_inputs_ready = (
        not missing_inputs
        and bool(has_machine_readable_curves)
        and bool(has_uncertainties)
        and bool(has_shared_ensemble)
    )
    trend_allowed = evidence_grade not in {"overclaimed_or_forbidden", "not_supported"}
    if evidence_grade == "thermodynamic_scope_boundary" or source_modality == "theory":
        evidence_class = "thermodynamic_scope_boundary"
        quantitative_allowed = False
        blocker = "renewal_dynamics_not_thermodynamic_theory"
    elif evidence_grade == "pending_trajectory_reanalysis" or source_modality == "metadata_repository":
        evidence_class = "metadata_reanalysis_candidate"
        quantitative_allowed = False
        blocker = missing_inputs[0] if missing_inputs else "completed_reanalysis_gate"
    elif all_inputs_ready and not requires_external_closure and evidence_grade == "direct_dynamical_support":
        evidence_class = "uncertainty_weighted_quantitative_test"
        quantitative_allowed = True
        blocker = "none"
    elif source_modality in {"experiment", "mixed_simulation_experiment"} and requires_external_closure:
        evidence_class = "closure_assisted_experimental_constraint"
        quantitative_allowed = False
        blocker = missing_inputs[0] if missing_inputs else "external_closure"
    elif source_modality in {"experiment", "mixed_simulation_experiment"}:
        evidence_class = "qualitative_experimental_trend"
        quantitative_allowed = False
        blocker = missing_inputs[0] if missing_inputs else "uncertainty_weighted_protocol"
    elif evidence_grade == "direct_dynamical_support":
        evidence_class = "structural_simulation_support"
        quantitative_allowed = False
        blocker = missing_inputs[0] if missing_inputs else "uncertainty_weighted_protocol"
    elif evidence_grade == "closure_assisted_support":
        evidence_class = "closure_assisted_simulation_constraint"
        quantitative_allowed = False
        blocker = missing_inputs[0] if missing_inputs else "external_closure"
    else:
        evidence_class = "not_supported"
        quantitative_allowed = False
        blocker = "source_support"

    return {
        "class_id": class_id,
        "source_key": source_key,
        "source_modality": source_modality,
        "evidence_grade": evidence_grade,
        "observed_signatures": ";".join(observed),
        "model_supported_signatures": ";".join(supported),
        "missing_model_supported_signatures": ";".join(missing_supported) if missing_supported else "none",
        "available_quantitative_inputs": ";".join(available_inputs),
        "required_quantitative_inputs": ";".join(required_inputs),
        "missing_quantitative_inputs": ";".join(missing_inputs) if missing_inputs else "none",
        "has_machine_readable_curves": float(has_machine_readable_curves),
        "has_uncertainties": float(has_uncertainties),
        "has_shared_ensemble": float(has_shared_ensemble),
        "requires_external_closure": float(requires_external_closure),
        "trend_comparison_allowed": float(trend_allowed),
        "quantitative_inversion_allowed": float(quantitative_allowed),
        "evidence_class": evidence_class,
        "primary_blocker": blocker,
    }


def sota_signed_constraint_audit(
    *,
    constraint_id: str,
    source_key: str,
    model_scope: str,
    source_observation: str,
    expected_signatures: Sequence[str],
    passed_signatures: Sequence[str],
    forbidden_claims: Sequence[str],
    made_claims: Sequence[str],
    support_level: str,
    quantitative_fit_ready: bool,
) -> dict[str, float | str]:
    """Check source-level SOTA conclusions as signed constraints on the model."""

    for name, value in {
        "constraint_id": constraint_id,
        "source_key": source_key,
        "model_scope": model_scope,
        "source_observation": source_observation,
        "support_level": support_level,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    allowed_scopes = {
        "dynamical_signature",
        "transport_decoupling",
        "spatial_heterogeneity",
        "thermodynamic_transition",
    }
    allowed_support = {"derived", "effective_closure", "closure_only", "not_supported"}
    if model_scope not in allowed_scopes:
        raise ValueError("model_scope is not recognized")
    if support_level not in allowed_support:
        raise ValueError("support_level is not recognized")
    if not expected_signatures:
        raise ValueError("expected_signatures must be nonempty")
    for name, values in {
        "expected_signatures": expected_signatures,
        "passed_signatures": passed_signatures,
        "forbidden_claims": forbidden_claims,
        "made_claims": made_claims,
    }.items():
        if any(not value for value in values):
            raise ValueError(f"{name} must contain nonempty strings")
    if model_scope == "thermodynamic_transition" and support_level == "derived":
        raise ValueError("renewal dynamics cannot derive thermodynamic transition constraints")
    if model_scope == "spatial_heterogeneity" and support_level == "derived":
        raise ValueError("spatial heterogeneity must be marked as an effective closure")

    expected = list(dict.fromkeys(expected_signatures))
    passed = list(dict.fromkeys(passed_signatures))
    forbidden = list(dict.fromkeys(forbidden_claims))
    made = list(dict.fromkeys(made_claims))
    passed_set = set(passed)
    made_set = set(made)
    missing = [signature for signature in expected if signature not in passed_set]
    forbidden_made = [claim for claim in forbidden if claim in made_set]
    requires_closure = support_level in {"effective_closure", "closure_only"}
    all_required_pass = not missing
    no_forbidden_claims = not forbidden_made

    if forbidden_made:
        constraint_class = "overclaimed_boundary"
    elif missing:
        constraint_class = "missing_signature"
    elif support_level == "not_supported":
        constraint_class = "not_supported"
    elif model_scope == "thermodynamic_transition":
        constraint_class = "scope_boundary_consistent"
    elif requires_closure:
        constraint_class = "closure_assisted_consistent"
    else:
        constraint_class = "sota_consistent"

    publishable_alignment = (
        constraint_class
        in {
            "sota_consistent",
            "closure_assisted_consistent",
            "scope_boundary_consistent",
        }
    )

    return {
        "constraint_id": constraint_id,
        "source_key": source_key,
        "model_scope": model_scope,
        "source_observation": source_observation,
        "expected_signatures": ";".join(expected),
        "passed_signatures": ";".join(passed) if passed else "none",
        "missing_expected_signatures": ";".join(missing) if missing else "none",
        "forbidden_claims": ";".join(forbidden) if forbidden else "none",
        "made_claims": ";".join(made) if made else "none",
        "forbidden_claims_made": ";".join(forbidden_made) if forbidden_made else "none",
        "support_level": support_level,
        "requires_external_closure": float(requires_closure),
        "all_required_signatures_pass": float(all_required_pass),
        "no_forbidden_claims_made": float(no_forbidden_claims),
        "quantitative_fit_ready": float(quantitative_fit_ready),
        "publishable_alignment": float(publishable_alignment),
        "signed_constraint_class": constraint_class,
    }


def observable_falsification_matrix(
    *,
    benchmark_id: str,
    benchmark_source: str,
    available_observables: list[str],
    diagnostic_requirements: dict[str, list[str]],
    has_machine_readable_data: bool,
    has_uncertainty_estimates: bool,
) -> list[dict[str, float | str]]:
    """Map literature observables to the diagnostic tests they can falsify."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    if not benchmark_source:
        raise ValueError("benchmark_source must be nonempty")
    if not diagnostic_requirements:
        raise ValueError("diagnostic_requirements must be nonempty")
    if any(not item for item in available_observables):
        raise ValueError("available_observables must be nonempty strings")

    available = list(dict.fromkeys(available_observables))
    available_set = set(available)
    rows: list[dict[str, float | str]] = []
    for diagnostic_id, required_observables in diagnostic_requirements.items():
        if not diagnostic_id:
            raise ValueError("diagnostic ids must be nonempty")
        if not required_observables:
            raise ValueError("diagnostic requirements must be nonempty")
        required = list(dict.fromkeys(required_observables))
        if any(not item for item in required):
            raise ValueError("diagnostic requirements must be nonempty strings")
        missing = [item for item in required if item not in available_set]
        coverage = (len(required) - len(missing)) / len(required)
        structural_ready = len(missing) == 0
        quantitative_ready = structural_ready and has_machine_readable_data and has_uncertainty_estimates
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "benchmark_source": benchmark_source,
                "diagnostic_id": diagnostic_id,
                "required_observables": ";".join(required),
                "available_observables": ";".join(available) if available else "none",
                "missing_observables": ";".join(missing) if missing else "none",
                "primary_blocker": missing[0] if missing else "none",
                "observable_coverage_fraction": coverage,
                "structural_falsification_ready": float(structural_ready),
                "has_machine_readable_data": float(has_machine_readable_data),
                "has_uncertainty_estimates": float(has_uncertainty_estimates),
                "quantitative_falsification_ready": float(quantitative_ready),
            }
        )
    return rows


def benchmark_fusion_readiness(
    *,
    fusion_id: str,
    benchmark_sources: list[str],
    required_observables: list[str],
    available_observables_by_benchmark: dict[str, list[str]],
    system_tags: dict[str, str],
    temperature_grid_tags: dict[str, str],
    ensemble_tags: dict[str, str],
    has_machine_readable_data: bool,
    has_uncertainty_estimates: bool,
) -> dict[str, float | str]:
    """Check whether multiple benchmark sources can be fused for one diagnostic."""

    if not fusion_id:
        raise ValueError("fusion_id must be nonempty")
    if not benchmark_sources:
        raise ValueError("benchmark_sources must be nonempty")
    if not required_observables:
        raise ValueError("required_observables must be nonempty")
    if any(not source for source in benchmark_sources):
        raise ValueError("benchmark_sources must be nonempty strings")
    if any(not observable for observable in required_observables):
        raise ValueError("required_observables must be nonempty strings")

    sources = list(dict.fromkeys(benchmark_sources))
    required = list(dict.fromkeys(required_observables))
    available: list[str] = []
    for source in sources:
        if source not in available_observables_by_benchmark:
            raise ValueError(f"missing observables for {source}")
        if source not in system_tags or source not in temperature_grid_tags or source not in ensemble_tags:
            raise ValueError(f"missing compatibility tags for {source}")
        source_observables = available_observables_by_benchmark[source]
        if any(not observable for observable in source_observables):
            raise ValueError("available observables must be nonempty strings")
        available.extend(source_observables)

    available_unique = list(dict.fromkeys(available))
    available_set = set(available_unique)
    missing = [observable for observable in required if observable not in available_set]
    coverage = (len(required) - len(missing)) / len(required)
    shared_system = len({system_tags[source] for source in sources}) == 1
    shared_temperature_grid = len({temperature_grid_tags[source] for source in sources}) == 1
    shared_ensemble = len({ensemble_tags[source] for source in sources}) == 1
    structural_ready = (
        len(missing) == 0
        and shared_system
        and shared_temperature_grid
        and shared_ensemble
    )
    quantitative_ready = structural_ready and has_machine_readable_data and has_uncertainty_estimates
    if missing:
        blocker = missing[0]
    elif not shared_system:
        blocker = "system_mismatch"
    elif not shared_temperature_grid:
        blocker = "temperature_grid_mismatch"
    elif not shared_ensemble:
        blocker = "ensemble_mismatch"
    elif not has_machine_readable_data:
        blocker = "machine_readable_data"
    elif not has_uncertainty_estimates:
        blocker = "uncertainty_estimates"
    else:
        blocker = "none"

    return {
        "fusion_id": fusion_id,
        "benchmark_sources": ";".join(sources),
        "required_observables": ";".join(required),
        "available_observables": ";".join(available_unique) if available_unique else "none",
        "missing_observables": ";".join(missing) if missing else "none",
        "observable_coverage_fraction": coverage,
        "system_tags": ";".join(system_tags[source] for source in sources),
        "temperature_grid_tags": ";".join(temperature_grid_tags[source] for source in sources),
        "ensemble_tags": ";".join(ensemble_tags[source] for source in sources),
        "shared_system_consistent": float(shared_system),
        "shared_temperature_grid_consistent": float(shared_temperature_grid),
        "shared_ensemble_consistent": float(shared_ensemble),
        "has_machine_readable_data": float(has_machine_readable_data),
        "has_uncertainty_estimates": float(has_uncertainty_estimates),
        "structural_fusion_ready": float(structural_ready),
        "quantitative_fusion_ready": float(quantitative_ready),
        "primary_blocker": blocker,
    }


def raw_curve_ingestion_contract(
    *,
    benchmark_id: str,
    observable_requirements: dict[str, dict[str, list[str] | str]],
    available_columns_by_observable: dict[str, list[str]],
    machine_readable: bool,
    shared_temperature_grid: bool,
    shared_time_units: bool,
) -> list[dict[str, float | str]]:
    """Check whether raw benchmark curves satisfy the ingestion contract."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    if not observable_requirements:
        raise ValueError("observable_requirements must be nonempty")

    rows: list[dict[str, float | str]] = []
    for observable_id, requirement in observable_requirements.items():
        if not observable_id:
            raise ValueError("observable ids must be nonempty")
        if "required_columns" not in requirement:
            raise ValueError(f"missing required_columns for {observable_id}")
        if "uncertainty_columns" not in requirement:
            raise ValueError(f"missing uncertainty_columns for {observable_id}")
        if "target_diagnostic" not in requirement:
            raise ValueError(f"missing target_diagnostic for {observable_id}")
        required_columns = list(dict.fromkeys(requirement["required_columns"]))  # type: ignore[arg-type]
        uncertainty_columns = list(dict.fromkeys(requirement["uncertainty_columns"]))  # type: ignore[arg-type]
        target_diagnostic = str(requirement["target_diagnostic"])
        if not required_columns:
            raise ValueError("required_columns must be nonempty")
        if any(not column for column in required_columns + uncertainty_columns):
            raise ValueError("column names must be nonempty")
        if not target_diagnostic:
            raise ValueError("target_diagnostic must be nonempty")

        available_columns = list(
            dict.fromkeys(available_columns_by_observable.get(observable_id, []))
        )
        if any(not column for column in available_columns):
            raise ValueError("available columns must be nonempty strings")
        available_set = set(available_columns)
        missing_columns = [column for column in required_columns if column not in available_set]
        missing_uncertainty_columns = [
            column for column in uncertainty_columns if column not in available_set
        ]
        structural_ready = (
            machine_readable
            and shared_temperature_grid
            and shared_time_units
            and len(missing_columns) == 0
        )
        uncertainty_ready = structural_ready and len(missing_uncertainty_columns) == 0
        if not machine_readable:
            blocker = "machine_readable_data"
        elif not shared_temperature_grid:
            blocker = "temperature_grid_mismatch"
        elif not shared_time_units:
            blocker = "time_unit_mismatch"
        elif missing_columns:
            blocker = missing_columns[0]
        elif missing_uncertainty_columns:
            blocker = missing_uncertainty_columns[0]
        else:
            blocker = "none"

        rows.append(
            {
                "benchmark_id": benchmark_id,
                "observable_id": observable_id,
                "target_diagnostic": target_diagnostic,
                "required_columns": ";".join(required_columns),
                "available_columns": ";".join(available_columns) if available_columns else "none",
                "missing_columns": ";".join(missing_columns) if missing_columns else "none",
                "uncertainty_columns": ";".join(uncertainty_columns) if uncertainty_columns else "none",
                "missing_uncertainty_columns": (
                    ";".join(missing_uncertainty_columns)
                    if missing_uncertainty_columns
                    else "none"
                ),
                "machine_readable": float(machine_readable),
                "shared_temperature_grid": float(shared_temperature_grid),
                "shared_time_units": float(shared_time_units),
                "structural_ingestion_ready": float(structural_ready),
                "uncertainty_ingestion_ready": float(uncertainty_ready),
                "primary_blocker": blocker,
            }
        )
    return rows


def raw_curve_diagnostic_readiness(
    *,
    benchmark_id: str,
    contract_rows: list[dict[str, float | str]],
    diagnostic_observables: dict[str, list[str]],
) -> list[dict[str, float | str]]:
    """Aggregate raw-curve ingestion rows into diagnostic-level readiness."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    if not contract_rows:
        raise ValueError("contract_rows must be nonempty")
    if not diagnostic_observables:
        raise ValueError("diagnostic_observables must be nonempty")

    by_observable = {str(row["observable_id"]): row for row in contract_rows}
    rows: list[dict[str, float | str]] = []
    for diagnostic_id, observables in diagnostic_observables.items():
        if not diagnostic_id:
            raise ValueError("diagnostic ids must be nonempty")
        if not observables:
            raise ValueError("diagnostic observable lists must be nonempty")
        if any(not observable for observable in observables):
            raise ValueError("diagnostic observables must be nonempty strings")

        unique_observables = list(dict.fromkeys(observables))
        missing_observables = [
            observable for observable in unique_observables if observable not in by_observable
        ]
        required_rows = [
            by_observable[observable]
            for observable in unique_observables
            if observable in by_observable
        ]
        structural_ready = (
            len(missing_observables) == 0
            and all(float(row["structural_ingestion_ready"]) == 1.0 for row in required_rows)
        )
        uncertainty_ready = (
            structural_ready
            and all(float(row["uncertainty_ingestion_ready"]) == 1.0 for row in required_rows)
        )
        blockers: list[str] = []
        blockers.extend(missing_observables)
        for row in required_rows:
            blocker = str(row["primary_blocker"])
            if blocker != "none" and blocker not in blockers:
                blockers.append(blocker)
        primary_blocker = blockers[0] if blockers else "none"
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "diagnostic_id": diagnostic_id,
                "required_observables": ";".join(unique_observables),
                "missing_observables": (
                    ";".join(missing_observables) if missing_observables else "none"
                ),
                "structural_diagnostic_ready": float(structural_ready),
                "uncertainty_diagnostic_ready": float(uncertainty_ready),
                "blocking_observables_or_columns": ";".join(blockers) if blockers else "none",
                "primary_blocker": primary_blocker,
            }
        )
    return rows


def _as_strictly_increasing_curve(
    curve: tuple[np.ndarray, np.ndarray],
    *,
    name: str,
    values_must_be_positive: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    time, value = curve
    time = np.asarray(time, dtype=float)
    value = np.asarray(value, dtype=float)
    if time.ndim != 1 or value.ndim != 1 or time.size != value.size or time.size < 2:
        raise ValueError(f"{name} must contain matching one-dimensional time and value arrays")
    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(value)):
        raise ValueError(f"{name} must contain finite values")
    if np.any(time <= 0.0) or np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{name} time grid must be positive and strictly increasing")
    if values_must_be_positive and np.any(value <= 0.0):
        raise ValueError(f"{name} values must be positive")
    return time, value


def _first_log_threshold_crossing_time(
    time: np.ndarray,
    value: np.ndarray,
    *,
    threshold: float,
    name: str,
) -> float:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    if value[0] <= threshold:
        raise ValueError(f"{name} starts below the alpha threshold")
    crossing_indices = np.flatnonzero(value <= threshold)
    if crossing_indices.size == 0:
        raise ValueError(f"{name} does not cross the alpha threshold")
    index = int(crossing_indices[0])
    if index == 0:
        return float(time[0])
    t0 = float(time[index - 1])
    t1 = float(time[index])
    y0 = float(value[index - 1])
    y1 = float(value[index])
    if y0 <= 0.0 or y1 <= 0.0:
        raise ValueError(f"{name} values must be positive near the crossing")
    if math.isclose(y0, y1, rel_tol=1e-14, abs_tol=1e-14):
        return t1
    fraction = (math.log(threshold) - math.log(y0)) / (math.log(y1) - math.log(y0))
    return t0 + min(1.0, max(0.0, fraction)) * (t1 - t0)


def raw_curve_persistence_exchange_protocol(
    *,
    benchmark_id: str,
    anchor_wave_number: float,
    alpha_curves_by_k: dict[float, tuple[np.ndarray, np.ndarray]],
    jump_variance: float,
    diffusion_coefficient: float,
    late_time: float,
    ngp_curve: tuple[np.ndarray, np.ndarray],
    chi4_curve: tuple[np.ndarray, np.ndarray],
    tau_alpha_relative_error_by_k: dict[float, float],
    late_ngp_relative_error: float,
    chi4_peak_relative_error: float,
    cage_variance: float = 1.0,
    cage_tau: float = 0.2,
    threshold: float = math.exp(-1.0),
    z_threshold: float = 2.0,
) -> dict[str, float | str]:
    """Extract raw-curve observables and run the persistence/exchange protocol."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    if anchor_wave_number <= 0.0:
        raise ValueError("anchor_wave_number must be positive")
    if not alpha_curves_by_k:
        raise ValueError("alpha_curves_by_k must be nonempty")
    if anchor_wave_number not in alpha_curves_by_k:
        raise ValueError("alpha_curves_by_k must include the anchor wave number")

    wave_numbers = sorted(alpha_curves_by_k)
    observed_tau_alpha_by_k: dict[float, float] = {}
    for wave_number in wave_numbers:
        if wave_number <= 0.0:
            raise ValueError("alpha wave numbers must be positive")
        time, alpha = _as_strictly_increasing_curve(
            alpha_curves_by_k[wave_number],
            name=f"alpha curve k={wave_number}",
        )
        observed_tau_alpha_by_k[wave_number] = _first_log_threshold_crossing_time(
            time,
            alpha,
            threshold=threshold,
            name=f"alpha curve k={wave_number}",
        )

    ngp_time, ngp_value = _as_strictly_increasing_curve(ngp_curve, name="NGP curve")
    if late_time < float(ngp_time[0]) or late_time > float(ngp_time[-1]):
        raise ValueError("late_time must lie inside the NGP raw-curve time grid")
    observed_late_ngp = float(np.interp(late_time, ngp_time, ngp_value))
    if observed_late_ngp <= 0.0:
        raise ValueError("interpolated late NGP must be positive")

    chi4_time, chi4_value = _as_strictly_increasing_curve(
        chi4_curve,
        name="chi4 curve",
        values_must_be_positive=False,
    )
    if np.any(chi4_value < 0.0):
        raise ValueError("chi4 values must be nonnegative")
    observed_chi4_peak = float(np.max(chi4_value))
    if observed_chi4_peak <= 0.0:
        raise ValueError("observed chi4 peak must be positive")

    scored = persistence_exchange_data_protocol(
        anchor_wave_number=anchor_wave_number,
        wave_numbers=wave_numbers,
        observed_tau_alpha_by_k=observed_tau_alpha_by_k,
        tau_alpha_relative_error_by_k=tau_alpha_relative_error_by_k,
        jump_variance=jump_variance,
        diffusion_coefficient=diffusion_coefficient,
        late_time=late_time,
        observed_late_ngp=observed_late_ngp,
        late_ngp_relative_error=late_ngp_relative_error,
        observed_chi4_peak=observed_chi4_peak,
        chi4_peak_relative_error=chi4_peak_relative_error,
        time_grid=chi4_time,
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        threshold=threshold,
        z_threshold=z_threshold,
    )
    out: dict[str, float | str] = {
        "benchmark_id": benchmark_id,
        "alpha_wave_numbers": ";".join(f"{wave_number:g}" for wave_number in wave_numbers),
        "raw_curve_protocol_passes": scored["passes_uncertainty_protocol"],
    }
    out.update(scored)
    return out


def _trajectory_adapter_sort_key(value: object) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def trajectory_table_adapter(
    *,
    records: Sequence[dict[str, object]],
    frame_column: str,
    time_column: str,
    particle_column: str,
    coordinate_columns: Sequence[str],
) -> dict[str, object]:
    """Convert a local particle table into protocol-ready trajectory arrays."""

    if not records:
        raise ValueError("records must be nonempty")
    for name, value in {
        "frame_column": frame_column,
        "time_column": time_column,
        "particle_column": particle_column,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    if not coordinate_columns:
        raise ValueError("coordinate_columns must be nonempty")
    if any(not column for column in coordinate_columns):
        raise ValueError("coordinate_columns must contain nonempty strings")

    coordinate_names = list(dict.fromkeys(coordinate_columns))
    required_columns = [frame_column, time_column, particle_column, *coordinate_names]
    for idx, record in enumerate(records):
        missing = [column for column in required_columns if column not in record]
        if missing:
            raise ValueError(f"record {idx} is missing required columns: {';'.join(missing)}")

    frame_values = sorted(
        dict.fromkeys(record[frame_column] for record in records),
        key=_trajectory_adapter_sort_key,
    )
    particle_values = sorted(
        dict.fromkeys(record[particle_column] for record in records),
        key=_trajectory_adapter_sort_key,
    )
    if len(frame_values) < 2:
        raise ValueError("at least two frames are required")
    if not particle_values:
        raise ValueError("at least one particle is required")

    frame_index = {value: idx for idx, value in enumerate(frame_values)}
    particle_index = {value: idx for idx, value in enumerate(particle_values)}
    times_by_frame: dict[object, float] = {}
    positions = np.empty((len(frame_values), len(particle_values), len(coordinate_names)), dtype=float)
    seen: set[tuple[int, int]] = set()
    for record in records:
        frame_value = record[frame_column]
        particle_value = record[particle_column]
        frame_idx = frame_index[frame_value]
        particle_idx = particle_index[particle_value]
        key = (frame_idx, particle_idx)
        if key in seen:
            raise ValueError("records must contain at most one row per frame-particle pair")
        seen.add(key)

        time_value = float(record[time_column])
        previous_time = times_by_frame.get(frame_value)
        if previous_time is not None and not math.isclose(previous_time, time_value, rel_tol=1e-14, abs_tol=1e-14):
            raise ValueError("all rows for a frame must share the same time")
        times_by_frame[frame_value] = time_value
        for dim_idx, column in enumerate(coordinate_names):
            positions[frame_idx, particle_idx, dim_idx] = float(record[column])

    expected_count = len(frame_values) * len(particle_values)
    if len(seen) != expected_count:
        raise ValueError("records must form a complete rectangular frame-particle table")

    times = np.array([times_by_frame[frame_value] for frame_value in frame_values], dtype=float)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("frame times must be strictly increasing after sorting")

    return {
        "positions": positions,
        "times": times,
        "frame_ids": ";".join(str(value) for value in frame_values),
        "particle_ids": ";".join(str(value) for value in particle_values),
        "coordinate_columns": ";".join(coordinate_names),
        "frame_count": float(len(frame_values)),
        "particle_count": float(len(particle_values)),
        "dimension": float(len(coordinate_names)),
        "adapter_ready": 1.0,
    }


def trajectory_table_csv_adapter(
    *,
    csv_path: object,
    frame_column: str,
    time_column: str,
    particle_column: str,
    coordinate_columns: Sequence[str],
    metadata: dict[str, object] | None = None,
    required_metadata_fields: Sequence[str] = (
        "box_geometry",
        "temperature_or_state_point",
        "species_labels",
        "units_metadata",
    ),
) -> dict[str, object]:
    """Load a local trajectory CSV and gate metadata before observable extraction."""

    path = Path(csv_path)
    if not str(path):
        raise ValueError("csv_path must be nonempty")
    if not path.exists():
        raise ValueError(f"csv_path does not exist: {path}")
    if not required_metadata_fields:
        raise ValueError("required_metadata_fields must be nonempty")
    if any(not field for field in required_metadata_fields):
        raise ValueError("required_metadata_fields must contain nonempty strings")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("trajectory CSV must include a header")
        records = list(reader)

    adapted = trajectory_table_adapter(
        records=records,
        frame_column=frame_column,
        time_column=time_column,
        particle_column=particle_column,
        coordinate_columns=coordinate_columns,
    )
    metadata_values = dict(metadata or {})
    required_metadata = list(dict.fromkeys(required_metadata_fields))
    missing_metadata = [
        field
        for field in required_metadata
        if field not in metadata_values or metadata_values[field] in {"", None}
    ]
    adapter_ready = not missing_metadata
    stage = "local_csv_trajectory_ready" if adapter_ready else "metadata_incomplete_csv_adapter"
    blocker = "none" if adapter_ready else missing_metadata[0]

    out = dict(adapted)
    out.update(
        {
            "csv_path": str(path),
            "csv_columns": ";".join(reader.fieldnames),
            "row_count": float(len(records)),
            "required_metadata_fields": ";".join(required_metadata),
            "available_metadata_fields": ";".join(
                field for field in required_metadata if field in metadata_values and metadata_values[field] not in {"", None}
            )
            or "none",
            "missing_metadata_fields": ";".join(missing_metadata) if missing_metadata else "none",
            "box_geometry": str(metadata_values.get("box_geometry", "none")),
            "temperature_or_state_point": str(metadata_values.get("temperature_or_state_point", "none")),
            "species_labels": str(metadata_values.get("species_labels", "none")),
            "units_metadata": str(metadata_values.get("units_metadata", "none")),
            "adapter_ready": float(adapter_ready),
            "primary_blocker": blocker,
            "adapter_stage": stage,
        }
    )
    return out


def trajectory_observable_protocol(
    *,
    positions: np.ndarray,
    times: np.ndarray,
    lag_indices: Sequence[int],
    wave_numbers: Sequence[float],
    overlap_radius: float,
) -> list[dict[str, float | str]]:
    """Extract model-facing observables directly from particle trajectories."""

    position_array = np.asarray(positions, dtype=float)
    time_array = np.asarray(times, dtype=float)
    if position_array.ndim != 3:
        raise ValueError("positions must have shape (frames, particles, dimensions)")
    if time_array.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if position_array.shape[0] != time_array.size:
        raise ValueError("times length must match the number of trajectory frames")
    if position_array.shape[0] < 2:
        raise ValueError("at least two frames are required")
    if position_array.shape[1] < 1 or position_array.shape[2] < 1:
        raise ValueError("positions must include particles and dimensions")
    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if not lag_indices:
        raise ValueError("lag_indices must be nonempty")
    if not wave_numbers:
        raise ValueError("wave_numbers must be nonempty")
    if overlap_radius <= 0.0:
        raise ValueError("overlap_radius must be positive")
    for lag in lag_indices:
        if int(lag) != lag or lag <= 0 or lag >= position_array.shape[0]:
            raise ValueError("lag_indices must be positive frame offsets smaller than the trajectory length")
    for wave_number in wave_numbers:
        if wave_number <= 0.0:
            raise ValueError("wave_numbers must be positive")

    unique_lags = list(dict.fromkeys(int(lag) for lag in lag_indices))
    unique_wave_numbers = list(dict.fromkeys(float(wave_number) for wave_number in wave_numbers))
    dimension = position_array.shape[2]
    particle_count = position_array.shape[1]
    rows: list[dict[str, float | str]] = []
    for lag in unique_lags:
        displacements = position_array[lag:, :, :] - position_array[:-lag, :, :]
        squared_radius = np.sum(displacements * displacements, axis=2)
        flat_squared_radius = squared_radius.reshape(-1)
        msd = float(np.mean(flat_squared_radius))
        fourth_moment = float(np.mean(flat_squared_radius * flat_squared_radius))
        ngp = 0.0 if msd == 0.0 else float(dimension / (dimension + 2.0) * fourth_moment / (msd * msd) - 1.0)
        projected_displacements = displacements[:, :, 0].reshape(-1)
        fs_values = [
            float(np.mean(np.cos(wave_number * projected_displacements)))
            for wave_number in unique_wave_numbers
        ]
        overlap_by_origin = np.mean(np.sqrt(squared_radius) <= overlap_radius, axis=1)
        lag_times = time_array[lag:] - time_array[:-lag]
        rows.append(
            {
                "lag_index": float(lag),
                "lag_time": float(np.mean(lag_times)),
                "time_origin_count": float(displacements.shape[0]),
                "particle_count": float(particle_count),
                "dimension": float(dimension),
                "msd": msd,
                "ngp": ngp,
                "wave_numbers": ";".join(f"{wave_number:g}" for wave_number in unique_wave_numbers),
                "self_intermediate_scattering_by_k": ";".join(f"{value:.12g}" for value in fs_values),
                "self_intermediate_scattering": fs_values[0],
                "overlap_radius": float(overlap_radius),
                "overlap_mean": float(np.mean(overlap_by_origin)),
                "chi4_overlap": float(particle_count * np.var(overlap_by_origin)),
                "structural_observable_set": "msd;ngp;self_intermediate_scattering;overlap_chi4",
            }
        )
    return rows


def trajectory_cage_jump_event_protocol(
    *,
    protocol_id: str,
    positions: np.ndarray,
    times: np.ndarray,
    jump_displacement_threshold: float,
    min_particles_with_jumps: int = 1,
    min_exchange_interval_count: int = 0,
) -> dict[str, float | str]:
    """Segment particle-resolved cage jumps into persistence and exchange clocks."""

    if not protocol_id:
        raise ValueError("protocol_id must be nonempty")
    position_array = np.asarray(positions, dtype=float)
    time_array = np.asarray(times, dtype=float)
    if position_array.ndim != 3:
        raise ValueError("positions must have shape (frames, particles, dimensions)")
    if time_array.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if position_array.shape[0] != time_array.size:
        raise ValueError("times length must match the number of trajectory frames")
    if position_array.shape[0] < 2:
        raise ValueError("at least two frames are required")
    if position_array.shape[1] < 1 or position_array.shape[2] < 1:
        raise ValueError("positions must include particles and dimensions")
    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if jump_displacement_threshold <= 0.0:
        raise ValueError("jump_displacement_threshold must be positive")
    if int(min_particles_with_jumps) != min_particles_with_jumps or min_particles_with_jumps < 1:
        raise ValueError("min_particles_with_jumps must be a positive integer")
    if int(min_exchange_interval_count) != min_exchange_interval_count or min_exchange_interval_count < 0:
        raise ValueError("min_exchange_interval_count must be a nonnegative integer")

    step_displacements = position_array[1:, :, :] - position_array[:-1, :, :]
    step_lengths = np.sqrt(np.sum(step_displacements * step_displacements, axis=2))
    jump_mask = step_lengths >= float(jump_displacement_threshold)
    jump_step_indices, jump_particle_indices = np.nonzero(jump_mask)
    jump_lengths = step_lengths[jump_mask]
    jump_times = time_array[jump_step_indices + 1]

    persistence_intervals: list[float] = []
    exchange_intervals: list[float] = []
    particles_with_exchange = 0
    for particle_idx in range(position_array.shape[1]):
        particle_jump_times = jump_times[jump_particle_indices == particle_idx]
        if particle_jump_times.size == 0:
            continue
        particle_jump_times = np.sort(particle_jump_times)
        persistence_intervals.append(float(particle_jump_times[0] - time_array[0]))
        if particle_jump_times.size >= 2:
            particles_with_exchange += 1
            exchange_intervals.extend(float(value) for value in np.diff(particle_jump_times))

    total_jump_event_count = int(jump_times.size)
    particles_with_jump_count = len(persistence_intervals)
    exchange_interval_count = len(exchange_intervals)
    particle_ready = particles_with_jump_count >= int(min_particles_with_jumps)
    exchange_ready = exchange_interval_count >= int(min_exchange_interval_count)
    event_clock_ready = particle_ready and exchange_ready

    if event_clock_ready:
        stage = "particle_resolved_cage_jump_event_clock_ready"
        blocker = "none"
    elif total_jump_event_count == 0:
        stage = "particle_resolved_cage_jump_events_incomplete"
        blocker = "jump_displacement_threshold"
    elif not particle_ready:
        stage = "particle_resolved_cage_jump_events_incomplete"
        blocker = "particles_with_jump_count"
    else:
        stage = "persistence_exchange_event_clock_incomplete"
        blocker = "exchange_intervals"

    first_jump_time = float(np.min(jump_times)) if total_jump_event_count else math.nan
    last_jump_time = float(np.max(jump_times)) if total_jump_event_count else math.nan
    mean_jump_length = float(np.mean(jump_lengths)) if total_jump_event_count else math.nan
    jump_length_variance = float(np.var(jump_lengths)) if total_jump_event_count else math.nan
    mean_squared_jump_length = float(np.mean(jump_lengths * jump_lengths)) if total_jump_event_count else math.nan
    persistence_mean = float(np.mean(persistence_intervals)) if persistence_intervals else math.nan
    exchange_mean = float(np.mean(exchange_intervals)) if exchange_intervals else math.nan

    return {
        "protocol_id": protocol_id,
        "frame_count": float(position_array.shape[0]),
        "particle_count": float(position_array.shape[1]),
        "dimension": float(position_array.shape[2]),
        "jump_displacement_threshold": float(jump_displacement_threshold),
        "total_jump_event_count": float(total_jump_event_count),
        "particles_with_jump_count": float(particles_with_jump_count),
        "particles_with_exchange_count": float(particles_with_exchange),
        "exchange_interval_count": float(exchange_interval_count),
        "first_jump_time_min": first_jump_time,
        "last_jump_time_max": last_jump_time,
        "mean_jump_length": mean_jump_length,
        "jump_length_variance": jump_length_variance,
        "mean_squared_jump_length": mean_squared_jump_length,
        "persistence_mean": persistence_mean,
        "exchange_mean": exchange_mean,
        "particle_resolved_jump_events_ready": float(particle_ready),
        "physical_time_jump_clock_ready": float(particle_ready),
        "persistence_exchange_event_clock_ready": float(event_clock_ready),
        "thermodynamic_claim_allowed": 0.0,
        "primary_blocker": blocker,
        "event_protocol_stage": stage,
    }


def _parse_semicolon_float_values(value: object, *, name: str) -> list[float]:
    parts = str(value).split(";")
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"{name} must contain semicolon-separated numeric values")
    return [float(part) for part in parts]


def trajectory_observable_curve_bridge(
    *,
    benchmark_id: str,
    rows: Sequence[dict[str, object]],
    required_wave_numbers: Sequence[float],
    anchor_wave_number: float,
    threshold: float = math.exp(-1.0),
) -> dict[str, float | str]:
    """Summarize trajectory-observable rows into raw-curve inversion inputs."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    if not rows:
        raise ValueError("rows must be nonempty")
    if not required_wave_numbers:
        raise ValueError("required_wave_numbers must be nonempty")
    if anchor_wave_number <= 0.0:
        raise ValueError("anchor_wave_number must be positive")
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("threshold must be between zero and one")
    required = list(dict.fromkeys(float(wave_number) for wave_number in required_wave_numbers))
    if any(wave_number <= 0.0 for wave_number in required):
        raise ValueError("required_wave_numbers must be positive")
    if float(anchor_wave_number) not in required:
        raise ValueError("required_wave_numbers must include the anchor wave number")

    required_columns = {
        "lag_time",
        "dimension",
        "msd",
        "ngp",
        "wave_numbers",
        "self_intermediate_scattering_by_k",
        "chi4_overlap",
    }
    sorted_rows = sorted(rows, key=lambda row: float(row["lag_time"]))
    lag_time = np.array([float(row["lag_time"]) for row in sorted_rows], dtype=float)
    if np.any(np.diff(lag_time) <= 0.0):
        raise ValueError("lag_time values must be strictly increasing")
    missing_columns = sorted(required_columns.difference(sorted_rows[0]))
    if missing_columns:
        raise ValueError(f"trajectory rows are missing required columns: {';'.join(missing_columns)}")

    dimension = float(sorted_rows[-1]["dimension"])
    latest_msd = float(sorted_rows[-1]["msd"])
    latest_time = float(sorted_rows[-1]["lag_time"])
    if dimension <= 0.0 or latest_time <= 0.0 or latest_msd <= 0.0:
        raise ValueError("dimension, latest lag_time, and latest msd must be positive")
    diffusion_coefficient = latest_msd / (2.0 * dimension * latest_time)
    ngp_values = np.array([float(row["ngp"]) for row in sorted_rows], dtype=float)
    chi4_values = np.array([float(row["chi4_overlap"]) for row in sorted_rows], dtype=float)
    if np.any(chi4_values < 0.0):
        raise ValueError("chi4_overlap values must be nonnegative")

    fs_by_k: dict[float, list[float]] = {wave_number: [] for wave_number in required}
    missing_fs = False
    for row in sorted_rows:
        row_wave_numbers = _parse_semicolon_float_values(row["wave_numbers"], name="wave_numbers")
        row_fs = _parse_semicolon_float_values(
            row["self_intermediate_scattering_by_k"],
            name="self_intermediate_scattering_by_k",
        )
        if len(row_wave_numbers) != len(row_fs):
            raise ValueError("wave_numbers and self_intermediate_scattering_by_k lengths must match")
        fs_lookup = {float(wave_number): float(value) for wave_number, value in zip(row_wave_numbers, row_fs)}
        for wave_number in required:
            if wave_number not in fs_lookup:
                missing_fs = True
            else:
                fs_by_k[wave_number].append(fs_lookup[wave_number])

    tau_by_k: dict[float, float] = {}
    alpha_crossing_missing = False
    if not missing_fs:
        for wave_number in required:
            fs_values = np.array(fs_by_k[wave_number], dtype=float)
            try:
                tau_by_k[wave_number] = _first_log_threshold_crossing_time(
                    lag_time,
                    fs_values,
                    threshold=threshold,
                    name=f"trajectory F_s k={wave_number}",
                )
            except ValueError:
                alpha_crossing_missing = True

    if missing_fs:
        stage = "trajectory_curve_bridge_incomplete"
        blocker = "self_intermediate_scattering_by_k"
        ready = False
    elif alpha_crossing_missing:
        stage = "trajectory_curve_bridge_incomplete"
        blocker = "alpha_threshold_crossing"
        ready = False
    else:
        stage = "trajectory_curve_bridge_ready"
        blocker = "none"
        ready = True

    anchor_tau = tau_by_k.get(float(anchor_wave_number), float("nan"))
    return {
        "benchmark_id": benchmark_id,
        "lag_count": float(len(sorted_rows)),
        "wave_numbers": ";".join(f"{wave_number:g}" for wave_number in required),
        "anchor_wave_number": float(anchor_wave_number),
        "tau_alpha_by_k": ";".join(
            f"{wave_number:g}:{tau_by_k[wave_number]:.12g}" for wave_number in required if wave_number in tau_by_k
        )
        or "none",
        "anchor_tau_alpha": float(anchor_tau) if np.isfinite(anchor_tau) else 0.0,
        "diffusion_coefficient": float(diffusion_coefficient),
        "d_tau_alpha_product": float(diffusion_coefficient * anchor_tau) if np.isfinite(anchor_tau) else 0.0,
        "late_time": float(latest_time),
        "late_ngp": float(ngp_values[-1]),
        "chi4_peak": float(np.max(chi4_values)),
        "alpha_curve_ready": float(not missing_fs and not alpha_crossing_missing),
        "ngp_curve_ready": 1.0,
        "chi4_curve_ready": 1.0,
        "diffusion_ready": 1.0,
        "curve_bridge_ready": float(ready),
        "primary_blocker": blocker,
        "bridge_stage": stage,
    }


def _parse_wave_number_value_map(value: object, *, name: str) -> dict[float, float]:
    text = str(value)
    if not text or text == "none":
        raise ValueError(f"{name} must contain wave:value entries")
    out: dict[float, float] = {}
    for entry in text.split(";"):
        if ":" not in entry:
            raise ValueError(f"{name} entries must have wave:value format")
        wave_text, value_text = entry.split(":", 1)
        wave_number = float(wave_text)
        if wave_number <= 0.0:
            raise ValueError(f"{name} wave numbers must be positive")
        out[wave_number] = float(value_text)
    return out


def trajectory_event_clock_macro_prediction_protocol(
    *,
    protocol_id: str,
    event_row: dict[str, object],
    anchor_wave_number: float,
    wave_numbers: Sequence[float],
    observed_diffusion_coefficient: float,
    diffusion_relative_error: float,
    observed_tau_alpha_by_k: dict[float, float],
    tau_alpha_relative_error_by_k: dict[float, float],
    late_time: float,
    observed_late_ngp: float,
    late_ngp_relative_error: float,
    observed_chi4_peak: float,
    chi4_peak_relative_error: float,
    time_grid: np.ndarray,
    cage_variance: float = 1.0,
    cage_tau: float = 0.2,
    threshold: float = math.exp(-1.0),
    z_threshold: float = 2.0,
) -> dict[str, float | str]:
    """Predict macro dynamical signatures directly from a trajectory event clock."""

    if not protocol_id:
        raise ValueError("protocol_id must be nonempty")
    event_ready = float(event_row.get("persistence_exchange_event_clock_ready", 0.0)) == 1.0
    if not event_ready:
        return {
            "protocol_id": protocol_id,
            "source_event_protocol_id": str(event_row.get("protocol_id", "unknown")),
            "micro_to_macro_prediction_ready": 0.0,
            "micro_to_macro_predictions_pass": 0.0,
            "calibrated_from_event_clock_only": 0.0,
            "fit_parameters_from_macro_observables": 0.0,
            "thermodynamic_claim_allowed": 0.0,
            "primary_blocker": str(event_row.get("primary_blocker", "persistence_exchange_event_clock_ready")),
            "prediction_stage": "event_clock_incomplete",
        }

    required_event_fields = [
        "persistence_mean",
        "exchange_mean",
        "mean_squared_jump_length",
        "dimension",
    ]
    missing_event_fields = [field for field in required_event_fields if field not in event_row]
    if missing_event_fields:
        return {
            "protocol_id": protocol_id,
            "source_event_protocol_id": str(event_row.get("protocol_id", "unknown")),
            "micro_to_macro_prediction_ready": 0.0,
            "micro_to_macro_predictions_pass": 0.0,
            "calibrated_from_event_clock_only": 0.0,
            "fit_parameters_from_macro_observables": 0.0,
            "thermodynamic_claim_allowed": 0.0,
            "primary_blocker": missing_event_fields[0],
            "prediction_stage": "event_clock_missing_jump_statistics",
        }

    if anchor_wave_number <= 0.0:
        raise ValueError("anchor_wave_number must be positive")
    if not wave_numbers:
        raise ValueError("wave_numbers must be nonempty")
    if observed_diffusion_coefficient <= 0.0 or late_time <= 0.0:
        raise ValueError("observed_diffusion_coefficient and late_time must be positive")
    if observed_late_ngp <= 0.0 or observed_chi4_peak <= 0.0:
        raise ValueError("observed_late_ngp and observed_chi4_peak must be positive")
    if cage_variance <= 0.0 or cage_tau <= 0.0:
        raise ValueError("cage parameters must be positive")
    if z_threshold <= 0.0:
        raise ValueError("z_threshold must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")

    unique_wave_numbers = list(dict.fromkeys(float(wave_number) for wave_number in wave_numbers))
    if anchor_wave_number not in unique_wave_numbers:
        unique_wave_numbers.insert(0, float(anchor_wave_number))
    for wave_number in unique_wave_numbers:
        if wave_number <= 0.0:
            raise ValueError("wave_numbers must be positive")
        if wave_number not in observed_tau_alpha_by_k:
            raise ValueError("observed_tau_alpha_by_k must include every wave number")
        if wave_number not in tau_alpha_relative_error_by_k:
            raise ValueError("tau_alpha_relative_error_by_k must include every wave number")
        if observed_tau_alpha_by_k[wave_number] <= 0.0:
            raise ValueError("observed tau_alpha values must be positive")

    persistence_mean = float(event_row["persistence_mean"])
    exchange_mean = float(event_row["exchange_mean"])
    dimension = float(event_row["dimension"])
    mean_squared_jump_length = float(event_row["mean_squared_jump_length"])
    if persistence_mean <= 0.0 or exchange_mean <= 0.0 or dimension <= 0.0 or mean_squared_jump_length <= 0.0:
        raise ValueError("event-clock means, dimension, and jump second moment must be positive")
    jump_variance = mean_squared_jump_length / dimension

    time_grid = np.asarray(time_grid, dtype=float)
    if time_grid.ndim != 1 or time_grid.size == 0 or np.any(time_grid <= 0.0):
        raise ValueError("time_grid must be a positive one-dimensional array")

    params = PersistenceExchangeParams(
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        jump_variance=jump_variance,
        persistence_mean=persistence_mean,
        exchange_mean=exchange_mean,
    )
    predicted_diffusion = persistence_exchange_diffusion_coefficient(params)
    predicted_tau_by_k = {
        wave_number: persistence_exchange_alpha_relaxation_time(
            wave_number,
            params,
            threshold=threshold,
        )
        for wave_number in unique_wave_numbers
    }
    predicted_late_ngp = float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0])
    predicted_chi4_peak = float(
        np.max(persistence_exchange_scattering_susceptibility(anchor_wave_number, time_grid, params))
    )

    diffusion_z = abs(math.log(observed_diffusion_coefficient / predicted_diffusion)) / _log_sigma_from_relative_error(
        diffusion_relative_error,
        "diffusion_relative_error",
    )
    max_tau_z = 0.0
    for wave_number in unique_wave_numbers:
        sigma = _log_sigma_from_relative_error(
            tau_alpha_relative_error_by_k[wave_number],
            "tau_alpha_relative_error_by_k",
        )
        residual = math.log(observed_tau_alpha_by_k[wave_number] / predicted_tau_by_k[wave_number])
        max_tau_z = max(max_tau_z, abs(residual) / sigma)
    late_ngp_z = abs(math.log(observed_late_ngp / predicted_late_ngp)) / _log_sigma_from_relative_error(
        late_ngp_relative_error,
        "late_ngp_relative_error",
    )
    chi4_z = abs(math.log(observed_chi4_peak / predicted_chi4_peak)) / _log_sigma_from_relative_error(
        chi4_peak_relative_error,
        "chi4_peak_relative_error",
    )
    diffusion_pass = diffusion_z <= z_threshold
    tau_pass = max_tau_z <= z_threshold
    late_ngp_pass = late_ngp_z <= z_threshold
    chi4_pass = chi4_z <= z_threshold
    predictions_pass = diffusion_pass and tau_pass and late_ngp_pass and chi4_pass

    return {
        "protocol_id": protocol_id,
        "source_event_protocol_id": str(event_row.get("protocol_id", "unknown")),
        "wave_numbers": ";".join(f"{wave_number:g}" for wave_number in unique_wave_numbers),
        "anchor_wave_number": float(anchor_wave_number),
        "jump_variance_from_event_clock": float(jump_variance),
        "persistence_mean": persistence_mean,
        "exchange_mean": exchange_mean,
        "persistence_exchange_ratio": persistence_mean / exchange_mean,
        "observed_diffusion_coefficient": float(observed_diffusion_coefficient),
        "predicted_diffusion_coefficient": float(predicted_diffusion),
        "diffusion_z": float(diffusion_z),
        "observed_tau_alpha_by_k": ";".join(
            f"{wave_number:g}:{observed_tau_alpha_by_k[wave_number]:.12g}" for wave_number in unique_wave_numbers
        ),
        "predicted_tau_alpha_by_k": ";".join(
            f"{wave_number:g}:{predicted_tau_by_k[wave_number]:.12g}" for wave_number in unique_wave_numbers
        ),
        "max_tau_alpha_z": float(max_tau_z),
        "late_time": float(late_time),
        "observed_late_ngp": float(observed_late_ngp),
        "predicted_late_ngp": float(predicted_late_ngp),
        "late_ngp_z": float(late_ngp_z),
        "observed_chi4_peak": float(observed_chi4_peak),
        "predicted_chi4_peak": float(predicted_chi4_peak),
        "chi4_peak_z": float(chi4_z),
        "z_threshold": float(z_threshold),
        "diffusion_prediction_pass": float(diffusion_pass),
        "tau_alpha_prediction_pass": float(tau_pass),
        "late_ngp_prediction_pass": float(late_ngp_pass),
        "chi4_peak_prediction_pass": float(chi4_pass),
        "micro_to_macro_prediction_ready": 1.0,
        "micro_to_macro_predictions_pass": float(predictions_pass),
        "calibrated_from_event_clock_only": 1.0,
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
        "primary_blocker": "none" if predictions_pass else "heldout_macro_signature_mismatch",
        "prediction_stage": "event_clock_micro_to_macro_prediction_ready"
        if predictions_pass
        else "event_clock_micro_to_macro_prediction_failed",
    }


def trajectory_event_clock_threshold_robustness_protocol(
    *,
    protocol_id: str,
    positions: np.ndarray,
    times: np.ndarray,
    thresholds: Sequence[float],
    reference_threshold: float,
    anchor_wave_number: float,
    wave_numbers: Sequence[float],
    late_time: float,
    time_grid: np.ndarray,
    min_particles_with_jumps: int = 1,
    min_exchange_interval_count: int = 0,
    cage_variance: float = 1.0,
    cage_tau: float = 0.2,
    diffusion_relative_error: float = 0.10,
    tau_alpha_relative_error: float = 0.10,
    late_ngp_relative_error: float = 0.20,
    chi4_peak_relative_error: float = 0.20,
    z_threshold: float = 2.0,
) -> list[dict[str, float | str]]:
    """Audit whether event-clock macro predictions are robust to jump threshold."""

    if not protocol_id:
        raise ValueError("protocol_id must be nonempty")
    if not thresholds:
        raise ValueError("thresholds must be nonempty")
    unique_thresholds = list(dict.fromkeys(float(threshold) for threshold in thresholds))
    if reference_threshold not in unique_thresholds:
        unique_thresholds.append(float(reference_threshold))
    reference_event = trajectory_cage_jump_event_protocol(
        protocol_id=f"{protocol_id}_reference_event_clock",
        positions=positions,
        times=times,
        jump_displacement_threshold=float(reference_threshold),
        min_particles_with_jumps=min_particles_with_jumps,
        min_exchange_interval_count=min_exchange_interval_count,
    )
    if float(reference_event.get("persistence_exchange_event_clock_ready", 0.0)) != 1.0:
        raise ValueError("reference_threshold must yield a ready event clock")

    reference_params = PersistenceExchangeParams(
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        jump_variance=float(reference_event["mean_squared_jump_length"]) / float(reference_event["dimension"]),
        persistence_mean=float(reference_event["persistence_mean"]),
        exchange_mean=float(reference_event["exchange_mean"]),
    )
    wave_list = list(dict.fromkeys(float(wave_number) for wave_number in wave_numbers))
    if anchor_wave_number not in wave_list:
        wave_list.insert(0, float(anchor_wave_number))
    observed_tau_alpha_by_k = {
        wave_number: persistence_exchange_alpha_relaxation_time(wave_number, reference_params)
        for wave_number in wave_list
    }
    observed_late_ngp = float(persistence_exchange_ngp_1d(np.array([late_time]), reference_params)[0])
    observed_chi4_peak = float(
        np.max(persistence_exchange_scattering_susceptibility(anchor_wave_number, time_grid, reference_params))
    )
    observed_diffusion = persistence_exchange_diffusion_coefficient(reference_params)
    tau_errors = {wave_number: tau_alpha_relative_error for wave_number in wave_list}

    rows: list[dict[str, float | str]] = []
    pass_count = 0
    pending_pass_rows: list[dict[str, float | str]] = []
    for threshold in unique_thresholds:
        event = trajectory_cage_jump_event_protocol(
            protocol_id=f"{protocol_id}_threshold_{threshold:g}",
            positions=positions,
            times=times,
            jump_displacement_threshold=threshold,
            min_particles_with_jumps=min_particles_with_jumps,
            min_exchange_interval_count=min_exchange_interval_count,
        )
        row: dict[str, float | str]
        if float(event.get("persistence_exchange_event_clock_ready", 0.0)) != 1.0:
            row = {
                "protocol_id": protocol_id,
                "jump_displacement_threshold": float(threshold),
                "reference_threshold": float(reference_threshold),
                "event_clock_ready": 0.0,
                "threshold_prediction_pass": 0.0,
                "stable_threshold_window_count": 0.0,
                "fit_parameters_from_macro_observables": 0.0,
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": str(event.get("primary_blocker", "event_clock_ready")),
                "robustness_stage": "event_clock_threshold_event_clock_incomplete",
            }
        else:
            prediction = trajectory_event_clock_macro_prediction_protocol(
                protocol_id=f"{protocol_id}_threshold_{threshold:g}",
                event_row=event,
                anchor_wave_number=anchor_wave_number,
                wave_numbers=wave_list,
                observed_diffusion_coefficient=observed_diffusion,
                diffusion_relative_error=diffusion_relative_error,
                observed_tau_alpha_by_k=observed_tau_alpha_by_k,
                tau_alpha_relative_error_by_k=tau_errors,
                late_time=late_time,
                observed_late_ngp=observed_late_ngp,
                late_ngp_relative_error=late_ngp_relative_error,
                observed_chi4_peak=observed_chi4_peak,
                chi4_peak_relative_error=chi4_peak_relative_error,
                time_grid=time_grid,
                cage_variance=cage_variance,
                cage_tau=cage_tau,
                z_threshold=z_threshold,
            )
            pass_flag = float(prediction["micro_to_macro_predictions_pass"]) == 1.0
            if pass_flag:
                pass_count += 1
            row = {
                "protocol_id": protocol_id,
                "jump_displacement_threshold": float(threshold),
                "reference_threshold": float(reference_threshold),
                "event_clock_ready": 1.0,
                "threshold_prediction_pass": float(pass_flag),
                "stable_threshold_window_count": 0.0,
                "total_jump_event_count": float(event["total_jump_event_count"]),
                "particles_with_jump_count": float(event["particles_with_jump_count"]),
                "exchange_interval_count": float(event["exchange_interval_count"]),
                "persistence_mean": float(event["persistence_mean"]),
                "exchange_mean": float(event["exchange_mean"]),
                "jump_variance_from_event_clock": float(prediction["jump_variance_from_event_clock"]),
                "diffusion_z": float(prediction["diffusion_z"]),
                "max_tau_alpha_z": float(prediction["max_tau_alpha_z"]),
                "late_ngp_z": float(prediction["late_ngp_z"]),
                "chi4_peak_z": float(prediction["chi4_peak_z"]),
                "fit_parameters_from_macro_observables": 0.0,
                "thermodynamic_claim_allowed": 0.0,
                "primary_blocker": "none" if pass_flag else "threshold_macro_signature_mismatch",
                "robustness_stage": "event_clock_threshold_prediction_passed"
                if pass_flag
                else "event_clock_threshold_prediction_failed",
            }
            if pass_flag:
                pending_pass_rows.append(row)
        rows.append(row)

    for row in pending_pass_rows:
        row["stable_threshold_window_count"] = float(pass_count)
    return rows


def trajectory_curve_persistence_exchange_gate(
    *,
    bridge_row: dict[str, object],
    jump_variance: float,
    tau_alpha_relative_error_by_k: dict[float, float],
    late_ngp_relative_error: float,
    chi4_peak_relative_error: float,
    cage_variance: float = 1.0,
    cage_tau: float = 0.2,
    threshold: float = math.exp(-1.0),
    z_threshold: float = 2.0,
) -> dict[str, float | str]:
    """Run the persistence/exchange protocol only after trajectory curve bridging."""

    benchmark_id = str(bridge_row.get("benchmark_id", "trajectory_curve_bridge"))
    bridge_ready = float(bridge_row.get("curve_bridge_ready", 0.0)) == 1.0
    if not bridge_ready:
        blocker = str(bridge_row.get("primary_blocker", "curve_bridge_ready"))
        return {
            "benchmark_id": benchmark_id,
            "trajectory_pe_protocol_ready": 0.0,
            "primary_blocker": blocker,
            "gate_stage": "trajectory_curve_bridge_incomplete",
        }
    if jump_variance <= 0.0:
        raise ValueError("jump_variance must be positive")

    required_columns = [
        "wave_numbers",
        "anchor_wave_number",
        "tau_alpha_by_k",
        "diffusion_coefficient",
        "late_time",
        "late_ngp",
        "chi4_peak",
    ]
    missing = [column for column in required_columns if column not in bridge_row]
    if missing:
        return {
            "benchmark_id": benchmark_id,
            "trajectory_pe_protocol_ready": 0.0,
            "primary_blocker": missing[0],
            "gate_stage": "trajectory_curve_bridge_incomplete",
        }

    wave_numbers = _parse_semicolon_float_values(bridge_row["wave_numbers"], name="wave_numbers")
    observed_tau_alpha_by_k = _parse_wave_number_value_map(
        bridge_row["tau_alpha_by_k"],
        name="tau_alpha_by_k",
    )
    anchor_wave_number = float(bridge_row["anchor_wave_number"])
    diffusion = float(bridge_row["diffusion_coefficient"])
    late_time = float(bridge_row["late_time"])
    late_ngp = float(bridge_row["late_ngp"])
    chi4_peak = float(bridge_row["chi4_peak"])
    if diffusion <= 0.0 or late_time <= 0.0 or late_ngp <= 0.0 or chi4_peak <= 0.0:
        raise ValueError("diffusion, late_time, late_ngp, and chi4_peak must be positive")

    error_by_k = {float(key): float(value) for key, value in tau_alpha_relative_error_by_k.items()}
    missing_error = [wave_number for wave_number in wave_numbers if wave_number not in error_by_k]
    if missing_error:
        return {
            "benchmark_id": benchmark_id,
            "trajectory_pe_protocol_ready": 0.0,
            "primary_blocker": "tau_alpha_relative_error_by_k",
            "gate_stage": "trajectory_uncertainty_incomplete",
        }

    max_time = max(10.0, 2.0 * late_time, 20.0 * max(observed_tau_alpha_by_k.values()))
    min_time = max(1e-4, min(cage_tau, min(observed_tau_alpha_by_k.values())) / 100.0)
    time_grid = np.geomspace(min_time, max_time, 1200)
    scored = persistence_exchange_data_protocol(
        anchor_wave_number=anchor_wave_number,
        wave_numbers=wave_numbers,
        observed_tau_alpha_by_k=observed_tau_alpha_by_k,
        tau_alpha_relative_error_by_k=error_by_k,
        jump_variance=jump_variance,
        diffusion_coefficient=diffusion,
        late_time=late_time,
        observed_late_ngp=late_ngp,
        late_ngp_relative_error=late_ngp_relative_error,
        observed_chi4_peak=chi4_peak,
        chi4_peak_relative_error=chi4_peak_relative_error,
        time_grid=time_grid,
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        threshold=threshold,
        z_threshold=z_threshold,
    )
    out: dict[str, float | str] = {
        "benchmark_id": benchmark_id,
        "trajectory_pe_protocol_ready": 1.0,
        "primary_blocker": "none",
        "gate_stage": "trajectory_persistence_exchange_protocol_ready",
    }
    out.update(scored)
    return out


def trajectory_pe_heldout_prediction_gate(
    *,
    pe_gate_row: dict[str, object],
    jump_variance: float,
    heldout_wave_number: float,
    observed_heldout_tau_alpha: float,
    heldout_tau_alpha_relative_error: float,
    heldout_late_time: float,
    observed_heldout_late_ngp: float,
    heldout_late_ngp_relative_error: float,
    cage_variance: float = 1.0,
    cage_tau: float = 0.2,
    threshold: float = math.exp(-1.0),
    z_threshold: float = 2.0,
) -> dict[str, float | str]:
    """Score held-out trajectory predictions after persistence/exchange inversion."""

    benchmark_id = str(pe_gate_row.get("benchmark_id", "trajectory_pe_gate"))
    pe_ready = float(pe_gate_row.get("trajectory_pe_protocol_ready", 0.0)) == 1.0
    if not pe_ready:
        blocker = str(pe_gate_row.get("primary_blocker", "trajectory_pe_protocol_ready"))
        return {
            "benchmark_id": benchmark_id,
            "heldout_prediction_ready": 0.0,
            "heldout_predictions_pass": 0.0,
            "primary_blocker": blocker,
            "prediction_stage": "trajectory_pe_gate_incomplete",
        }
    for name, value in {
        "jump_variance": jump_variance,
        "heldout_wave_number": heldout_wave_number,
        "observed_heldout_tau_alpha": observed_heldout_tau_alpha,
        "heldout_late_time": heldout_late_time,
        "observed_heldout_late_ngp": observed_heldout_late_ngp,
        "z_threshold": z_threshold,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    missing = [column for column in ["exchange_mean", "persistence_mean"] if column not in pe_gate_row]
    if missing:
        return {
            "benchmark_id": benchmark_id,
            "heldout_prediction_ready": 0.0,
            "heldout_predictions_pass": 0.0,
            "primary_blocker": missing[0],
            "prediction_stage": "trajectory_pe_gate_incomplete",
        }

    params = PersistenceExchangeParams(
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        jump_variance=jump_variance,
        persistence_mean=float(pe_gate_row["persistence_mean"]),
        exchange_mean=float(pe_gate_row["exchange_mean"]),
    )
    predicted_tau = persistence_exchange_alpha_relaxation_time(
        heldout_wave_number,
        params,
        threshold=threshold,
    )
    predicted_ngp = float(persistence_exchange_ngp_1d(np.array([heldout_late_time]), params)[0])
    tau_sigma = _log_sigma_from_relative_error(
        heldout_tau_alpha_relative_error,
        "heldout_tau_alpha_relative_error",
    )
    ngp_sigma = _log_sigma_from_relative_error(
        heldout_late_ngp_relative_error,
        "heldout_late_ngp_relative_error",
    )
    tau_z = abs(math.log(observed_heldout_tau_alpha / predicted_tau)) / tau_sigma
    ngp_z = abs(math.log(observed_heldout_late_ngp / predicted_ngp)) / ngp_sigma
    tau_pass = tau_z <= z_threshold
    ngp_pass = ngp_z <= z_threshold
    predictions_pass = tau_pass and ngp_pass

    return {
        "benchmark_id": benchmark_id,
        "heldout_wave_number": float(heldout_wave_number),
        "observed_heldout_tau_alpha": float(observed_heldout_tau_alpha),
        "predicted_heldout_tau_alpha": float(predicted_tau),
        "heldout_tau_alpha_z": float(tau_z),
        "heldout_late_time": float(heldout_late_time),
        "observed_heldout_late_ngp": float(observed_heldout_late_ngp),
        "predicted_heldout_late_ngp": float(predicted_ngp),
        "heldout_late_ngp_z": float(ngp_z),
        "z_threshold": float(z_threshold),
        "heldout_tau_alpha_pass": float(tau_pass),
        "heldout_late_ngp_pass": float(ngp_pass),
        "heldout_predictions_pass": float(predictions_pass),
        "heldout_prediction_ready": 1.0,
        "primary_blocker": "none" if predictions_pass else "heldout_prediction_mismatch",
        "prediction_stage": "trajectory_pe_heldout_prediction_ready"
        if predictions_pass
        else "trajectory_pe_heldout_prediction_failed",
    }


def trajectory_prediction_falsification_gate(
    *,
    protocol_id: str,
    prediction_row: dict[str, object],
    calibration_observables: Sequence[str],
    heldout_observables: Sequence[str],
    required_prediction_passes: Sequence[str],
) -> dict[str, float | str]:
    """Convert held-out trajectory predictions into a falsifiable protocol status."""

    if not protocol_id:
        raise ValueError("protocol_id must be nonempty")
    if not calibration_observables:
        raise ValueError("calibration_observables must be nonempty")
    if not required_prediction_passes:
        raise ValueError("required_prediction_passes must be nonempty")
    for name, values in {
        "calibration_observables": calibration_observables,
        "heldout_observables": heldout_observables,
        "required_prediction_passes": required_prediction_passes,
    }.items():
        if any(not value for value in values):
            raise ValueError(f"{name} must contain nonempty strings")

    benchmark_id = str(prediction_row.get("benchmark_id", "trajectory_prediction"))
    calibration = list(dict.fromkeys(calibration_observables))
    heldout = list(dict.fromkeys(heldout_observables))
    required = list(dict.fromkeys(required_prediction_passes))

    base: dict[str, float | str] = {
        "protocol_id": protocol_id,
        "benchmark_id": benchmark_id,
        "calibration_observables": ";".join(calibration),
        "heldout_observables": ";".join(heldout) if heldout else "none",
        "required_prediction_passes": ";".join(required),
        "calibration_count": float(len(calibration)),
        "heldout_count": float(len(heldout)),
        "required_prediction_count": float(len(required)),
    }

    upstream_ready = float(prediction_row.get("heldout_prediction_ready", 0.0)) == 1.0
    if not upstream_ready:
        base.update(
            {
                "trajectory_falsification_ready": 0.0,
                "trajectory_predictions_falsified": 0.0,
                "all_required_predictions_pass": 0.0,
                "fit_only_overclaim_risk": 0.0,
                "primary_blocker": str(
                    prediction_row.get("primary_blocker", "heldout_prediction_ready")
                ),
                "falsification_stage": "upstream_prediction_incomplete",
            }
        )
        return base

    if not heldout:
        base.update(
            {
                "trajectory_falsification_ready": 0.0,
                "trajectory_predictions_falsified": 0.0,
                "all_required_predictions_pass": 0.0,
                "fit_only_overclaim_risk": 1.0,
                "primary_blocker": "heldout_observables",
                "falsification_stage": "fit_only_overclaim_risk",
            }
        )
        return base

    failed_flags = [
        flag for flag in required if float(prediction_row.get(flag, 0.0)) != 1.0
    ]
    predictions_pass = not failed_flags
    base.update(
        {
            "trajectory_falsification_ready": 1.0,
            "trajectory_predictions_falsified": float(not predictions_pass),
            "all_required_predictions_pass": float(predictions_pass),
            "fit_only_overclaim_risk": 0.0,
            "primary_blocker": "none" if predictions_pass else failed_flags[0],
            "falsification_stage": "trajectory_prediction_falsification_passed"
            if predictions_pass
            else "heldout_prediction_failed",
        }
    )
    return base


def benchmark_publication_ladder(
    *,
    ladder_id: str,
    source_key: str,
    source_class: str,
    evidence_grade: str,
    reanalysis_stage: str,
    readiness_stage: str,
    falsification_stage: str,
    primary_blocker: str,
) -> dict[str, float | str]:
    """Collapse benchmark gates into manuscript-safe publication claim levels."""

    for name, value in {
        "ladder_id": ladder_id,
        "source_key": source_key,
        "source_class": source_class,
        "evidence_grade": evidence_grade,
        "reanalysis_stage": reanalysis_stage,
        "readiness_stage": readiness_stage,
        "falsification_stage": falsification_stage,
        "primary_blocker": primary_blocker,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    allowed_source_classes = {"real_public_data", "synthetic_canary", "thermodynamic_boundary"}
    allowed_evidence = {
        "direct_dynamical_support",
        "closure_assisted_support",
        "thermodynamic_scope_boundary",
        "pending_trajectory_reanalysis",
        "overclaimed_or_forbidden",
        "not_supported",
    }
    if source_class not in allowed_source_classes:
        raise ValueError("source_class is not recognized")
    if evidence_grade not in allowed_evidence:
        raise ValueError("evidence_grade is not recognized")

    readiness_ready = readiness_stage == "uncertainty_weighted_trajectory_inversion"
    falsification_passed = falsification_stage == "trajectory_prediction_falsification_passed"
    fit_only_risk = falsification_stage == "fit_only_overclaim_risk"
    awaiting_reanalysis = reanalysis_stage.startswith("awaiting_")

    if evidence_grade == "overclaimed_or_forbidden" or fit_only_risk:
        stage = "fit_only_overclaim_blocked" if fit_only_risk else "forbidden_claim_blocked"
        allowed_claim = "do_not_claim_prediction"
        protocol_evidence = False
        real_comparison = False
        overreach = True
        next_action = "add_heldout_predictions_before_claiming_fit"
    elif evidence_grade == "thermodynamic_scope_boundary" or source_class == "thermodynamic_boundary":
        stage = "thermodynamic_scope_boundary"
        allowed_claim = "scope_boundary_only"
        protocol_evidence = True
        real_comparison = False
        overreach = False
        next_action = "do_not_convert_closure_to_thermodynamic_derivation"
    elif source_class == "synthetic_canary" and readiness_ready and falsification_passed:
        stage = "synthetic_prediction_canary_passed"
        allowed_claim = "protocol_canary_passed"
        protocol_evidence = True
        real_comparison = False
        overreach = False
        next_action = "repeat_protocol_on_real_public_trajectory"
    elif source_class == "real_public_data" and readiness_ready and falsification_passed:
        stage = "uncertainty_weighted_real_reanalysis"
        allowed_claim = "real_data_quantitative_comparison"
        protocol_evidence = True
        real_comparison = True
        overreach = False
        next_action = "report_uncertainty_weighted_residuals"
    elif evidence_grade == "pending_trajectory_reanalysis" or awaiting_reanalysis:
        stage = "metadata_verified_not_reanalysis"
        allowed_claim = "metadata_readiness_only"
        protocol_evidence = False
        real_comparison = False
        overreach = True
        next_action = (
            "cache_full_archive_and_verify_checksum"
            if "archive" in primary_blocker or "cache" in reanalysis_stage
            else "complete_trajectory_reanalysis_gate"
        )
    elif not readiness_ready:
        stage = "structural_or_uncertainty_gate_incomplete"
        allowed_claim = "readiness_gap_only"
        protocol_evidence = False
        real_comparison = False
        overreach = True
        next_action = primary_blocker
    elif not falsification_passed:
        stage = "heldout_prediction_not_passed"
        allowed_claim = "do_not_claim_prediction"
        protocol_evidence = False
        real_comparison = False
        overreach = True
        next_action = primary_blocker
    else:
        stage = "not_supported"
        allowed_claim = "do_not_claim"
        protocol_evidence = False
        real_comparison = False
        overreach = True
        next_action = primary_blocker

    return {
        "ladder_id": ladder_id,
        "source_key": source_key,
        "source_class": source_class,
        "evidence_grade": evidence_grade,
        "reanalysis_stage": reanalysis_stage,
        "readiness_stage": readiness_stage,
        "falsification_stage": falsification_stage,
        "primary_blocker": primary_blocker,
        "publication_stage": stage,
        "allowed_manuscript_claim": allowed_claim,
        "publishable_protocol_evidence": float(protocol_evidence),
        "real_data_quantitative_comparison": float(real_comparison),
        "claim_overreach_if_called_fit": float(overreach),
        "next_required_action": next_action,
    }


def trajectory_observable_uncertainty_protocol(
    *,
    positions: np.ndarray,
    times: np.ndarray,
    lag_indices: Sequence[int],
    wave_numbers: Sequence[float],
    overlap_radius: float,
    block_count: int,
) -> list[dict[str, float | str]]:
    """Add time-origin block-jackknife uncertainties to trajectory observables."""

    if int(block_count) != block_count or block_count < 2:
        raise ValueError("block_count must be an integer at least two")
    full_rows = trajectory_observable_protocol(
        positions=positions,
        times=times,
        lag_indices=lag_indices,
        wave_numbers=wave_numbers,
        overlap_radius=overlap_radius,
    )
    position_array = np.asarray(positions, dtype=float)
    unique_wave_numbers = list(dict.fromkeys(float(wave_number) for wave_number in wave_numbers))
    dimension = position_array.shape[2]
    particle_count = position_array.shape[1]

    def observables_for_subset(squared_radius: np.ndarray, displacements: np.ndarray) -> dict[str, float]:
        flat_squared_radius = squared_radius.reshape(-1)
        msd = float(np.mean(flat_squared_radius))
        fourth_moment = float(np.mean(flat_squared_radius * flat_squared_radius))
        ngp = 0.0 if msd == 0.0 else float(dimension / (dimension + 2.0) * fourth_moment / (msd * msd) - 1.0)
        projected_displacements = displacements[:, :, 0].reshape(-1)
        fs_values = [
            float(np.mean(np.cos(wave_number * projected_displacements)))
            for wave_number in unique_wave_numbers
        ]
        overlap_by_origin = np.mean(np.sqrt(squared_radius) <= overlap_radius, axis=1)
        return {
            "msd": msd,
            "ngp": ngp,
            "self_intermediate_scattering": fs_values[0],
            "overlap_mean": float(np.mean(overlap_by_origin)),
            "chi4_overlap": float(particle_count * np.var(overlap_by_origin)),
        }

    out: list[dict[str, float | str]] = []
    for full_row in full_rows:
        lag = int(float(full_row["lag_index"]))
        displacements = position_array[lag:, :, :] - position_array[:-lag, :, :]
        origin_count = displacements.shape[0]
        if origin_count < 2:
            raise ValueError("each lag must have at least two time origins for jackknife uncertainty")
        actual_block_count = min(int(block_count), origin_count)
        if actual_block_count < 2:
            raise ValueError("block_count leaves fewer than two jackknife blocks")
        origin_blocks = np.array_split(np.arange(origin_count), actual_block_count)
        jackknife_rows: list[dict[str, float]] = []
        squared_radius = np.sum(displacements * displacements, axis=2)
        for block in origin_blocks:
            mask = np.ones(origin_count, dtype=bool)
            mask[block] = False
            if not np.any(mask):
                raise ValueError("jackknife block removes all time origins")
            jackknife_rows.append(observables_for_subset(squared_radius[mask], displacements[mask]))

        row = dict(full_row)
        for key in [
            "msd",
            "ngp",
            "self_intermediate_scattering",
            "overlap_mean",
            "chi4_overlap",
        ]:
            values = np.array([float(jackknife_row[key]) for jackknife_row in jackknife_rows])
            mean = float(np.mean(values))
            sigma = math.sqrt((actual_block_count - 1.0) / actual_block_count * float(np.sum((values - mean) ** 2)))
            row[f"sigma_{key}"] = sigma
        row["uncertainty_method"] = "time_origin_block_jackknife"
        row["jackknife_block_count"] = float(actual_block_count)
        row["uncertainty_estimates"] = 1.0
        row["primary_blocker"] = "none"
        out.append(row)
    return out


def trajectory_member_ensemble_uncertainty_protocol(
    *,
    member_rows: Sequence[dict[str, object]],
    min_member_count: int,
) -> list[dict[str, float | str]]:
    """Aggregate independent trajectory-member observables into uncertainty rows."""

    if not member_rows:
        raise ValueError("member_rows must be nonempty")
    if int(min_member_count) != min_member_count or min_member_count < 2:
        raise ValueError("min_member_count must be an integer at least two")

    required_columns = {
        "member_id",
        "lag_time",
        "time_origin_count",
        "particle_count",
        "dimension",
        "msd",
        "ngp",
        "wave_numbers",
        "self_intermediate_scattering_by_k",
        "self_intermediate_scattering",
        "overlap_radius",
        "overlap_mean",
        "chi4_overlap",
    }
    missing = sorted(required_columns.difference(member_rows[0]))
    if missing:
        raise ValueError(f"member rows are missing required columns: {';'.join(missing)}")

    grouped: dict[float, list[dict[str, object]]] = {}
    for row in member_rows:
        missing = sorted(required_columns.difference(row))
        if missing:
            raise ValueError(f"member rows are missing required columns: {';'.join(missing)}")
        lag_time = float(row["lag_time"])
        if lag_time <= 0.0:
            raise ValueError("lag_time values must be positive")
        grouped.setdefault(lag_time, []).append(row)

    def mean_and_standard_error(values: Sequence[float]) -> tuple[float, float]:
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            raise ValueError("cannot aggregate an empty value set")
        if not np.all(np.isfinite(array)):
            raise ValueError("member observable values must be finite")
        mean = float(np.mean(array))
        if array.size < 2:
            return mean, 0.0
        sigma = float(np.std(array, ddof=1) / math.sqrt(array.size))
        return mean, sigma

    out: list[dict[str, float | str]] = []
    for lag_time in sorted(grouped):
        group = grouped[lag_time]
        members = sorted({str(row["member_id"]) for row in group if str(row["member_id"])})
        member_count = len(members)
        wave_numbers = str(group[0]["wave_numbers"])
        dimension = float(group[0]["dimension"])
        particle_count = float(group[0]["particle_count"])
        overlap_radius = float(group[0]["overlap_radius"])
        if dimension <= 0.0 or particle_count <= 0.0 or overlap_radius <= 0.0:
            raise ValueError("dimension, particle_count, and overlap_radius must be positive")
        fs_by_member: list[list[float]] = []
        for row in group:
            if str(row["wave_numbers"]) != wave_numbers:
                raise ValueError("all members at a lag must share the same wave_numbers")
            if float(row["dimension"]) != dimension:
                raise ValueError("all members at a lag must share the same dimension")
            if float(row["particle_count"]) != particle_count:
                raise ValueError("all members at a lag must share the same particle_count")
            values = _parse_semicolon_float_values(
                row["self_intermediate_scattering_by_k"],
                name="self_intermediate_scattering_by_k",
            )
            if any(not np.isfinite(value) for value in values):
                raise ValueError("self_intermediate_scattering_by_k values must be finite")
            fs_by_member.append(values)
        fs_lengths = {len(values) for values in fs_by_member}
        if len(fs_lengths) != 1:
            raise ValueError("all members at a lag must have the same number of Fs values")

        msd, sigma_msd = mean_and_standard_error([float(row["msd"]) for row in group])
        ngp, sigma_ngp = mean_and_standard_error([float(row["ngp"]) for row in group])
        fs, sigma_fs = mean_and_standard_error(
            [float(row["self_intermediate_scattering"]) for row in group]
        )
        overlap_mean, sigma_overlap = mean_and_standard_error(
            [float(row["overlap_mean"]) for row in group]
        )
        chi4, sigma_chi4 = mean_and_standard_error([float(row["chi4_overlap"]) for row in group])
        fs_array = np.asarray(fs_by_member, dtype=float)
        fs_means = np.mean(fs_array, axis=0)
        if member_count < 2:
            fs_sigmas = np.zeros(fs_array.shape[1])
        else:
            fs_sigmas = np.std(fs_array, axis=0, ddof=1) / math.sqrt(member_count)
        ready = member_count >= int(min_member_count)
        out.append(
            {
                "lag_index": float(group[0].get("lag_index", lag_time)),
                "lag_time": float(lag_time),
                "member_ids": ";".join(members) if members else "none",
                "member_count": float(member_count),
                "min_member_count": float(min_member_count),
                "time_origin_count_mean": float(
                    np.mean([float(row["time_origin_count"]) for row in group])
                ),
                "particle_count": float(particle_count),
                "dimension": float(dimension),
                "msd": msd,
                "ngp": ngp,
                "wave_numbers": wave_numbers,
                "self_intermediate_scattering_by_k": ";".join(f"{value:.12g}" for value in fs_means),
                "self_intermediate_scattering": fs,
                "overlap_radius": float(overlap_radius),
                "overlap_mean": overlap_mean,
                "chi4_overlap": chi4,
                "sigma_msd": sigma_msd,
                "sigma_ngp": sigma_ngp,
                "sigma_self_intermediate_scattering_by_k": ";".join(
                    f"{value:.12g}" for value in fs_sigmas
                ),
                "sigma_self_intermediate_scattering": sigma_fs,
                "sigma_overlap_mean": sigma_overlap,
                "sigma_chi4_overlap": sigma_chi4,
                "structural_observable_set": "msd;ngp;self_intermediate_scattering;overlap_chi4",
                "uncertainty_method": "member_ensemble_standard_error",
                "ensemble_uncertainty_ready": float(ready),
                "uncertainty_estimates": float(ready),
                "primary_blocker": "none" if ready else "member_count",
                "ensemble_stage": "member_ensemble_uncertainty_ready"
                if ready
                else "member_ensemble_below_threshold",
            }
        )
    return out


def trajectory_inversion_readiness_gate(
    *,
    benchmark_id: str,
    source_key: str,
    target_protocol: str,
    trajectory_rows: Sequence[dict[str, float | str]],
    required_observables: Sequence[str],
    required_uncertainty_columns: Sequence[str],
    has_shared_time_grid: bool,
    has_shared_particle_identity: bool,
) -> dict[str, float | str]:
    """Gate trajectory-derived observables before uncertainty-weighted inversion."""

    for name, value in {
        "benchmark_id": benchmark_id,
        "source_key": source_key,
        "target_protocol": target_protocol,
    }.items():
        if not value:
            raise ValueError(f"{name} must be nonempty")
    if not trajectory_rows:
        raise ValueError("trajectory_rows must be nonempty")
    if not required_observables:
        raise ValueError("required_observables must be nonempty")
    if any(not observable for observable in required_observables):
        raise ValueError("required_observables must contain nonempty strings")
    if any(not column for column in required_uncertainty_columns):
        raise ValueError("required_uncertainty_columns must contain nonempty strings")

    required = list(dict.fromkeys(required_observables))
    required_sigmas = list(dict.fromkeys(required_uncertainty_columns))
    available: list[str] = []
    available_sigmas: list[str] = []
    positive_sigma_columns: set[str] = set()
    for row in trajectory_rows:
        structural_set = str(row.get("structural_observable_set", ""))
        if structural_set and structural_set != "none":
            available.extend(item for item in structural_set.split(";") if item)
        for key, value in row.items():
            if key.startswith("sigma_"):
                available_sigmas.append(key)
                try:
                    if float(value) > 0.0:
                        positive_sigma_columns.add(key)
                except (TypeError, ValueError):
                    pass
        for observable in required:
            if observable in row:
                available.append(observable)
    available_unique = list(dict.fromkeys(available))
    sigma_unique = list(dict.fromkeys(available_sigmas))
    available_set = set(available_unique)
    sigma_set = set(sigma_unique)
    missing_observables = [observable for observable in required if observable not in available_set]
    missing_sigma_columns = [column for column in required_sigmas if column not in sigma_set]
    nonpositive_sigma_columns = [
        column
        for column in required_sigmas
        if column in sigma_set and column not in positive_sigma_columns
    ]

    structural_ready = (
        not missing_observables
        and has_shared_time_grid
        and has_shared_particle_identity
    )
    uncertainty_ready = (
        structural_ready
        and not missing_sigma_columns
        and not nonpositive_sigma_columns
    )
    if uncertainty_ready:
        stage = "uncertainty_weighted_trajectory_inversion"
        blocker = "none"
    elif not has_shared_time_grid:
        stage = "trajectory_blocked"
        blocker = "shared_time_grid"
    elif not has_shared_particle_identity:
        stage = "trajectory_blocked"
        blocker = "shared_particle_identity"
    elif missing_observables:
        stage = "trajectory_blocked"
        blocker = missing_observables[0]
    elif missing_sigma_columns:
        stage = "structural_trajectory_only"
        blocker = missing_sigma_columns[0]
    elif nonpositive_sigma_columns:
        stage = "structural_trajectory_only"
        blocker = nonpositive_sigma_columns[0]
    else:
        stage = "structural_trajectory_only"
        blocker = "uncertainty_columns"

    return {
        "benchmark_id": benchmark_id,
        "source_key": source_key,
        "target_protocol": target_protocol,
        "required_observables": ";".join(required),
        "available_observables": ";".join(available_unique) if available_unique else "none",
        "missing_observables": ";".join(missing_observables) if missing_observables else "none",
        "required_uncertainty_columns": ";".join(required_sigmas) if required_sigmas else "none",
        "available_uncertainty_columns": ";".join(sigma_unique) if sigma_unique else "none",
        "missing_uncertainty_columns": ";".join(missing_sigma_columns) if missing_sigma_columns else "none",
        "nonpositive_uncertainty_columns": (
            ";".join(nonpositive_sigma_columns) if nonpositive_sigma_columns else "none"
        ),
        "lag_count": float(len(trajectory_rows)),
        "shared_time_grid": float(has_shared_time_grid),
        "shared_particle_identity": float(has_shared_particle_identity),
        "structural_trajectory_ready": float(structural_ready),
        "uncertainty_weighted_ready": float(uncertainty_ready),
        "primary_blocker": blocker,
        "readiness_stage": stage,
    }


def van_hove_tail_benchmark_consistency(
    *,
    benchmark_id: str,
    observed_transient_van_hove_tail: bool,
    observed_late_gaussian_recovery: bool,
    peak_tail_ratio: float,
    late_tail_ratio: float,
    peak_ngp: float,
    min_peak_tail_ratio: float,
    max_late_tail_deviation: float,
    min_peak_ngp: float,
) -> dict[str, float | str]:
    """Check transient self-van-Hove tails together with late Gaussian recovery."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    for name, value in {
        "peak_tail_ratio": peak_tail_ratio,
        "late_tail_ratio": late_tail_ratio,
        "min_peak_tail_ratio": min_peak_tail_ratio,
        "max_late_tail_deviation": max_late_tail_deviation,
        "min_peak_ngp": min_peak_ngp,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if peak_ngp < 0.0:
        raise ValueError("peak_ngp must be nonnegative")

    tail_flag = peak_tail_ratio >= min_peak_tail_ratio
    late_deviation = abs(late_tail_ratio - 1.0)
    recovery_flag = late_deviation <= max_late_tail_deviation
    peak_ngp_flag = peak_ngp >= min_peak_ngp
    model_tail_flag = tail_flag and peak_ngp_flag
    tail_consistent = model_tail_flag == observed_transient_van_hove_tail
    recovery_consistent = recovery_flag == observed_late_gaussian_recovery
    peak_ngp_consistent = peak_ngp_flag == observed_transient_van_hove_tail
    return {
        "benchmark_id": benchmark_id,
        "observed_transient_van_hove_tail": float(observed_transient_van_hove_tail),
        "observed_late_tail_gaussian_recovery": float(observed_late_gaussian_recovery),
        "peak_tail_ratio": peak_tail_ratio,
        "late_tail_ratio": late_tail_ratio,
        "late_tail_abs_deviation": late_deviation,
        "peak_ngp_benchmark": peak_ngp,
        "min_peak_tail_ratio": min_peak_tail_ratio,
        "max_late_tail_deviation": max_late_tail_deviation,
        "min_peak_ngp": min_peak_ngp,
        "model_predicts_transient_van_hove_tail": float(model_tail_flag),
        "model_predicts_tail_gaussian_recovery": float(recovery_flag),
        "van_hove_tail_consistent": float(tail_consistent),
        "tail_recovery_consistent": float(recovery_consistent),
        "peak_ngp_consistent": float(peak_ngp_consistent),
        "overall_consistent": float(tail_consistent and recovery_consistent and peak_ngp_consistent),
    }


def fragility_benchmark_consistency(
    *,
    benchmark_id: str,
    observed_fragility_growth: bool,
    observed_adam_gibbs_slowdown: bool,
    hot_activation_energy: float,
    cold_activation_energy: float,
    hot_fragility_index: float,
    cold_fragility_index: float,
    adam_gibbs_slowdown: float,
    material_specific_origin_claimed: bool,
    min_activation_growth: float,
    min_fragility_growth: float,
    min_adam_gibbs_slowdown: float,
) -> dict[str, float | str]:
    """Check effective fragility growth while preserving the material-origin boundary."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    for name, value in {
        "hot_activation_energy": hot_activation_energy,
        "cold_activation_energy": cold_activation_energy,
        "hot_fragility_index": hot_fragility_index,
        "cold_fragility_index": cold_fragility_index,
        "adam_gibbs_slowdown": adam_gibbs_slowdown,
        "min_activation_growth": min_activation_growth,
        "min_fragility_growth": min_fragility_growth,
        "min_adam_gibbs_slowdown": min_adam_gibbs_slowdown,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")

    activation_growth = cold_activation_energy / hot_activation_energy
    fragility_growth = cold_fragility_index / hot_fragility_index
    activation_flag = activation_growth >= min_activation_growth
    fragility_flag = fragility_growth >= min_fragility_growth
    adam_gibbs_flag = adam_gibbs_slowdown >= min_adam_gibbs_slowdown
    model_fragility_flag = activation_flag and fragility_flag
    scope_boundary_flag = not material_specific_origin_claimed
    return {
        "benchmark_id": benchmark_id,
        "observed_fragility_growth": float(observed_fragility_growth),
        "observed_adam_gibbs_slowdown": float(observed_adam_gibbs_slowdown),
        "hot_activation_energy": hot_activation_energy,
        "cold_activation_energy": cold_activation_energy,
        "activation_energy_growth": activation_growth,
        "hot_fragility_index": hot_fragility_index,
        "cold_fragility_index": cold_fragility_index,
        "fragility_index_growth": fragility_growth,
        "adam_gibbs_slowdown": adam_gibbs_slowdown,
        "min_activation_growth": min_activation_growth,
        "min_fragility_growth": min_fragility_growth,
        "min_adam_gibbs_slowdown": min_adam_gibbs_slowdown,
        "material_specific_origin_claimed": float(material_specific_origin_claimed),
        "model_predicts_fragility_growth": float(model_fragility_flag),
        "model_predicts_adam_gibbs_slowdown": float(adam_gibbs_flag),
        "activation_growth_consistent": float(activation_flag == observed_fragility_growth),
        "fragility_index_consistent": float(fragility_flag == observed_fragility_growth),
        "adam_gibbs_slowdown_consistent": float(adam_gibbs_flag == observed_adam_gibbs_slowdown),
        "fragility_scope_boundary_consistent": float(scope_boundary_flag),
        "overall_consistent": float(
            model_fragility_flag == observed_fragility_growth
            and adam_gibbs_flag == observed_adam_gibbs_slowdown
            and scope_boundary_flag
        ),
    }


def thermodynamic_scope_benchmark_consistency(
    *,
    benchmark_id: str,
    observed_heat_capacity_anomaly: bool,
    observed_kauzmann_extrapolation: bool,
    dynamic_model_derives_entropy: bool,
    entropy_closure_supplied: bool,
    adam_gibbs_slowdown: float,
    min_adam_gibbs_slowdown: float,
    material_specific_entropy_origin_claimed: bool,
) -> dict[str, float | str]:
    """Check the thermodynamic-transition boundary of the kinetic theory."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    for name, value in {
        "adam_gibbs_slowdown": adam_gibbs_slowdown,
        "min_adam_gibbs_slowdown": min_adam_gibbs_slowdown,
    }.items():
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be positive and finite")

    predicts_heat_capacity_from_dynamics = bool(dynamic_model_derives_entropy)
    predicts_kauzmann_from_dynamics = bool(dynamic_model_derives_entropy)
    entropy_closure_required = not dynamic_model_derives_entropy
    adam_gibbs_flag = bool(entropy_closure_supplied) and adam_gibbs_slowdown >= min_adam_gibbs_slowdown
    heat_capacity_scope_flag = (
        (not observed_heat_capacity_anomaly) or not predicts_heat_capacity_from_dynamics
    )
    kauzmann_scope_flag = (
        (not observed_kauzmann_extrapolation) or not predicts_kauzmann_from_dynamics
    )
    boundary_flag = entropy_closure_required and not material_specific_entropy_origin_claimed
    overall = heat_capacity_scope_flag and kauzmann_scope_flag and adam_gibbs_flag and boundary_flag

    return {
        "benchmark_id": benchmark_id,
        "observed_heat_capacity_anomaly": float(observed_heat_capacity_anomaly),
        "observed_kauzmann_extrapolation": float(observed_kauzmann_extrapolation),
        "dynamic_model_derives_entropy": float(dynamic_model_derives_entropy),
        "entropy_closure_supplied": float(entropy_closure_supplied),
        "thermodynamic_adam_gibbs_slowdown": adam_gibbs_slowdown,
        "min_thermodynamic_adam_gibbs_slowdown": min_adam_gibbs_slowdown,
        "material_specific_entropy_origin_claimed": float(material_specific_entropy_origin_claimed),
        "model_predicts_heat_capacity_anomaly_from_dynamics": float(
            predicts_heat_capacity_from_dynamics
        ),
        "model_predicts_kauzmann_transition_from_dynamics": float(predicts_kauzmann_from_dynamics),
        "entropy_closure_required": float(entropy_closure_required),
        "adam_gibbs_slowdown_consistent": float(adam_gibbs_flag),
        "heat_capacity_scope_consistent": float(heat_capacity_scope_flag),
        "kauzmann_scope_consistent": float(kauzmann_scope_flag),
        "thermodynamic_scope_boundary_consistent": float(boundary_flag),
        "overall_consistent": float(overall),
    }


def temperature_dependent_gamma_exchange(
    temperature: float,
    law: FacilitatedExchangeLawParams,
) -> GammaExchangeParams:
    """Map reduced temperature to finite-exchange heterogeneity parameters."""

    _validate_facilitated_exchange_law(law)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    inverse_temperature_shift = 1.0 / temperature - 1.0 / law.reference_temperature
    return GammaExchangeParams(
        shape=law.shape_ref * math.exp(-law.shape_broadening_barrier * inverse_temperature_shift),
        exchange_renewal_count=law.exchange_renewal_count_ref
        * math.exp(law.exchange_slowing_barrier * inverse_temperature_shift),
    )


def local_cage_variance(t: np.ndarray, params: DelayedRenewalCageParams) -> np.ndarray:
    """Gaussian intra-cage displacement variance."""

    _validate(params)
    t = np.asarray(t, dtype=float)
    if np.any(t < 0.0):
        raise ValueError("time values must be nonnegative")
    return params.cage_variance * (1.0 - np.exp(-t / params.cage_tau))


def delayed_poisson_mean(t: np.ndarray, params: DelayedRenewalCageParams) -> np.ndarray:
    """Mean number of delayed renewal events by time t.

    The renewal intensity is lambda(t) = lambda_inf (1 - exp(-t/tau_d))^2.
    Its time integral is analytic.
    """

    _validate(params)
    t = np.asarray(t, dtype=float)
    if np.any(t < 0.0):
        raise ValueError("time values must be nonnegative")

    tau = params.renewal_delay
    x = t / tau
    direct = params.renewal_rate * (
        t
        - 2.0 * tau * (1.0 - np.exp(-t / tau))
        + 0.5 * tau * (1.0 - np.exp(-2.0 * t / tau))
    )
    series = params.renewal_rate * tau * (
        (x**3) / 3.0 - (x**4) / 4.0 + 7.0 * (x**5) / 60.0
    )
    return np.where(x < 1e-3, series, direct)


def moments_1d(t: np.ndarray, params: DelayedRenewalCageParams) -> dict[str, np.ndarray]:
    """Return second and fourth moments for the 1D delayed renewal cage model.

    Conditional on the renewal count N(t), displacement is Gaussian with variance

        V(t | N) = local_variance(t) + N * jump_variance.

    With Poisson renewal count R(t), E[V] = L + qR and
    Var[V] = q^2 R. Since E[x^4 | V] = 3V^2, E[x^4] = 3E[V^2].
    """

    local = local_cage_variance(t, params)
    renewal = delayed_poisson_mean(t, params)
    mean_variance = local + params.jump_variance * renewal
    variance_of_variance = (params.jump_variance**2) * renewal
    fourth = 3.0 * (mean_variance**2 + variance_of_variance)
    return {
        "time": np.asarray(t, dtype=float),
        "local_variance": local,
        "renewal_mean": renewal,
        "m2": mean_variance,
        "m4": fourth,
    }


def ngp_1d(t: np.ndarray, params: DelayedRenewalCageParams) -> np.ndarray:
    """One-dimensional non-Gaussian parameter."""

    moments = moments_1d(t, params)
    m2 = moments["m2"]
    m4 = moments["m4"]
    out = np.zeros_like(m2)
    mask = m2 > 0.0
    out[mask] = m4[mask] / (3.0 * m2[mask] ** 2) - 1.0
    return out


def moments_3d(t: np.ndarray, params: DelayedRenewalCageParams) -> dict[str, np.ndarray]:
    """Return isotropic 3D radial moments.

    The scalar variance in moments_1d is the per-coordinate variance. Conditional on
    that variance V, an isotropic 3D Gaussian has <r^2|V> = 3V and <r^4|V> = 15V^2.
    """

    moments = moments_1d(t, params)
    m2_1d = moments["m2"]
    # E[V^2] = m4_1d / 3.
    variance_square_mean = moments["m4"] / 3.0
    return {
        "time": moments["time"],
        "r2": 3.0 * m2_1d,
        "r4": 15.0 * variance_square_mean,
    }


def ngp_3d(t: np.ndarray, params: DelayedRenewalCageParams) -> np.ndarray:
    """Return the standard 3D NGP: 3<r^4>/(5<r^2>^2)-1."""

    moments = moments_3d(t, params)
    r2 = moments["r2"]
    r4 = moments["r4"]
    out = np.zeros_like(r2)
    mask = r2 > 0.0
    out[mask] = 3.0 * r4[mask] / (5.0 * r2[mask] ** 2) - 1.0
    return out


def _persistence_exchange_poisson_row(mean: float, max_count: int) -> np.ndarray:
    row = np.zeros(max_count + 1, dtype=float)
    row[0] = math.exp(-mean)
    for count in range(1, max_count):
        row[count] = row[count - 1] * mean / count
    row[max_count] = max(0.0, 1.0 - float(np.sum(row[:max_count])))
    return row


def _persistence_exchange_row(
    time: float,
    first_rate: float,
    exchange_rate: float,
    max_count: int,
) -> np.ndarray:
    if time == 0.0:
        row = np.zeros(max_count + 1, dtype=float)
        row[0] = 1.0
        return row
    grid_size = max(800, int(math.ceil(30.0 * time)))
    elapsed_after_first = np.linspace(0.0, time, grid_size)
    first_density = first_rate * np.exp(-first_rate * (time - elapsed_after_first))
    exchange_mean = exchange_rate * elapsed_after_first
    poisson = np.exp(-exchange_mean)
    cumulative = poisson.copy()
    row = np.zeros(max_count + 1, dtype=float)
    row[0] = math.exp(-first_rate * time)
    row[1] = float(np.trapezoid(first_density * poisson, elapsed_after_first))
    for count in range(2, max_count):
        poisson = poisson * exchange_mean / (count - 1)
        cumulative += poisson
        row[count] = float(np.trapezoid(first_density * poisson, elapsed_after_first))
    tail = np.maximum(0.0, 1.0 - cumulative)
    row[max_count] = float(np.trapezoid(first_density * tail, elapsed_after_first))
    total = float(np.sum(row))
    if total <= 0.0:
        raise RuntimeError("persistence-exchange count distribution failed to normalize")
    return row / total


def persistence_exchange_count_distribution(
    t: np.ndarray,
    params: PersistenceExchangeParams,
    *,
    max_count: int = 300,
) -> np.ndarray:
    """Count law for a process with decoupled first-persistence and exchange clocks."""

    _validate_persistence_exchange(params)
    if max_count < 2:
        raise ValueError("max_count must be at least two")
    t = np.asarray(t, dtype=float)
    if np.any(t < 0.0):
        raise ValueError("time values must be nonnegative")
    first_rate = 1.0 / params.persistence_mean
    exchange_rate = 1.0 / params.exchange_mean
    rows = np.zeros((len(t), max_count + 1), dtype=float)
    for index, time in enumerate(t):
        if math.isclose(first_rate, exchange_rate, rel_tol=1e-13, abs_tol=1e-13):
            rows[index] = _persistence_exchange_poisson_row(first_rate * float(time), max_count)
        else:
            rows[index] = _persistence_exchange_row(float(time), first_rate, exchange_rate, max_count)
    return rows


def persistence_exchange_count_pgf(
    z: float,
    t: np.ndarray,
    params: PersistenceExchangeParams,
) -> np.ndarray:
    """Probability-generating function for the persistence/exchange count.

    For first-escape rate ``a=1/tau_p`` and exchange rate ``b=1/tau_x``,
    ``G(z,t)=E[z^N]`` is

        exp(-a t) + z a [exp(b(z-1)t)-exp(-a t)] / [a+b(z-1)].

    The removable singularity is evaluated by its integral limit.
    """

    _validate_persistence_exchange(params)
    if z < 0.0:
        raise ValueError("z must be nonnegative")
    t = np.asarray(t, dtype=float)
    if np.any(t < 0.0):
        raise ValueError("time values must be nonnegative")
    first_rate = 1.0 / params.persistence_mean
    exchange_rate = 1.0 / params.exchange_mean
    denominator = first_rate + exchange_rate * (z - 1.0)
    unrenewed = np.exp(-first_rate * t)
    if math.isclose(denominator, 0.0, rel_tol=1e-13, abs_tol=1e-13):
        renewed = z * first_rate * t * unrenewed
    else:
        renewed = z * first_rate * (np.exp(exchange_rate * (z - 1.0) * t) - unrenewed) / denominator
    return unrenewed + renewed


def persistence_exchange_count_moments(
    t: np.ndarray,
    params: PersistenceExchangeParams,
    *,
    max_count: int = 300,
) -> dict[str, np.ndarray]:
    """Closed mean and variance of the decoupled persistence/exchange renewal count."""

    _validate_persistence_exchange(params)
    t = np.asarray(t, dtype=float)
    if np.any(t < 0.0):
        raise ValueError("time values must be nonnegative")
    first_rate = 1.0 / params.persistence_mean
    exchange_rate = 1.0 / params.exchange_mean
    if math.isclose(first_rate, exchange_rate, rel_tol=1e-13, abs_tol=1e-13):
        mean = first_rate * t
        return {"mean": mean, "variance": mean}

    unrenewed = np.exp(-first_rate * t)
    renewed_probability = 1.0 - unrenewed
    first_elapsed_mean = t - renewed_probability / first_rate
    first_elapsed_second = t**2 - 2.0 * t / first_rate + 2.0 * renewed_probability / first_rate**2
    mean = renewed_probability + exchange_rate * first_elapsed_mean
    second = renewed_probability + 3.0 * exchange_rate * first_elapsed_mean + exchange_rate**2 * first_elapsed_second
    variance = second - mean**2
    return {"mean": mean, "variance": np.maximum(variance, 0.0)}


def persistence_exchange_count_moments_from_distribution(
    t: np.ndarray,
    params: PersistenceExchangeParams,
    *,
    max_count: int = 300,
) -> dict[str, np.ndarray]:
    """Numerical count moments from the explicit distribution, used as a check."""

    probability = persistence_exchange_count_distribution(t, params, max_count=max_count)
    counts = np.arange(probability.shape[1], dtype=float)
    mean = probability @ counts
    second = probability @ (counts**2)
    return {"mean": mean, "variance": second - mean**2}



def persistence_exchange_diffusion_coefficient(params: PersistenceExchangeParams) -> float:
    """Long-time one-dimensional diffusion coefficient set by exchange events."""

    _validate_persistence_exchange(params)
    return params.jump_variance / (2.0 * params.exchange_mean)


def persistence_exchange_msd(
    t: np.ndarray,
    params: PersistenceExchangeParams,
    *,
    max_count: int = 300,
) -> np.ndarray:
    """MSD for the persistence/exchange renewal cage."""

    _validate_persistence_exchange(params)
    t = np.asarray(t, dtype=float)
    moments = persistence_exchange_count_moments(t, params, max_count=max_count)
    local = params.cage_variance * (1.0 - np.exp(-t / params.cage_tau))
    return local + params.jump_variance * moments["mean"]


def persistence_exchange_ngp_1d(
    t: np.ndarray,
    params: PersistenceExchangeParams,
    *,
    max_count: int = 300,
) -> np.ndarray:
    """One-dimensional NGP for the persistence/exchange renewal count."""

    _validate_persistence_exchange(params)
    t = np.asarray(t, dtype=float)
    moments = persistence_exchange_count_moments(t, params)
    local = params.cage_variance * (1.0 - np.exp(-t / params.cage_tau))
    mean_variance = local + params.jump_variance * moments["mean"]
    variance_of_variance = params.jump_variance**2 * moments["variance"]
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = variance_of_variance / mean_variance**2
    alpha[~np.isfinite(alpha)] = 0.0
    return alpha


def persistence_exchange_normalized_alpha_decay(
    wave_number: float,
    t: np.ndarray,
    params: PersistenceExchangeParams,
    *,
    max_count: int = 300,
) -> np.ndarray:
    """Cage-normalized self-intermediate scattering decay."""

    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    jump_factor = math.exp(-0.5 * wave_number**2 * params.jump_variance)
    return persistence_exchange_count_pgf(jump_factor, t, params)


def persistence_exchange_scattering_susceptibility(
    wave_number: float,
    t: np.ndarray,
    params: PersistenceExchangeParams,
) -> np.ndarray:
    """Single-domain alpha susceptibility proxy for persistence/exchange clocks."""

    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    jump_factor = math.exp(-0.5 * wave_number**2 * params.jump_variance)
    mean = persistence_exchange_count_pgf(jump_factor, t, params)
    second = persistence_exchange_count_pgf(jump_factor * jump_factor, t, params)
    return np.maximum(second - mean * mean, 0.0)


def persistence_exchange_alpha_relaxation_time(
    wave_number: float,
    params: PersistenceExchangeParams,
    *,
    threshold: float = math.exp(-1.0),
    max_count: int = 600,
) -> float:
    """Solve the alpha time after removing the local cage Debye-Waller factor."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    lower = 0.0
    upper = max(params.persistence_mean, params.exchange_mean)
    while persistence_exchange_normalized_alpha_decay(wave_number, np.array([upper]), params, max_count=max_count)[0] > threshold:
        upper *= 2.0
        if upper > 1.0e7:
            raise RuntimeError("alpha relaxation time bracket failed")
    for _ in range(70):
        mid = 0.5 * (lower + upper)
        value = persistence_exchange_normalized_alpha_decay(wave_number, np.array([mid]), params, max_count=max_count)[0]
        if value > threshold:
            lower = mid
        else:
            upper = mid
    return 0.5 * (lower + upper)


def _translation_component(params: TranslationRotationExchangeParams) -> PersistenceExchangeParams:
    return PersistenceExchangeParams(
        cage_variance=params.cage_variance,
        cage_tau=params.cage_tau,
        jump_variance=params.jump_variance,
        persistence_mean=params.translational_persistence_mean,
        exchange_mean=params.translational_exchange_mean,
    )


def _rotation_component(params: TranslationRotationExchangeParams) -> PersistenceExchangeParams:
    return PersistenceExchangeParams(
        cage_variance=1.0,
        cage_tau=1.0,
        jump_variance=1.0,
        persistence_mean=params.rotational_persistence_mean,
        exchange_mean=params.rotational_exchange_mean,
    )


def translation_rotation_rotational_correlation(
    t: np.ndarray,
    params: TranslationRotationExchangeParams,
) -> np.ndarray:
    """Normalized orientational correlation for a rotational renewal clock."""

    _validate_translation_rotation(params)
    return persistence_exchange_count_pgf(params.rotational_step_correlation, t, _rotation_component(params))


def translation_rotation_rotational_relaxation_time(
    params: TranslationRotationExchangeParams,
    *,
    threshold: float = math.exp(-1.0),
) -> float:
    """Solve the rotational alpha time from the renewal orientational correlation."""

    _validate_translation_rotation(params)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    lower = 0.0
    upper = max(params.rotational_persistence_mean, params.rotational_exchange_mean)
    while translation_rotation_rotational_correlation(np.array([upper]), params)[0] > threshold:
        upper *= 2.0
        if upper > 1.0e7:
            raise RuntimeError("rotational relaxation time bracket failed")
    for _ in range(70):
        mid = 0.5 * (lower + upper)
        if translation_rotation_rotational_correlation(np.array([mid]), params)[0] > threshold:
            lower = mid
        else:
            upper = mid
    return 0.5 * (lower + upper)


def translation_rotation_decoupling_diagnostic(
    scenario: str,
    params: TranslationRotationExchangeParams,
    *,
    wave_number: float,
    threshold: float = math.exp(-1.0),
    min_decoupling_ratio: float = 1.5,
) -> dict[str, float | str]:
    """Compare translational and rotational relaxation clocks at fixed diffusion."""

    if not scenario:
        raise ValueError("scenario must be nonempty")
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if min_decoupling_ratio <= 1.0:
        raise ValueError("min_decoupling_ratio must exceed one")
    _validate_translation_rotation(params)
    translation = _translation_component(params)
    tau_alpha = persistence_exchange_alpha_relaxation_time(wave_number, translation, threshold=threshold)
    tau_rot = translation_rotation_rotational_relaxation_time(params, threshold=threshold)
    diffusion = persistence_exchange_diffusion_coefficient(translation)
    ratio = tau_rot / tau_alpha
    translational_clock_ratio = params.translational_persistence_mean / params.translational_exchange_mean
    rotational_clock_ratio = params.rotational_persistence_mean / params.rotational_exchange_mean
    persistence_ratio = params.rotational_persistence_mean / params.translational_persistence_mean
    decoupled = ratio >= min_decoupling_ratio and persistence_ratio >= min_decoupling_ratio

    return {
        "scenario": scenario,
        "wave_number": wave_number,
        "diffusion_coefficient": diffusion,
        "tau_alpha": tau_alpha,
        "rotational_relaxation_time": tau_rot,
        "translational_persistence_mean": params.translational_persistence_mean,
        "translational_exchange_mean": params.translational_exchange_mean,
        "rotational_persistence_mean": params.rotational_persistence_mean,
        "rotational_exchange_mean": params.rotational_exchange_mean,
        "rotational_step_correlation": params.rotational_step_correlation,
        "translational_clock_ratio": translational_clock_ratio,
        "rotational_clock_ratio": rotational_clock_ratio,
        "rotational_to_translational_persistence_ratio": persistence_ratio,
        "translation_rotation_ratio": ratio,
        "stokes_einstein_product": diffusion * tau_alpha,
        "rotational_dse_product": diffusion * tau_rot,
        "translation_rotation_decoupling_detected": float(decoupled),
    }


def translation_rotation_inversion_protocol(
    *,
    benchmark_id: str,
    wave_number: float,
    jump_variance: float,
    diffusion_coefficient: float,
    observed_tau_alpha: float,
    observed_rotational_relaxation_time: float,
    rotational_step_correlation: float,
    rotational_exchange_mean: float,
    cage_variance: float = 1.0,
    cage_tau: float = 0.2,
    threshold: float = math.exp(-1.0),
    min_decoupling_ratio: float = 1.5,
) -> dict[str, float | str]:
    """Infer translational and rotational persistence clocks from transport clocks."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    if observed_rotational_relaxation_time <= 0.0:
        raise ValueError("observed_rotational_relaxation_time must be positive")
    if not 0.0 < rotational_step_correlation < 1.0:
        raise ValueError("rotational_step_correlation must lie between zero and one")
    if rotational_exchange_mean <= 0.0:
        raise ValueError("rotational_exchange_mean must be positive")

    translation = infer_persistence_exchange_from_alpha_transport(
        wave_number=wave_number,
        jump_variance=jump_variance,
        diffusion_coefficient=diffusion_coefficient,
        observed_tau_alpha=observed_tau_alpha,
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        threshold=threshold,
    )

    def rotational_tau(persistence_mean: float) -> float:
        params = TranslationRotationExchangeParams(
            cage_variance=cage_variance,
            cage_tau=cage_tau,
            jump_variance=jump_variance,
            translational_persistence_mean=translation["persistence_mean"],
            translational_exchange_mean=translation["exchange_mean"],
            rotational_persistence_mean=persistence_mean,
            rotational_exchange_mean=rotational_exchange_mean,
            rotational_step_correlation=rotational_step_correlation,
        )
        return translation_rotation_rotational_relaxation_time(params, threshold=threshold)

    poisson_rot_tau = rotational_tau(rotational_exchange_mean)
    if observed_rotational_relaxation_time < poisson_rot_tau and not math.isclose(
        observed_rotational_relaxation_time,
        poisson_rot_tau,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError("observed_rotational_relaxation_time is faster than the rotational Poisson baseline")

    lower = rotational_exchange_mean
    if math.isclose(observed_rotational_relaxation_time, poisson_rot_tau, rel_tol=1e-12, abs_tol=1e-12):
        rotational_persistence_mean = lower
    else:
        upper = max(2.0 * lower, observed_rotational_relaxation_time, lower + 1.0e-12)
        while rotational_tau(upper) < observed_rotational_relaxation_time:
            upper *= 2.0
            if upper > 1.0e10:
                raise RuntimeError("rotational persistence inversion bracket failed")
        for _ in range(80):
            mid = 0.5 * (lower + upper)
            if rotational_tau(mid) < observed_rotational_relaxation_time:
                lower = mid
            else:
                upper = mid
        rotational_persistence_mean = 0.5 * (lower + upper)

    params = TranslationRotationExchangeParams(
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        jump_variance=jump_variance,
        translational_persistence_mean=translation["persistence_mean"],
        translational_exchange_mean=translation["exchange_mean"],
        rotational_persistence_mean=rotational_persistence_mean,
        rotational_exchange_mean=rotational_exchange_mean,
        rotational_step_correlation=rotational_step_correlation,
    )
    row = translation_rotation_decoupling_diagnostic(
        benchmark_id,
        params,
        wave_number=wave_number,
        threshold=threshold,
        min_decoupling_ratio=min_decoupling_ratio,
    )
    row.update(
        {
            "benchmark_id": benchmark_id,
            "observed_tau_alpha": observed_tau_alpha,
            "observed_rotational_relaxation_time": observed_rotational_relaxation_time,
            "tau_alpha_log_residual": math.log(row["tau_alpha"] / observed_tau_alpha),
            "rotational_tau_log_residual": math.log(
                row["rotational_relaxation_time"] / observed_rotational_relaxation_time
            ),
            "poisson_rotational_relaxation_time": poisson_rot_tau,
        }
    )
    return row


def infer_persistence_exchange_from_alpha_transport(
    *,
    wave_number: float,
    jump_variance: float,
    diffusion_coefficient: float,
    observed_tau_alpha: float,
    cage_variance: float = 1.0,
    cage_tau: float = 0.2,
    threshold: float = math.exp(-1.0),
    late_time: float | None = None,
    observed_late_ngp: float | None = None,
) -> dict[str, float]:
    """Infer persistence/exchange clocks from transport and alpha relaxation.

    Long-time diffusion fixes the post-escape exchange mean,
    ``tau_x=q/(2D)``. If the observed alpha time is slower than the Poisson
    ``tau_p=tau_x`` baseline, the first-persistence mean is the unique
    ``tau_p>=tau_x`` for which the cage-normalized alpha decay reaches the
    requested threshold at ``observed_tau_alpha``. A late NGP value, when
    supplied, is a held-out falsification observable rather than a fitted input.
    """

    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if jump_variance <= 0.0:
        raise ValueError("jump_variance must be positive")
    if diffusion_coefficient <= 0.0:
        raise ValueError("diffusion_coefficient must be positive")
    if observed_tau_alpha <= 0.0:
        raise ValueError("observed_tau_alpha must be positive")
    if cage_variance <= 0.0 or cage_tau <= 0.0:
        raise ValueError("cage parameters must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    if late_time is not None and late_time <= 0.0:
        raise ValueError("late_time must be positive")
    if observed_late_ngp is not None and observed_late_ngp <= 0.0:
        raise ValueError("observed_late_ngp must be positive")

    exchange_mean = jump_variance / (2.0 * diffusion_coefficient)
    poisson_params = PersistenceExchangeParams(
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        jump_variance=jump_variance,
        persistence_mean=exchange_mean,
        exchange_mean=exchange_mean,
    )
    poisson_tau_alpha = persistence_exchange_alpha_relaxation_time(
        wave_number,
        poisson_params,
        threshold=threshold,
    )
    if observed_tau_alpha < poisson_tau_alpha and not math.isclose(
        observed_tau_alpha,
        poisson_tau_alpha,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError("observed_tau_alpha is faster than the Poisson exchange baseline")

    def residual(persistence_mean: float) -> float:
        params = PersistenceExchangeParams(
            cage_variance=cage_variance,
            cage_tau=cage_tau,
            jump_variance=jump_variance,
            persistence_mean=persistence_mean,
            exchange_mean=exchange_mean,
        )
        return float(
            persistence_exchange_normalized_alpha_decay(
                wave_number,
                np.array([observed_tau_alpha]),
                params,
            )[0]
            - threshold
        )

    lower = exchange_mean
    lower_residual = residual(lower)
    if math.isclose(lower_residual, 0.0, rel_tol=1e-12, abs_tol=1e-12):
        persistence_mean = lower
    else:
        upper = max(2.0 * lower, observed_tau_alpha, lower + 1.0e-12)
        upper_residual = residual(upper)
        while upper_residual < 0.0:
            upper *= 2.0
            upper_residual = residual(upper)
            if upper > 1.0e10:
                raise RuntimeError("persistence/exchange inversion bracket failed")
        for _ in range(80):
            mid = 0.5 * (lower + upper)
            if residual(mid) < 0.0:
                lower = mid
            else:
                upper = mid
        persistence_mean = 0.5 * (lower + upper)

    inferred_params = PersistenceExchangeParams(
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        jump_variance=jump_variance,
        persistence_mean=persistence_mean,
        exchange_mean=exchange_mean,
    )
    reconstructed_tau_alpha = persistence_exchange_alpha_relaxation_time(
        wave_number,
        inferred_params,
        threshold=threshold,
    )
    predicted_late_ngp = float("nan")
    late_ngp_log_residual = float("nan")
    if late_time is not None:
        predicted_late_ngp = float(persistence_exchange_ngp_1d(np.array([late_time]), inferred_params)[0])
        if observed_late_ngp is not None:
            late_ngp_log_residual = math.log(observed_late_ngp / predicted_late_ngp)

    return {
        "exchange_mean": exchange_mean,
        "persistence_mean": persistence_mean,
        "persistence_exchange_ratio": persistence_mean / exchange_mean,
        "poisson_tau_alpha": poisson_tau_alpha,
        "reconstructed_tau_alpha": reconstructed_tau_alpha,
        "tau_alpha_log_residual": math.log(reconstructed_tau_alpha / observed_tau_alpha),
        "diffusion_coefficient": diffusion_coefficient,
        "stokes_einstein_product": diffusion_coefficient * observed_tau_alpha,
        "late_time": float(late_time) if late_time is not None else float("nan"),
        "predicted_late_ngp": predicted_late_ngp,
        "late_ngp_log_residual": late_ngp_log_residual,
    }


def persistence_exchange_joint_diagnostic(
    *,
    anchor_wave_number: float,
    wave_numbers: list[float],
    observed_tau_alpha_by_k: dict[float, float],
    jump_variance: float,
    diffusion_coefficient: float,
    late_time: float,
    observed_late_ngp: float,
    time_grid: np.ndarray,
    cage_variance: float = 1.0,
    cage_tau: float = 0.2,
    threshold: float = math.exp(-1.0),
    max_multik_abs_log_residual: float = 0.05,
    max_late_ngp_abs_log_residual: float = 0.1,
    min_chi4_peak_growth: float = 1.0,
) -> dict[str, float]:
    """Jointly test persistence/exchange inversion against held-out observables.

    The anchor alpha time and diffusion infer ``tau_p`` and ``tau_x``. Other
    wave numbers, the late NGP, and the alpha-susceptibility proxy are then
    predictions rather than fitted quantities.
    """

    if anchor_wave_number <= 0.0:
        raise ValueError("anchor_wave_number must be positive")
    if not wave_numbers:
        raise ValueError("wave_numbers must be nonempty")
    if anchor_wave_number not in observed_tau_alpha_by_k:
        raise ValueError("observed_tau_alpha_by_k must include the anchor wave number")
    for wave_number in wave_numbers:
        if wave_number <= 0.0:
            raise ValueError("wave_numbers must be positive")
        if wave_number not in observed_tau_alpha_by_k:
            raise ValueError("observed_tau_alpha_by_k must include every wave number")
        if observed_tau_alpha_by_k[wave_number] <= 0.0:
            raise ValueError("observed alpha times must be positive")
    time_grid = np.asarray(time_grid, dtype=float)
    if time_grid.ndim != 1 or time_grid.size == 0 or np.any(time_grid <= 0.0):
        raise ValueError("time_grid must be a positive one-dimensional array")
    if max_multik_abs_log_residual < 0.0 or max_late_ngp_abs_log_residual < 0.0:
        raise ValueError("residual thresholds must be nonnegative")
    if min_chi4_peak_growth <= 0.0:
        raise ValueError("min_chi4_peak_growth must be positive")

    anchor_tau_alpha = observed_tau_alpha_by_k[anchor_wave_number]
    inferred = infer_persistence_exchange_from_alpha_transport(
        wave_number=anchor_wave_number,
        jump_variance=jump_variance,
        diffusion_coefficient=diffusion_coefficient,
        observed_tau_alpha=anchor_tau_alpha,
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        threshold=threshold,
        late_time=late_time,
        observed_late_ngp=observed_late_ngp,
    )
    params = PersistenceExchangeParams(
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        jump_variance=jump_variance,
        persistence_mean=inferred["persistence_mean"],
        exchange_mean=inferred["exchange_mean"],
    )
    poisson_params = PersistenceExchangeParams(
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        jump_variance=jump_variance,
        persistence_mean=inferred["exchange_mean"],
        exchange_mean=inferred["exchange_mean"],
    )

    max_abs_residual = 0.0
    for wave_number in wave_numbers:
        predicted_tau = persistence_exchange_alpha_relaxation_time(
            wave_number,
            params,
            threshold=threshold,
        )
        residual = math.log(observed_tau_alpha_by_k[wave_number] / predicted_tau)
        max_abs_residual = max(max_abs_residual, abs(residual))

    chi4 = persistence_exchange_scattering_susceptibility(anchor_wave_number, time_grid, params)
    poisson_chi4 = persistence_exchange_scattering_susceptibility(anchor_wave_number, time_grid, poisson_params)
    chi4_peak = float(np.max(chi4))
    poisson_chi4_peak = float(np.max(poisson_chi4))
    chi4_growth = chi4_peak / poisson_chi4_peak
    poisson_tau_alpha = persistence_exchange_alpha_relaxation_time(
        anchor_wave_number,
        poisson_params,
        threshold=threshold,
    )
    se_growth = anchor_tau_alpha / poisson_tau_alpha
    multik_flag = max_abs_residual <= max_multik_abs_log_residual
    late_ngp_flag = abs(inferred["late_ngp_log_residual"]) <= max_late_ngp_abs_log_residual
    chi4_flag = chi4_growth >= min_chi4_peak_growth
    return {
        "exchange_mean": inferred["exchange_mean"],
        "persistence_mean": inferred["persistence_mean"],
        "persistence_exchange_ratio": inferred["persistence_exchange_ratio"],
        "anchor_wave_number": anchor_wave_number,
        "anchor_tau_alpha": anchor_tau_alpha,
        "poisson_tau_alpha": poisson_tau_alpha,
        "stokes_einstein_growth_over_poisson": se_growth,
        "max_multik_tau_alpha_abs_log_residual": max_abs_residual,
        "late_time": late_time,
        "observed_late_ngp": observed_late_ngp,
        "predicted_late_ngp": inferred["predicted_late_ngp"],
        "late_ngp_log_residual": inferred["late_ngp_log_residual"],
        "chi4_peak": chi4_peak,
        "poisson_chi4_peak": poisson_chi4_peak,
        "chi4_peak_growth_over_poisson": chi4_growth,
        "multik_tau_alpha_consistent": float(multik_flag),
        "late_ngp_consistent": float(late_ngp_flag),
        "chi4_proxy_growth_consistent": float(chi4_flag),
        "passes_joint_protocol": float(multik_flag and late_ngp_flag and chi4_flag),
    }


def _log_sigma_from_relative_error(relative_error: float, name: str) -> float:
    if relative_error <= 0.0:
        raise ValueError(f"{name} must be positive")
    return math.log1p(relative_error)


def persistence_exchange_data_protocol(
    *,
    anchor_wave_number: float,
    wave_numbers: list[float],
    observed_tau_alpha_by_k: dict[float, float],
    tau_alpha_relative_error_by_k: dict[float, float],
    jump_variance: float,
    diffusion_coefficient: float,
    late_time: float,
    observed_late_ngp: float,
    late_ngp_relative_error: float,
    observed_chi4_peak: float,
    chi4_peak_relative_error: float,
    time_grid: np.ndarray,
    cage_variance: float = 1.0,
    cage_tau: float = 0.2,
    threshold: float = math.exp(-1.0),
    z_threshold: float = 2.0,
) -> dict[str, float]:
    """Score a joint persistence/exchange inversion with measurement errors."""

    if observed_chi4_peak <= 0.0:
        raise ValueError("observed_chi4_peak must be positive")
    if z_threshold <= 0.0:
        raise ValueError("z_threshold must be positive")
    if not wave_numbers:
        raise ValueError("wave_numbers must be nonempty")
    for wave_number in wave_numbers:
        if wave_number not in tau_alpha_relative_error_by_k:
            raise ValueError("tau_alpha_relative_error_by_k must include every wave number")
        _log_sigma_from_relative_error(tau_alpha_relative_error_by_k[wave_number], "tau alpha relative errors")
    late_ngp_sigma = _log_sigma_from_relative_error(late_ngp_relative_error, "late_ngp_relative_error")
    chi4_sigma = _log_sigma_from_relative_error(chi4_peak_relative_error, "chi4_peak_relative_error")

    base = persistence_exchange_joint_diagnostic(
        anchor_wave_number=anchor_wave_number,
        wave_numbers=wave_numbers,
        observed_tau_alpha_by_k=observed_tau_alpha_by_k,
        jump_variance=jump_variance,
        diffusion_coefficient=diffusion_coefficient,
        late_time=late_time,
        observed_late_ngp=observed_late_ngp,
        time_grid=time_grid,
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        threshold=threshold,
    )
    params = PersistenceExchangeParams(
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        jump_variance=jump_variance,
        persistence_mean=base["persistence_mean"],
        exchange_mean=base["exchange_mean"],
    )

    max_tau_z = 0.0
    for wave_number in wave_numbers:
        predicted_tau = persistence_exchange_alpha_relaxation_time(
            wave_number,
            params,
            threshold=threshold,
        )
        residual = math.log(observed_tau_alpha_by_k[wave_number] / predicted_tau)
        sigma = _log_sigma_from_relative_error(
            tau_alpha_relative_error_by_k[wave_number],
            "tau alpha relative errors",
        )
        max_tau_z = max(max_tau_z, abs(residual) / sigma)

    late_ngp_z = abs(base["late_ngp_log_residual"]) / late_ngp_sigma
    chi4_log_residual = math.log(observed_chi4_peak / base["chi4_peak"])
    chi4_z = abs(chi4_log_residual) / chi4_sigma
    multik_flag = max_tau_z <= z_threshold
    late_flag = late_ngp_z <= z_threshold
    chi4_flag = chi4_z <= z_threshold
    out = dict(base)
    out.update(
        {
            "observed_chi4_peak": observed_chi4_peak,
            "predicted_chi4_peak": base["chi4_peak"],
            "chi4_peak_log_residual": chi4_log_residual,
            "max_multik_tau_alpha_z": max_tau_z,
            "late_ngp_z": late_ngp_z,
            "chi4_peak_z": chi4_z,
            "z_threshold": z_threshold,
            "multik_tau_alpha_z_consistent": float(multik_flag),
            "late_ngp_z_consistent": float(late_flag),
            "chi4_peak_z_consistent": float(chi4_flag),
            "passes_uncertainty_protocol": float(multik_flag and late_flag and chi4_flag),
        }
    )
    return out


def simultaneous_dynamical_signature_closure_gate(
    *,
    protocol_id: str,
    scored_row: dict[str, object],
    calibration_observables: Sequence[str],
    heldout_observables: Sequence[str],
    required_consistency_flags: Sequence[str],
    min_stokes_einstein_growth_over_poisson: float = 1.0,
) -> dict[str, float | str]:
    """Classify a minimal persistence/exchange inversion as a held-out closure test.

    This gate is deliberately narrower than a glass-transition theory claim. It
    asks whether diffusion plus one anchor alpha time can support simultaneous
    dynamical predictions: multi-k alpha shape, late NGP recovery, the
    Stokes-Einstein product, and a chi4 proxy. Thermodynamic transition claims
    are always out of scope for this row.
    """

    if not protocol_id:
        raise ValueError("protocol_id must be nonempty")
    if not calibration_observables:
        raise ValueError("calibration_observables must be nonempty")
    if not heldout_observables:
        raise ValueError("heldout_observables must be nonempty")
    if not required_consistency_flags:
        raise ValueError("required_consistency_flags must be nonempty")
    if min_stokes_einstein_growth_over_poisson <= 0.0:
        raise ValueError("min_stokes_einstein_growth_over_poisson must be positive")
    for name, values in {
        "calibration_observables": calibration_observables,
        "heldout_observables": heldout_observables,
        "required_consistency_flags": required_consistency_flags,
    }.items():
        if any(not value for value in values):
            raise ValueError(f"{name} must contain nonempty strings")

    calibration = list(dict.fromkeys(calibration_observables))
    heldout = list(dict.fromkeys(heldout_observables))
    required = list(dict.fromkeys(required_consistency_flags))
    missing_flags = [flag for flag in required if flag not in scored_row]
    se_growth = float(scored_row.get("stokes_einstein_growth_over_poisson", 0.0))

    base: dict[str, float | str] = {
        "protocol_id": protocol_id,
        "calibration_observables": ";".join(calibration),
        "heldout_observables": ";".join(heldout),
        "required_consistency_flags": ";".join(required),
        "calibration_count": float(len(calibration)),
        "heldout_count": float(len(heldout)),
        "required_consistency_count": float(len(required)),
        "stokes_einstein_growth_over_poisson": se_growth,
        "min_stokes_einstein_growth_over_poisson": float(min_stokes_einstein_growth_over_poisson),
        "thermodynamic_claim_allowed": 0.0,
    }

    if missing_flags:
        base.update(
            {
                "simultaneous_closure_ready": 0.0,
                "all_required_dynamical_predictions_pass": 0.0,
                "stokes_einstein_growth_pass": 0.0,
                "primary_blocker": missing_flags[0],
                "closure_stage": "scored_protocol_incomplete",
            }
        )
        return base

    failed_flags = [flag for flag in required if float(scored_row.get(flag, 0.0)) != 1.0]
    se_pass = se_growth >= min_stokes_einstein_growth_over_poisson
    if not se_pass:
        failed_flags.append("stokes_einstein_growth_over_poisson")
    predictions_pass = not failed_flags
    base.update(
        {
            "simultaneous_closure_ready": 1.0,
            "all_required_dynamical_predictions_pass": float(predictions_pass),
            "stokes_einstein_growth_pass": float(se_pass),
            "primary_blocker": "none" if predictions_pass else failed_flags[0],
            "closure_stage": "simultaneous_dynamical_signature_closure_passed"
            if predictions_pass
            else "dynamical_heldout_prediction_failed",
        }
    )
    for flag in required:
        base[flag] = float(scored_row.get(flag, 0.0))
    return base


def persistence_exchange_scan(
    *,
    ratios: list[float],
    exchange_mean: float,
    wave_number: float,
    cage_variance: float = 1.0,
    cage_tau: float = 0.2,
    jump_variance: float = 0.7,
) -> list[dict[str, float]]:
    """Scan first-persistence/exchange decoupling at fixed long-time diffusion."""

    if exchange_mean <= 0.0:
        raise ValueError("exchange_mean must be positive")
    rows: list[dict[str, float]] = []
    for ratio in ratios:
        if ratio <= 0.0:
            raise ValueError("ratios must be positive")
        params = PersistenceExchangeParams(
            cage_variance=cage_variance,
            cage_tau=cage_tau,
            jump_variance=jump_variance,
            persistence_mean=ratio * exchange_mean,
            exchange_mean=exchange_mean,
        )
        tau_alpha = persistence_exchange_alpha_relaxation_time(wave_number, params)
        diffusion = persistence_exchange_diffusion_coefficient(params)
        late_time = 80.0 * params.persistence_mean
        late_ngp = float(persistence_exchange_ngp_1d(np.array([late_time]), params, max_count=1200)[0])
        rows.append(
            {
                "persistence_exchange_ratio": float(ratio),
                "persistence_mean": params.persistence_mean,
                "exchange_mean": params.exchange_mean,
                "diffusion_coefficient": diffusion,
                "tau_alpha": tau_alpha,
                "stokes_einstein_product": diffusion * tau_alpha,
                "late_time": late_time,
                "late_ngp": late_ngp,
            }
        )
    return rows


def self_intermediate_scattering(
    wave_number: float,
    t: np.ndarray,
    params: DelayedRenewalCageParams,
) -> np.ndarray:
    """Self-intermediate scattering function for the renewal cage model.

    For an isotropic Gaussian displacement with per-coordinate variance V,
    F_s(k|V)=exp(-k^2 V/2). Averaging exp(-k^2[L(t)+qN(t)]/2) over the Poisson
    renewal count gives the closed form below.
    """

    _validate(params)
    if wave_number < 0.0:
        raise ValueError("wave_number must be nonnegative")
    t = np.asarray(t, dtype=float)
    local = local_cage_variance(t, params)
    renewal = delayed_poisson_mean(t, params)
    jump_characteristic = math.exp(-0.5 * wave_number**2 * params.jump_variance)
    return np.exp(-0.5 * wave_number**2 * local + renewal * (jump_characteristic - 1.0))


def normalized_alpha_decay(
    wave_number: float,
    t: np.ndarray,
    params: DelayedRenewalCageParams,
) -> np.ndarray:
    """Alpha-relaxation part of F_s after removing the local cage factor."""

    _validate(params)
    if wave_number < 0.0:
        raise ValueError("wave_number must be nonnegative")
    t = np.asarray(t, dtype=float)
    renewal = delayed_poisson_mean(t, params)
    jump_characteristic = math.exp(-0.5 * wave_number**2 * params.jump_variance)
    return np.exp(renewal * (jump_characteristic - 1.0))


def _validate_static_gamma_shape(shape: float) -> None:
    if shape <= 0.0:
        raise ValueError("shape must be positive")


def static_gamma_count_moments(
    t: np.ndarray,
    params: DelayedRenewalCageParams,
    shape: float,
) -> dict[str, np.ndarray]:
    """Mean and variance for a static gamma mixture of renewal rates.

    This is the non-exchanging mobility-disorder null model. A fixed
    trajectory mobility produces negative-binomial count statistics with
    constant gamma shape, so overdispersion grows as ``R(t)^2`` and does not
    self-average away.
    """

    _validate(params)
    _validate_static_gamma_shape(shape)
    renewal = delayed_poisson_mean(t, params)
    return {
        "mean": renewal,
        "variance": renewal + renewal**2 / shape,
        "shape": np.full_like(renewal, shape),
    }


def static_gamma_ngp_1d(
    t: np.ndarray,
    params: DelayedRenewalCageParams,
    shape: float,
) -> np.ndarray:
    """One-dimensional NGP for static gamma renewal-rate disorder."""

    _validate(params)
    _validate_static_gamma_shape(shape)
    local = local_cage_variance(t, params)
    count = static_gamma_count_moments(t, params, shape)
    mean_variance = local + params.jump_variance * count["mean"]
    variance_variance = params.jump_variance**2 * count["variance"]
    out = np.zeros_like(mean_variance)
    mask = mean_variance > 0.0
    out[mask] = variance_variance[mask] / (mean_variance[mask] ** 2)
    return out


def static_gamma_normalized_alpha_decay(
    wave_number: float,
    t: np.ndarray,
    params: DelayedRenewalCageParams,
    shape: float,
) -> np.ndarray:
    """Cage-normalized alpha decay for static gamma renewal-rate disorder."""

    _validate(params)
    _validate_static_gamma_shape(shape)
    if wave_number < 0.0:
        raise ValueError("wave_number must be nonnegative")
    renewal = delayed_poisson_mean(t, params)
    gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
    return (1.0 + gamma * renewal / shape) ** (-shape)


def static_gamma_asymptotic_diagnostics(
    wave_number: float,
    params: DelayedRenewalCageParams,
    shape: float,
) -> dict[str, float]:
    """Late-time predictions of the static gamma mobility-disorder null."""

    _validate(params)
    _validate_static_gamma_shape(shape)
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
    return {
        "wave_number": wave_number,
        "gamma_k": gamma,
        "static_shape": shape,
        "late_ngp_plateau": 1.0 / shape,
        "late_alpha_decay_per_renewal": 0.0,
        "late_alpha_rate": 0.0,
        "alpha_power_law_exponent": shape,
    }


def _gamma_exchange_effective_shape(
    renewal: np.ndarray,
    heterogeneity: GammaExchangeParams,
) -> np.ndarray:
    _validate_gamma_exchange(heterogeneity)
    return heterogeneity.shape * (1.0 + renewal / heterogeneity.exchange_renewal_count)


def gamma_exchange_count_moments(
    t: np.ndarray,
    params: DelayedRenewalCageParams,
    heterogeneity: GammaExchangeParams,
) -> dict[str, np.ndarray]:
    """Mean and variance of a finite-exchange gamma-mixed renewal count.

    A gamma mixture of Poisson renewal counts has negative-binomial
    overdispersion. The effective shape grows with the renewal count, modeling
    exchange between independent mobility environments and restoring
    self-averaging at long times.
    """

    _validate(params)
    _validate_gamma_exchange(heterogeneity)
    renewal = delayed_poisson_mean(t, params)
    effective_shape = _gamma_exchange_effective_shape(renewal, heterogeneity)
    return {
        "mean": renewal,
        "variance": renewal + renewal**2 / effective_shape,
        "effective_shape": effective_shape,
    }


def gamma_exchange_ngp_1d(
    t: np.ndarray,
    params: DelayedRenewalCageParams,
    heterogeneity: GammaExchangeParams,
) -> np.ndarray:
    """One-dimensional NGP with finite-exchange renewal heterogeneity."""

    _validate(params)
    local = local_cage_variance(t, params)
    count = gamma_exchange_count_moments(t, params, heterogeneity)
    mean_variance = local + params.jump_variance * count["mean"]
    variance_variance = params.jump_variance**2 * count["variance"]
    out = np.zeros_like(mean_variance)
    mask = mean_variance > 0.0
    out[mask] = variance_variance[mask] / (mean_variance[mask] ** 2)
    return out


def gamma_exchange_normalized_alpha_decay(
    wave_number: float,
    t: np.ndarray,
    params: DelayedRenewalCageParams,
    heterogeneity: GammaExchangeParams,
) -> np.ndarray:
    """Cage-normalized alpha decay for finite-exchange gamma heterogeneity."""

    _validate(params)
    _validate_gamma_exchange(heterogeneity)
    if wave_number < 0.0:
        raise ValueError("wave_number must be nonnegative")
    renewal = delayed_poisson_mean(t, params)
    effective_shape = _gamma_exchange_effective_shape(renewal, heterogeneity)
    gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
    return (1.0 + gamma * renewal / effective_shape) ** (-effective_shape)


def gamma_exchange_alpha_relaxation_time(
    wave_number: float,
    params: DelayedRenewalCageParams,
    heterogeneity: GammaExchangeParams,
    *,
    threshold: float = math.exp(-1.0),
) -> float:
    """Alpha time for finite-exchange heterogeneity after removing the cage factor."""

    _validate(params)
    _validate_gamma_exchange(heterogeneity)
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie between 0 and 1")

    low = 0.0
    high = alpha_relaxation_time(wave_number, params, threshold=threshold)
    while gamma_exchange_normalized_alpha_decay(
        wave_number,
        np.array([high]),
        params,
        heterogeneity,
    )[0] > threshold:
        high *= 2.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        if gamma_exchange_normalized_alpha_decay(
            wave_number,
            np.array([mid]),
            params,
            heterogeneity,
        )[0] > threshold:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def gamma_exchange_self_intermediate_scattering(
    wave_number: float,
    t: np.ndarray,
    params: DelayedRenewalCageParams,
    heterogeneity: GammaExchangeParams,
) -> np.ndarray:
    """Self-intermediate scattering with gamma-exchange renewal counts."""

    _validate(params)
    if wave_number < 0.0:
        raise ValueError("wave_number must be nonnegative")
    local = local_cage_variance(t, params)
    alpha_decay = gamma_exchange_normalized_alpha_decay(wave_number, t, params, heterogeneity)
    return np.exp(-0.5 * wave_number**2 * local) * alpha_decay


def gamma_exchange_scattering_susceptibility(
    wave_number: float,
    t: np.ndarray,
    params: DelayedRenewalCageParams,
    heterogeneity: GammaExchangeParams,
) -> np.ndarray:
    """Renewal-count scattering variance for gamma-exchange heterogeneity."""

    _validate(params)
    _validate_gamma_exchange(heterogeneity)
    if wave_number < 0.0:
        raise ValueError("wave_number must be nonnegative")
    t = np.asarray(t, dtype=float)
    local = local_cage_variance(t, params)
    renewal = delayed_poisson_mean(t, params)
    effective_shape = _gamma_exchange_effective_shape(renewal, heterogeneity)
    jump_characteristic = math.exp(-0.5 * wave_number**2 * params.jump_variance)
    second_moment = np.exp(-wave_number**2 * local) * (
        1.0 + renewal * (1.0 - jump_characteristic**2) / effective_shape
    ) ** (-effective_shape)
    mean_square = gamma_exchange_self_intermediate_scattering(wave_number, t, params, heterogeneity) ** 2
    return second_moment - mean_square


def infer_gamma_exchange_ratio_from_alpha_rate(
    *,
    gamma_k: float,
    observed_decay_per_renewal: float,
) -> float:
    """Infer ``R_x/kappa_0`` from the late alpha-decay rate per renewal."""

    if gamma_k <= 0.0:
        raise ValueError("gamma_k must be positive")
    if observed_decay_per_renewal <= 0.0:
        raise ValueError("observed_decay_per_renewal must be positive")
    if observed_decay_per_renewal > gamma_k:
        raise ValueError("observed_decay_per_renewal cannot exceed the Poisson limit gamma_k")
    if math.isclose(observed_decay_per_renewal, gamma_k, rel_tol=1e-14, abs_tol=1e-14):
        return 0.0

    def rate_for_ratio(ratio: float) -> float:
        return math.log1p(gamma_k * ratio) / ratio

    low = 0.0
    high = 1.0
    while rate_for_ratio(high) > observed_decay_per_renewal:
        high *= 2.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        if rate_for_ratio(mid) > observed_decay_per_renewal:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def gamma_exchange_asymptotic_diagnostics(
    wave_number: float,
    params: DelayedRenewalCageParams,
    heterogeneity: GammaExchangeParams,
) -> dict[str, float]:
    """Late-time consistency predictions for finite-exchange heterogeneity."""

    _validate(params)
    _validate_gamma_exchange(heterogeneity)
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
    heterogeneity_ratio = heterogeneity.exchange_renewal_count / heterogeneity.shape
    late_alpha_per_renewal = math.log1p(gamma * heterogeneity_ratio) / heterogeneity_ratio
    return {
        "wave_number": wave_number,
        "gamma_k": gamma,
        "heterogeneity_ratio": heterogeneity_ratio,
        "late_ngp_renewal_amplitude": 1.0 + heterogeneity_ratio,
        "late_alpha_decay_per_renewal": late_alpha_per_renewal,
        "late_alpha_rate": params.renewal_rate * late_alpha_per_renewal,
        "poisson_alpha_decay_per_renewal": gamma,
        "poisson_alpha_rate": params.renewal_rate * gamma,
        "alpha_rate_renormalization": late_alpha_per_renewal / gamma,
        "static_gamma_late_ngp_plateau": 1.0 / heterogeneity.shape,
    }


def gamma_exchange_diagnostic_map(
    *,
    wave_number: float,
    params: DelayedRenewalCageParams,
    shape: float,
    heterogeneity_ratios: list[float],
    minimum_late_ngp_amplitude: float = 3.0,
    maximum_alpha_rate_renormalization: float = 0.75,
) -> list[dict[str, float]]:
    """Closed map of finite-exchange observability as ``c=R_x/kappa_0`` varies."""

    _validate(params)
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if shape <= 0.0:
        raise ValueError("shape must be positive")
    if minimum_late_ngp_amplitude <= 1.0:
        raise ValueError("minimum_late_ngp_amplitude must exceed the Poisson value")
    if not 0.0 < maximum_alpha_rate_renormalization < 1.0:
        raise ValueError("maximum_alpha_rate_renormalization must lie between 0 and 1")

    gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
    rows: list[dict[str, float]] = []
    for ratio in heterogeneity_ratios:
        if ratio < 0.0:
            raise ValueError("heterogeneity ratios must be nonnegative")
        if ratio == 0.0:
            late_alpha_per_renewal = gamma
            inferred_ratio_from_alpha_rate = 0.0
            static_plateau = 1.0 / shape
        else:
            heterogeneity = GammaExchangeParams(shape=shape, exchange_renewal_count=shape * ratio)
            diagnostics = gamma_exchange_asymptotic_diagnostics(wave_number, params, heterogeneity)
            late_alpha_per_renewal = diagnostics["late_alpha_decay_per_renewal"]
            inferred_ratio_from_alpha_rate = infer_gamma_exchange_ratio_from_alpha_rate(
                gamma_k=gamma,
                observed_decay_per_renewal=late_alpha_per_renewal,
            )
            static_plateau = diagnostics["static_gamma_late_ngp_plateau"]
        late_ngp_amplitude = 1.0 + ratio
        alpha_rate_renormalization = late_alpha_per_renewal / gamma
        inferred_ratio_from_late_ngp = late_ngp_amplitude - 1.0
        if ratio == 0.0:
            log_ratio_residual = 0.0
        else:
            log_ratio_residual = math.log(inferred_ratio_from_alpha_rate / ratio)
        passes_joint = (
            late_ngp_amplitude >= minimum_late_ngp_amplitude
            and alpha_rate_renormalization <= maximum_alpha_rate_renormalization
        )
        rows.append(
            {
                "heterogeneity_ratio": ratio,
                "log10_one_plus_ratio": math.log10(1.0 + ratio),
                "gamma_k": gamma,
                "late_ngp_renewal_amplitude": late_ngp_amplitude,
                "late_alpha_decay_per_renewal": late_alpha_per_renewal,
                "late_alpha_rate": params.renewal_rate * late_alpha_per_renewal,
                "poisson_alpha_decay_per_renewal": gamma,
                "poisson_alpha_rate": params.renewal_rate * gamma,
                "alpha_rate_renormalization": alpha_rate_renormalization,
                "static_gamma_late_ngp_plateau": static_plateau,
                "inferred_ratio_from_late_ngp_amplitude": inferred_ratio_from_late_ngp,
                "inferred_ratio_from_alpha_rate": inferred_ratio_from_alpha_rate,
                "log_ratio_residual": log_ratio_residual,
                "passes_joint_criterion": 1.0 if passes_joint else 0.0,
            }
        )
    return rows


def infer_gamma_exchange_from_late_observables(
    *,
    wave_number: float,
    params: DelayedRenewalCageParams,
    late_renewal_count: float,
    late_ngp: float,
    observed_alpha_decay_per_renewal: float,
    residual_tolerance: float = 0.15,
) -> dict[str, float]:
    """Infer finite-exchange consistency from late NGP and alpha-slope data."""

    _validate(params)
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if late_renewal_count <= 0.0:
        raise ValueError("late_renewal_count must be positive")
    if late_ngp <= 0.0:
        raise ValueError("late_ngp must be positive")
    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive")

    gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
    ratio_from_late_ngp = late_renewal_count * late_ngp - 1.0
    if ratio_from_late_ngp < 0.0:
        raise ValueError("late_renewal_count * late_ngp must be at least 1")
    ratio_from_alpha_rate = infer_gamma_exchange_ratio_from_alpha_rate(
        gamma_k=gamma,
        observed_decay_per_renewal=observed_alpha_decay_per_renewal,
    )
    if ratio_from_late_ngp == 0.0 and ratio_from_alpha_rate == 0.0:
        log_ratio_residual = 0.0
    elif ratio_from_late_ngp <= 0.0 or ratio_from_alpha_rate <= 0.0:
        log_ratio_residual = math.inf
    else:
        log_ratio_residual = math.log(ratio_from_alpha_rate / ratio_from_late_ngp)
    passes = math.isfinite(log_ratio_residual) and abs(log_ratio_residual) <= residual_tolerance
    return {
        "wave_number": wave_number,
        "gamma_k": gamma,
        "late_renewal_count": late_renewal_count,
        "late_ngp": late_ngp,
        "late_ngp_renewal_amplitude": late_renewal_count * late_ngp,
        "observed_alpha_decay_per_renewal": observed_alpha_decay_per_renewal,
        "poisson_alpha_decay_per_renewal": gamma,
        "alpha_rate_renormalization": observed_alpha_decay_per_renewal / gamma,
        "ratio_from_late_ngp": ratio_from_late_ngp,
        "ratio_from_alpha_rate": ratio_from_alpha_rate,
        "log_ratio_residual": log_ratio_residual,
        "passes_consistency": 1.0 if passes else 0.0,
        "residual_tolerance": residual_tolerance,
    }


def infer_gamma_exchange_uncertainty_from_late_observables(
    *,
    wave_number: float,
    params: DelayedRenewalCageParams,
    late_renewal_count: float,
    late_ngp: float,
    observed_alpha_decay_per_renewal: float,
    late_renewal_count_std: float,
    late_ngp_std: float,
    alpha_decay_per_renewal_std: float,
    z_threshold: float = 2.0,
) -> dict[str, float]:
    """Propagate late-observable errors into the finite-exchange residual."""

    if late_renewal_count_std < 0.0:
        raise ValueError("late_renewal_count_std must be nonnegative")
    if late_ngp_std < 0.0:
        raise ValueError("late_ngp_std must be nonnegative")
    if alpha_decay_per_renewal_std < 0.0:
        raise ValueError("alpha_decay_per_renewal_std must be nonnegative")
    if z_threshold <= 0.0:
        raise ValueError("z_threshold must be positive")

    inferred = infer_gamma_exchange_from_late_observables(
        wave_number=wave_number,
        params=params,
        late_renewal_count=late_renewal_count,
        late_ngp=late_ngp,
        observed_alpha_decay_per_renewal=observed_alpha_decay_per_renewal,
    )
    ratio_from_late_ngp = inferred["ratio_from_late_ngp"]
    ratio_from_alpha_rate = inferred["ratio_from_alpha_rate"]
    if ratio_from_late_ngp <= 0.0 or ratio_from_alpha_rate <= 0.0:
        raise ValueError("uncertainty propagation requires positive inferred ratios")

    ratio_from_late_ngp_std = math.hypot(
        late_ngp * late_renewal_count_std,
        late_renewal_count * late_ngp_std,
    )
    gamma = inferred["gamma_k"]
    derivative = (
        gamma * ratio_from_alpha_rate / (1.0 + gamma * ratio_from_alpha_rate)
        - math.log1p(gamma * ratio_from_alpha_rate)
    ) / ratio_from_alpha_rate**2
    if derivative == 0.0:
        raise ValueError("alpha-rate derivative vanished")
    ratio_from_alpha_rate_std = alpha_decay_per_renewal_std / abs(derivative)
    log_ratio_residual_std = math.hypot(
        ratio_from_late_ngp_std / ratio_from_late_ngp,
        ratio_from_alpha_rate_std / ratio_from_alpha_rate,
    )
    if log_ratio_residual_std == 0.0:
        z_score = math.inf if inferred["log_ratio_residual"] != 0.0 else 0.0
    else:
        z_score = abs(inferred["log_ratio_residual"]) / log_ratio_residual_std
    inferred.update(
        {
            "late_renewal_count_std": late_renewal_count_std,
            "late_ngp_std": late_ngp_std,
            "alpha_decay_per_renewal_std": alpha_decay_per_renewal_std,
            "ratio_from_late_ngp_std": ratio_from_late_ngp_std,
            "ratio_from_alpha_rate_std": ratio_from_alpha_rate_std,
            "log_ratio_residual_std": log_ratio_residual_std,
            "log_ratio_z_score": z_score,
            "z_threshold": z_threshold,
            "passes_statistical_consistency": 1.0 if z_score <= z_threshold else 0.0,
        }
    )
    return inferred


def infer_gamma_exchange_multik_collapse(
    *,
    wave_numbers: list[float],
    params: DelayedRenewalCageParams,
    late_renewal_count: float,
    late_ngp: float,
    observed_alpha_decay_per_renewal: list[float],
    alpha_decay_per_renewal_std: list[float],
    late_renewal_count_std: float,
    late_ngp_std: float,
    z_threshold: float = 2.0,
) -> dict[str, float | list[dict[str, float]]]:
    """Test whether alpha slopes at multiple wave numbers share one exchange ratio."""

    if not wave_numbers:
        raise ValueError("wave_numbers must not be empty")
    if len(wave_numbers) != len(observed_alpha_decay_per_renewal):
        raise ValueError("wave_numbers and observed_alpha_decay_per_renewal must have the same length")
    if len(wave_numbers) != len(alpha_decay_per_renewal_std):
        raise ValueError("wave_numbers and alpha_decay_per_renewal_std must have the same length")

    per_wave_number = []
    for wave_number, rate, rate_std in zip(
        wave_numbers,
        observed_alpha_decay_per_renewal,
        alpha_decay_per_renewal_std,
    ):
        per_wave_number.append(
            infer_gamma_exchange_uncertainty_from_late_observables(
                wave_number=wave_number,
                params=params,
                late_renewal_count=late_renewal_count,
                late_ngp=late_ngp,
                observed_alpha_decay_per_renewal=rate,
                late_renewal_count_std=late_renewal_count_std,
                late_ngp_std=late_ngp_std,
                alpha_decay_per_renewal_std=rate_std,
                z_threshold=z_threshold,
            )
        )

    ratio_from_late_ngp = per_wave_number[0]["ratio_from_late_ngp"]
    ratio_from_late_ngp_std = per_wave_number[0]["ratio_from_late_ngp_std"]
    weights = []
    ratios = []
    for row in per_wave_number:
        sigma = row["ratio_from_alpha_rate_std"]
        if sigma <= 0.0:
            raise ValueError("alpha-rate ratio standard deviations must be positive")
        weights.append(1.0 / sigma**2)
        ratios.append(row["ratio_from_alpha_rate"])
    weight_sum = sum(weights)
    weighted_mean = sum(weight * ratio for weight, ratio in zip(weights, ratios)) / weight_sum
    weighted_mean_std = math.sqrt(1.0 / weight_sum)
    collapse_std = math.hypot(weighted_mean_std, ratio_from_late_ngp_std)
    collapse_z_score = abs(weighted_mean - ratio_from_late_ngp) / collapse_std if collapse_std > 0.0 else math.inf
    chi_square = sum(weight * (ratio - weighted_mean) ** 2 for weight, ratio in zip(weights, ratios))
    degrees_of_freedom = max(0.0, float(len(ratios) - 1))
    reduced_chi_square = chi_square / degrees_of_freedom if degrees_of_freedom > 0.0 else 0.0
    passes = collapse_z_score <= z_threshold and reduced_chi_square <= z_threshold**2
    return {
        "ratio_from_late_ngp": ratio_from_late_ngp,
        "ratio_from_late_ngp_std": ratio_from_late_ngp_std,
        "weighted_mean_ratio_from_alpha": weighted_mean,
        "weighted_mean_ratio_from_alpha_std": weighted_mean_std,
        "collapse_z_score": collapse_z_score,
        "alpha_ratio_chi_square": chi_square,
        "alpha_ratio_degrees_of_freedom": degrees_of_freedom,
        "alpha_ratio_reduced_chi_square": reduced_chi_square,
        "z_threshold": z_threshold,
        "passes_multik_collapse": 1.0 if passes else 0.0,
        "per_wave_number": per_wave_number,
    }


def _log_residual(observed: float, predicted: float) -> float:
    if observed <= 0.0 or predicted <= 0.0:
        return math.inf
    return math.log(observed / predicted)


def _late_selection_row(
    *,
    model: str,
    predicted_earlier_ngp: float,
    predicted_later_ngp: float,
    predicted_alpha_decay_per_renewal: float,
    earlier_ngp: float,
    later_ngp: float,
    observed_alpha_decay_per_renewal: float,
    score_tolerance: float,
    extras: dict[str, float] | None = None,
) -> dict[str, float | str]:
    residuals = [
        _log_residual(earlier_ngp, predicted_earlier_ngp),
        _log_residual(later_ngp, predicted_later_ngp),
        _log_residual(observed_alpha_decay_per_renewal, predicted_alpha_decay_per_renewal),
    ]
    finite_residuals = [value for value in residuals if math.isfinite(value)]
    score = math.inf if len(finite_residuals) != len(residuals) else math.sqrt(sum(value**2 for value in residuals))
    row: dict[str, float | str] = {
        "model": model,
        "predicted_earlier_ngp": predicted_earlier_ngp,
        "predicted_later_ngp": predicted_later_ngp,
        "predicted_alpha_decay_per_renewal": predicted_alpha_decay_per_renewal,
        "earlier_ngp_log_residual": residuals[0],
        "later_ngp_log_residual": residuals[1],
        "alpha_slope_log_residual": residuals[2],
        "score": score,
        "score_tolerance": score_tolerance,
        "passes": 1.0 if score <= score_tolerance else 0.0,
    }
    if extras is not None:
        row.update(extras)
    return row


def late_mechanism_selection(
    *,
    wave_number: float,
    params: DelayedRenewalCageParams,
    earlier_renewal_count: float,
    earlier_ngp: float,
    later_renewal_count: float,
    later_ngp: float,
    observed_alpha_decay_per_renewal: float,
    score_tolerance: float = 0.2,
) -> dict[str, str | dict[str, float | str]]:
    """Classify late-time NGP recovery and alpha slowing mechanisms.

    The selector compares three asymptotic count mechanisms using two late NGP
    measurements and one cage-normalized alpha slope per renewal:

    * Poisson delayed renewal: alpha_2 ~= 1/R and alpha slope Gamma_k.
    * Static gamma disorder: alpha_2 ~= 1/R + 1/kappa and alpha slope tends 0.
    * Finite exchange: alpha_2 ~= (1+c)/R and alpha slope log(1+Gamma_k c)/c.

    This is intended as a falsification diagnostic, not a full likelihood model.
    The two NGP points distinguish true long-time recovery from a static plateau.
    """

    _validate(params)
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if earlier_renewal_count <= 0.0 or later_renewal_count <= 0.0:
        raise ValueError("renewal counts must be positive")
    if earlier_renewal_count >= later_renewal_count:
        raise ValueError("earlier_renewal_count must be smaller than later_renewal_count")
    if earlier_ngp <= 0.0 or later_ngp <= 0.0:
        raise ValueError("NGP values must be positive")
    if observed_alpha_decay_per_renewal <= 0.0:
        raise ValueError("observed_alpha_decay_per_renewal must be positive")
    if score_tolerance <= 0.0:
        raise ValueError("score_tolerance must be positive")

    gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
    if gamma <= 0.0:
        raise ValueError("wave_number and jump_variance imply zero alpha decay rate")

    poisson = _late_selection_row(
        model="poisson",
        predicted_earlier_ngp=1.0 / earlier_renewal_count,
        predicted_later_ngp=1.0 / later_renewal_count,
        predicted_alpha_decay_per_renewal=gamma,
        earlier_ngp=earlier_ngp,
        later_ngp=later_ngp,
        observed_alpha_decay_per_renewal=observed_alpha_decay_per_renewal,
        score_tolerance=score_tolerance,
        extras={"gamma_k": gamma},
    )

    static_plateau = later_ngp - 1.0 / later_renewal_count
    if static_plateau > 0.0:
        static_shape = 1.0 / static_plateau
        static_alpha_slope = static_shape * math.log1p(gamma * later_renewal_count / static_shape) / later_renewal_count
        static_gamma = _late_selection_row(
            model="static_gamma",
            predicted_earlier_ngp=1.0 / earlier_renewal_count + static_plateau,
            predicted_later_ngp=1.0 / later_renewal_count + static_plateau,
            predicted_alpha_decay_per_renewal=static_alpha_slope,
            earlier_ngp=earlier_ngp,
            later_ngp=later_ngp,
            observed_alpha_decay_per_renewal=observed_alpha_decay_per_renewal,
            score_tolerance=score_tolerance,
            extras={
                "gamma_k": gamma,
                "inferred_static_plateau": static_plateau,
                "inferred_static_shape": static_shape,
            },
        )
    else:
        static_gamma = _late_selection_row(
            model="static_gamma",
            predicted_earlier_ngp=math.nan,
            predicted_later_ngp=math.nan,
            predicted_alpha_decay_per_renewal=math.nan,
            earlier_ngp=earlier_ngp,
            later_ngp=later_ngp,
            observed_alpha_decay_per_renewal=observed_alpha_decay_per_renewal,
            score_tolerance=score_tolerance,
            extras={
                "gamma_k": gamma,
                "inferred_static_plateau": static_plateau,
                "inferred_static_shape": math.nan,
            },
        )

    exchange_ratio = later_renewal_count * later_ngp - 1.0
    if exchange_ratio >= 0.0:
        if math.isclose(exchange_ratio, 0.0, rel_tol=1e-14, abs_tol=1e-14):
            exchange_alpha_slope = gamma
        else:
            exchange_alpha_slope = math.log1p(gamma * exchange_ratio) / exchange_ratio
        finite_exchange = _late_selection_row(
            model="finite_exchange",
            predicted_earlier_ngp=(1.0 + exchange_ratio) / earlier_renewal_count,
            predicted_later_ngp=(1.0 + exchange_ratio) / later_renewal_count,
            predicted_alpha_decay_per_renewal=exchange_alpha_slope,
            earlier_ngp=earlier_ngp,
            later_ngp=later_ngp,
            observed_alpha_decay_per_renewal=observed_alpha_decay_per_renewal,
            score_tolerance=score_tolerance,
            extras={
                "gamma_k": gamma,
                "inferred_exchange_ratio": exchange_ratio,
                "late_ngp_renewal_amplitude": 1.0 + exchange_ratio,
            },
        )
    else:
        finite_exchange = _late_selection_row(
            model="finite_exchange",
            predicted_earlier_ngp=math.nan,
            predicted_later_ngp=math.nan,
            predicted_alpha_decay_per_renewal=math.nan,
            earlier_ngp=earlier_ngp,
            later_ngp=later_ngp,
            observed_alpha_decay_per_renewal=observed_alpha_decay_per_renewal,
            score_tolerance=score_tolerance,
            extras={
                "gamma_k": gamma,
                "inferred_exchange_ratio": exchange_ratio,
                "late_ngp_renewal_amplitude": later_renewal_count * later_ngp,
            },
        )

    rows = {
        "poisson": poisson,
        "static_gamma": static_gamma,
        "finite_exchange": finite_exchange,
    }
    best_model = min(rows, key=lambda name: float(rows[name]["score"]))
    return {
        "best_model": best_model,
        "poisson": poisson,
        "static_gamma": static_gamma,
        "finite_exchange": finite_exchange,
    }


def local_alpha_stretching_exponent(t: np.ndarray, decay: np.ndarray) -> np.ndarray:
    """Local KWW-like exponent ``d log[-log(decay)] / d log(t)``."""

    t = np.asarray(t, dtype=float)
    decay = np.asarray(decay, dtype=float)
    if t.ndim != 1 or decay.ndim != 1:
        raise ValueError("t and decay must be one-dimensional")
    if t.size != decay.size:
        raise ValueError("t and decay must have the same length")
    if np.any(t <= 0.0):
        raise ValueError("t values must be positive")
    if np.any((decay <= 0.0) | (decay > 1.0)):
        raise ValueError("decay values must lie in (0, 1]")
    log_decay = -np.log(decay)
    exponent = np.full_like(log_decay, np.nan)
    mask = log_decay > 0.0
    if np.count_nonzero(mask) >= 2:
        exponent[mask] = np.gradient(np.log(log_decay[mask]), np.log(t[mask]))
    return exponent


def kww_alpha_fit(
    t: np.ndarray,
    decay: np.ndarray,
    *,
    min_decay: float,
    max_decay: float,
) -> dict[str, float]:
    """Fit a KWW alpha window, ``decay=exp[-(t/tau)^beta]``."""

    t = np.asarray(t, dtype=float)
    decay = np.asarray(decay, dtype=float)
    if t.ndim != 1 or decay.ndim != 1:
        raise ValueError("t and decay must be one-dimensional")
    if t.size != decay.size:
        raise ValueError("t and decay must have the same length")
    if np.any(t <= 0.0):
        raise ValueError("t values must be positive")
    if np.any((decay <= 0.0) | (decay > 1.0)):
        raise ValueError("decay values must lie in (0, 1]")
    if not (0.0 < min_decay < max_decay < 1.0):
        raise ValueError("min_decay and max_decay must satisfy 0 < min < max < 1")

    mask = (decay >= min_decay) & (decay <= max_decay) & (-np.log(decay) > 0.0)
    if np.count_nonzero(mask) < 3:
        raise ValueError("KWW fit window must contain at least three points")
    x = np.log(t[mask])
    y = np.log(-np.log(decay[mask]))
    beta, intercept = np.polyfit(x, y, 1)
    if beta <= 0.0:
        raise ValueError("fitted KWW beta must be positive")
    tau = math.exp(-intercept / beta)
    fitted = beta * x + intercept
    residual = y - fitted
    return {
        "kww_beta": float(beta),
        "kww_tau": float(tau),
        "kww_intercept": float(intercept),
        "rms_log_residual": float(math.sqrt(np.mean(residual**2))),
        "points_used": float(np.count_nonzero(mask)),
        "fit_min_decay": min_decay,
        "fit_max_decay": max_decay,
    }


def stretched_alpha_benchmark_consistency(
    *,
    benchmark_id: str,
    observed_stretched_alpha: bool,
    hot_kww_beta: float,
    cold_kww_beta: float,
    min_beta_drop: float,
    max_cold_beta: float,
    max_fit_residual: float,
    cold_fit_residual: float,
) -> dict[str, float | str]:
    """Check KWW stretching and cooling trend against benchmark expectations."""

    if not benchmark_id:
        raise ValueError("benchmark_id must be nonempty")
    for name, value in {
        "hot_kww_beta": hot_kww_beta,
        "cold_kww_beta": cold_kww_beta,
        "max_cold_beta": max_cold_beta,
        "max_fit_residual": max_fit_residual,
        "cold_fit_residual": cold_fit_residual,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if min_beta_drop < 0.0:
        raise ValueError("min_beta_drop must be nonnegative")

    beta_drop = hot_kww_beta - cold_kww_beta
    drop_flag = beta_drop >= min_beta_drop
    cold_flag = cold_kww_beta <= max_cold_beta
    fit_flag = cold_fit_residual <= max_fit_residual
    model_flag = drop_flag and cold_flag and fit_flag
    return {
        "benchmark_id": benchmark_id,
        "observed_stretched_alpha": float(observed_stretched_alpha),
        "hot_kww_beta": hot_kww_beta,
        "cold_kww_beta": cold_kww_beta,
        "kww_beta_drop": beta_drop,
        "min_beta_drop": min_beta_drop,
        "max_cold_beta": max_cold_beta,
        "cold_fit_residual": cold_fit_residual,
        "max_fit_residual": max_fit_residual,
        "model_predicts_stretched_alpha": float(model_flag),
        "beta_drop_consistent": float(drop_flag == observed_stretched_alpha),
        "cold_beta_consistent": float(cold_flag == observed_stretched_alpha),
        "fit_quality_consistent": float(fit_flag == observed_stretched_alpha),
        "overall_consistent": float(model_flag == observed_stretched_alpha),
    }


def renewal_scattering_susceptibility(
    wave_number: float,
    t: np.ndarray,
    params: DelayedRenewalCageParams,
) -> np.ndarray:
    """Renewal-count contribution to the self-scattering variance.

    For W=E[exp(i k Delta x)|N], W=exp[-k^2 L(t)/2] a^N with
    a=exp(-k^2 q/2). The Poisson average gives Var(W) in closed form. This is
    the single-particle renewal-count analogue of a four-point susceptibility.
    """

    _validate(params)
    if wave_number < 0.0:
        raise ValueError("wave_number must be nonnegative")
    t = np.asarray(t, dtype=float)
    local = local_cage_variance(t, params)
    renewal = delayed_poisson_mean(t, params)
    jump_characteristic = math.exp(-0.5 * wave_number**2 * params.jump_variance)
    second_moment = np.exp(-wave_number**2 * local + renewal * (jump_characteristic**2 - 1.0))
    mean_square = self_intermediate_scattering(wave_number, t, params) ** 2
    return second_moment - mean_square


def correlated_domain_susceptibility(
    wave_number: float,
    t: np.ndarray,
    params: DelayedRenewalCageParams,
    *,
    correlation_size: float,
) -> np.ndarray:
    """Renewal contribution to chi4 for domains sharing one renewal count.

    If each correlated renewal domain contains ``correlation_size`` particles
    with the same renewal-count history, the per-particle four-point
    susceptibility contribution scales as ``N_corr * chi_R``.
    """

    if correlation_size <= 0.0:
        raise ValueError("correlation_size must be positive")
    return correlation_size * renewal_scattering_susceptibility(wave_number, t, params)


def infer_renewal_correlation_size(
    *,
    observed_chi4_peak: float,
    wave_number: float,
    t: np.ndarray,
    params: DelayedRenewalCageParams,
) -> dict[str, float]:
    """Infer correlated renewal-domain size from an observed chi4 peak."""

    if observed_chi4_peak <= 0.0:
        raise ValueError("observed_chi4_peak must be positive")
    susceptibility = renewal_scattering_susceptibility(wave_number, t, params)
    peak_idx = int(np.argmax(susceptibility))
    model_peak = float(susceptibility[peak_idx])
    if model_peak <= 0.0:
        raise ValueError("model renewal susceptibility peak must be positive")
    t = np.asarray(t, dtype=float)
    return {
        "correlation_size": observed_chi4_peak / model_peak,
        "observed_chi4_peak": observed_chi4_peak,
        "model_single_particle_peak": model_peak,
        "peak_time": float(t[peak_idx]),
    }


def spatial_facilitation_domain(
    *,
    persistence_time: float,
    dimension: int = 3,
    particle_density: float = 1.0,
    facilitation_diffusivity: float = 0.0,
    microscopic_length: float = 1.0,
) -> dict[str, float]:
    """Convert a persistence clock into a facilitation-front correlation volume.

    A minimal diffusive facilitation front explores
    ``xi=(ell0^2+2 d D_f tau_p)^(1/2)`` during the persistence time. The
    correlated renewal size is the expected particle count in a d-dimensional
    ball of radius ``xi``.
    """

    if persistence_time <= 0.0:
        raise ValueError("persistence_time must be positive")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if particle_density <= 0.0:
        raise ValueError("particle_density must be positive")
    if facilitation_diffusivity < 0.0:
        raise ValueError("facilitation_diffusivity must be nonnegative")
    if microscopic_length <= 0.0:
        raise ValueError("microscopic_length must be positive")

    length = math.sqrt(microscopic_length**2 + 2.0 * dimension * facilitation_diffusivity * persistence_time)
    unit_ball_volume = math.pi ** (0.5 * dimension) / math.gamma(0.5 * dimension + 1.0)
    correlation_volume = unit_ball_volume * length**dimension
    return {
        "persistence_time": persistence_time,
        "dimension": float(dimension),
        "particle_density": particle_density,
        "facilitation_diffusivity": facilitation_diffusivity,
        "microscopic_length": microscopic_length,
        "front_dynamic_exponent": 2.0,
        "dynamic_correlation_length": length,
        "correlation_volume": correlation_volume,
        "correlation_size": particle_density * correlation_volume,
    }


def infer_spatial_facilitation_diffusivity(
    *,
    persistence_times: np.ndarray,
    observed_dynamic_lengths: np.ndarray | None = None,
    observed_correlation_sizes: np.ndarray | None = None,
    dimension: int = 3,
    particle_density: float = 1.0,
    microscopic_length: float = 1.0,
) -> list[dict[str, float]]:
    """Infer the diffusive facilitation-front coefficient from spatial data."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if particle_density <= 0.0:
        raise ValueError("particle_density must be positive")
    if microscopic_length <= 0.0:
        raise ValueError("microscopic_length must be positive")
    if (observed_dynamic_lengths is None) == (observed_correlation_sizes is None):
        raise ValueError("provide exactly one of observed_dynamic_lengths or observed_correlation_sizes")

    persistence_times = np.asarray(persistence_times, dtype=float)
    if persistence_times.ndim != 1 or persistence_times.size == 0:
        raise ValueError("persistence_times must be a nonempty one-dimensional array")
    if np.any(persistence_times <= 0.0):
        raise ValueError("persistence_times must be positive")

    unit_ball_volume = math.pi ** (0.5 * dimension) / math.gamma(0.5 * dimension + 1.0)
    if observed_dynamic_lengths is not None:
        lengths = np.asarray(observed_dynamic_lengths, dtype=float)
        if lengths.shape != persistence_times.shape:
            raise ValueError("observed_dynamic_lengths must match persistence_times")
        if np.any(lengths <= 0.0):
            raise ValueError("observed_dynamic_lengths must be positive")
        correlation_sizes = particle_density * unit_ball_volume * lengths**dimension
    else:
        correlation_sizes = np.asarray(observed_correlation_sizes, dtype=float)
        if correlation_sizes.shape != persistence_times.shape:
            raise ValueError("observed_correlation_sizes must match persistence_times")
        if np.any(correlation_sizes <= 0.0):
            raise ValueError("observed_correlation_sizes must be positive")
        lengths = (correlation_sizes / (particle_density * unit_ball_volume)) ** (1.0 / dimension)

    diffusivities = (lengths**2 - microscopic_length**2) / (2.0 * dimension * persistence_times)
    if np.any(diffusivities < -1.0e-14):
        raise ValueError("observed lengths imply negative facilitation diffusivity")
    diffusivities = np.maximum(diffusivities, 0.0)

    rows: list[dict[str, float]] = []
    for idx, persistence_time in enumerate(persistence_times):
        rows.append(
            {
                "point_index": float(idx),
                "persistence_time": float(persistence_time),
                "dynamic_correlation_length": float(lengths[idx]),
                "correlation_size": float(correlation_sizes[idx]),
                "inferred_facilitation_diffusivity": float(diffusivities[idx]),
                "front_dynamic_exponent": 2.0,
            }
        )
    return rows


def spatial_facilitation_growth_law_consistency(
    *,
    persistence_times: np.ndarray,
    observed_diffusive_front_growth: bool,
    observed_dynamic_lengths: np.ndarray | None = None,
    observed_correlation_sizes: np.ndarray | None = None,
    dimension: int = 3,
    particle_density: float = 1.0,
    microscopic_length: float = 1.0,
    max_diffusivity_relative_std: float,
    min_length_growth: float,
) -> dict[str, float]:
    """Check whether spatial growth is consistent with one diffusive front law."""

    if max_diffusivity_relative_std < 0.0:
        raise ValueError("max_diffusivity_relative_std must be nonnegative")
    if min_length_growth <= 0.0:
        raise ValueError("min_length_growth must be positive")
    rows = infer_spatial_facilitation_diffusivity(
        persistence_times=persistence_times,
        observed_dynamic_lengths=observed_dynamic_lengths,
        observed_correlation_sizes=observed_correlation_sizes,
        dimension=dimension,
        particle_density=particle_density,
        microscopic_length=microscopic_length,
    )
    diffusivities = np.array([row["inferred_facilitation_diffusivity"] for row in rows], dtype=float)
    lengths = np.array([row["dynamic_correlation_length"] for row in rows], dtype=float)
    mean_diffusivity = float(np.mean(diffusivities))
    diffusivity_std = float(np.std(diffusivities))
    if mean_diffusivity > 0.0:
        diffusivity_relative_std = diffusivity_std / mean_diffusivity
    else:
        diffusivity_relative_std = 0.0 if diffusivity_std == 0.0 else math.inf
    length_growth = float(lengths[-1] / lengths[0])
    clock_growth = float(np.asarray(persistence_times, dtype=float)[-1] / np.asarray(persistence_times, dtype=float)[0])
    constant_front_flag = mean_diffusivity > 0.0 and diffusivity_relative_std <= max_diffusivity_relative_std
    length_growth_flag = length_growth >= min_length_growth
    model_flag = constant_front_flag and length_growth_flag
    consistent = model_flag == observed_diffusive_front_growth
    return {
        "observed_diffusive_front_growth": float(observed_diffusive_front_growth),
        "number_of_points": float(len(rows)),
        "facilitation_diffusivity_mean": mean_diffusivity,
        "facilitation_diffusivity_std": diffusivity_std,
        "facilitation_diffusivity_relative_std": diffusivity_relative_std,
        "max_diffusivity_relative_std": max_diffusivity_relative_std,
        "length_growth": length_growth,
        "persistence_time_growth": clock_growth,
        "min_length_growth": min_length_growth,
        "model_predicts_diffusive_front_growth": float(model_flag),
        "constant_diffusivity_consistent": float(constant_front_flag == observed_diffusive_front_growth),
        "length_growth_consistent": float(length_growth_flag == observed_diffusive_front_growth),
        "facilitation_growth_law_consistent": float(consistent),
        "overall_consistent": float(consistent),
    }


def spatial_facilitation_chi4_scan(
    *,
    temperatures: np.ndarray,
    law: TemperatureLawParams,
    wave_number: float,
    facilitation_diffusivity: float,
    particle_density: float = 1.0,
    dimension: int = 3,
    microscopic_length: float = 1.0,
    time_points: int = 500,
) -> list[dict[str, float]]:
    """Temperature scan for clock-derived dynamic length and chi4 amplitude."""

    _validate_temperature_law(law)
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if time_points < 50:
        raise ValueError("time_points must be at least 50")
    temperatures = np.asarray(temperatures, dtype=float)
    if np.any(temperatures <= 0.0):
        raise ValueError("temperatures must be positive")

    rows: list[dict[str, float]] = []
    reference_peak = None
    reference_length = None
    reference_size = None
    for temperature in temperatures:
        params = temperature_dependent_params(float(temperature), law)
        domain = spatial_facilitation_domain(
            persistence_time=params.renewal_delay,
            dimension=dimension,
            particle_density=particle_density,
            facilitation_diffusivity=facilitation_diffusivity,
            microscopic_length=microscopic_length,
        )
        tau_alpha = alpha_relaxation_time(wave_number, params)
        upper = max(80.0 * params.renewal_delay, 10.0 * tau_alpha, 10.0 * params.cage_tau)
        time_grid = np.linspace(0.0, upper, time_points)
        single_particle = renewal_scattering_susceptibility(wave_number, time_grid, params)
        chi4 = domain["correlation_size"] * single_particle
        peak_index = int(np.argmax(chi4))
        chi4_peak = float(chi4[peak_index])
        if reference_peak is None:
            reference_peak = chi4_peak
            reference_length = domain["dynamic_correlation_length"]
            reference_size = domain["correlation_size"]
        rows.append(
            {
                "temperature": float(temperature),
                "renewal_delay": params.renewal_delay,
                "renewal_rate": params.renewal_rate,
                "tau_alpha": tau_alpha,
                "dynamic_correlation_length": domain["dynamic_correlation_length"],
                "correlation_size": domain["correlation_size"],
                "single_particle_chi_peak": float(single_particle[peak_index]),
                "chi4_peak": chi4_peak,
                "chi4_peak_time": float(time_grid[peak_index]),
                "length_growth": domain["dynamic_correlation_length"] / reference_length,
                "correlation_size_growth": domain["correlation_size"] / reference_size,
                "chi4_peak_growth": chi4_peak / reference_peak,
            }
        )
    return rows


def alpha_relaxation_time(
    wave_number: float,
    params: DelayedRenewalCageParams,
    *,
    threshold: float = math.exp(-1.0),
) -> float:
    """Time at which the cage-normalized alpha decay reaches ``threshold``."""

    _validate(params)
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie between 0 and 1")
    gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
    if gamma <= 0.0:
        raise ValueError("wave_number and jump_variance imply zero alpha decay rate")
    target_renewal_count = -math.log(threshold) / gamma
    return _find_time_for_renewal_count(target_renewal_count, params)


def peak_relaxation_coupling(
    wave_number: float,
    params: DelayedRenewalCageParams,
    *,
    threshold: float = math.exp(-1.0),
) -> dict[str, float]:
    """Closed relation between the NGP peak time and alpha relaxation time."""

    _validate(params)
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie between 0 and 1")
    gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
    if gamma <= 0.0:
        raise ValueError("wave_number and jump_variance imply zero alpha decay rate")

    peak = dimensionless_peak_prediction(params)
    peak_time = peak["peak_time"]
    tau_alpha = alpha_relaxation_time(wave_number, params, threshold=threshold)
    alpha_renewal_count = -math.log(threshold) / gamma
    peak_renewal_count = peak["target_renewal_count"]

    return {
        "wave_number": wave_number,
        "threshold": threshold,
        "gamma_k": gamma,
        "peak_time": peak_time,
        "tau_alpha": tau_alpha,
        "tau_alpha_over_peak_time": tau_alpha / peak_time,
        "peak_renewal_count": peak_renewal_count,
        "alpha_renewal_count": alpha_renewal_count,
        "alpha_to_peak_renewal_count_ratio": alpha_renewal_count / peak_renewal_count,
        "peak_ngp": peak["peak_ngp"],
    }


def _alpha_shape_control(wave_number: float, params: DelayedRenewalCageParams) -> float:
    gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
    return gamma * params.renewal_rate * params.renewal_delay


def alpha_relaxation_shape_curve(
    wave_number: float,
    params: DelayedRenewalCageParams,
    scaled_time: np.ndarray,
    *,
    threshold: float = math.exp(-1.0),
) -> np.ndarray:
    """Alpha-relaxation shape after scaling time by ``tau_alpha``.

    The returned curve is ``-log Phi_alpha(k, u tau_alpha) / [-log(threshold)]``.
    It equals one at ``u=1`` and is independent of the cage Debye-Waller factor.
    For the minimal Poisson renewal model its shape is controlled by the single
    dimensionless number ``Gamma_k lambda tau_d``.
    """

    _validate(params)
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie between 0 and 1")
    scaled_time = np.asarray(scaled_time, dtype=float)
    if np.any(scaled_time <= 0.0):
        raise ValueError("scaled_time values must be positive")
    gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
    tau_alpha = alpha_relaxation_time(wave_number, params, threshold=threshold)
    return gamma * delayed_poisson_mean(scaled_time * tau_alpha, params) / (-math.log(threshold))


def alpha_shape_superposition_residual(
    wave_number: float,
    reference_params: DelayedRenewalCageParams,
    candidate_params: DelayedRenewalCageParams,
    scaled_time: np.ndarray,
    *,
    threshold: float = math.exp(-1.0),
) -> dict[str, float]:
    """RMS log-shape residual for time-temperature superposition tests."""

    scaled_time = np.asarray(scaled_time, dtype=float)
    reference_curve = alpha_relaxation_shape_curve(
        wave_number,
        reference_params,
        scaled_time,
        threshold=threshold,
    )
    candidate_curve = alpha_relaxation_shape_curve(
        wave_number,
        candidate_params,
        scaled_time,
        threshold=threshold,
    )
    if np.any(reference_curve <= 0.0) or np.any(candidate_curve <= 0.0):
        raise ValueError("alpha shape curves must be positive")
    reference_tau_alpha = alpha_relaxation_time(wave_number, reference_params, threshold=threshold)
    candidate_tau_alpha = alpha_relaxation_time(wave_number, candidate_params, threshold=threshold)
    residual = np.log(candidate_curve) - np.log(reference_curve)
    return {
        "wave_number": wave_number,
        "threshold": threshold,
        "reference_control": _alpha_shape_control(wave_number, reference_params),
        "candidate_control": _alpha_shape_control(wave_number, candidate_params),
        "reference_tau_alpha": reference_tau_alpha,
        "candidate_tau_alpha": candidate_tau_alpha,
        "reference_tau_alpha_over_delay": reference_tau_alpha / reference_params.renewal_delay,
        "candidate_tau_alpha_over_delay": candidate_tau_alpha / candidate_params.renewal_delay,
        "rms_log_shape_residual": float(math.sqrt(float(np.mean(residual**2)))),
        "max_abs_log_shape_residual": float(np.max(np.abs(residual))),
    }


def long_time_diffusion_coefficient(params: DelayedRenewalCageParams) -> float:
    """Long-time self-diffusion coefficient per coordinate."""

    _validate(params)
    return 0.5 * params.renewal_rate * params.jump_variance


def stokes_einstein_product(wave_number: float, params: DelayedRenewalCageParams) -> float:
    """Return D times the cage-normalized alpha-relaxation time."""

    return long_time_diffusion_coefficient(params) * alpha_relaxation_time(wave_number, params)


def fractional_stokes_einstein_exponents(
    diffusion: np.ndarray,
    tau_alpha: np.ndarray,
) -> np.ndarray:
    """Return local exponents in ``D ~ tau_alpha^{-xi}``.

    The returned value is ``xi=-d log(D)/d log(tau_alpha)`` evaluated by finite
    differences on the supplied scan. Ordinary Stokes-Einstein scaling gives
    ``xi=1``; fractional Stokes-Einstein decoupling gives ``0<xi<1``.
    """

    diffusion = np.asarray(diffusion, dtype=float)
    tau_alpha = np.asarray(tau_alpha, dtype=float)
    if diffusion.ndim != 1 or tau_alpha.ndim != 1:
        raise ValueError("diffusion and tau_alpha must be one-dimensional")
    if diffusion.size != tau_alpha.size:
        raise ValueError("diffusion and tau_alpha must have the same length")
    if diffusion.size < 2:
        raise ValueError("at least two scan points are required")
    if np.any(diffusion <= 0.0) or np.any(tau_alpha <= 0.0):
        raise ValueError("diffusion and tau_alpha values must be positive")

    log_tau = np.log(tau_alpha)
    if np.any(np.isclose(np.diff(log_tau), 0.0)):
        raise ValueError("tau_alpha values must be distinct")
    return -np.gradient(np.log(diffusion), log_tau)


def apparent_alpha_activation_energies(
    temperatures: np.ndarray,
    tau_alpha: np.ndarray,
) -> np.ndarray:
    """Return local slopes ``d ln(tau_alpha) / d(1/T)``.

    For Arrhenius relaxation ``tau_alpha=tau_0 exp(E/T)``, this diagnostic
    returns the constant barrier ``E``. Growth of this slope on cooling is the
    minimal fragility signal captured by the temperature-dependent model.
    """

    temperatures = np.asarray(temperatures, dtype=float)
    tau_alpha = np.asarray(tau_alpha, dtype=float)
    if temperatures.ndim != 1 or tau_alpha.ndim != 1:
        raise ValueError("temperatures and tau_alpha must be one-dimensional")
    if temperatures.size != tau_alpha.size:
        raise ValueError("temperatures and tau_alpha must have the same length")
    if temperatures.size < 2:
        raise ValueError("at least two scan points are required")
    if np.any(temperatures <= 0.0) or np.any(tau_alpha <= 0.0):
        raise ValueError("temperatures and tau_alpha values must be positive")

    inverse_temperature = 1.0 / temperatures
    if np.any(np.isclose(np.diff(inverse_temperature), 0.0)):
        raise ValueError("temperature values must be distinct")
    return np.gradient(np.log(tau_alpha), inverse_temperature)


def infer_parameters_from_scattering_transport(
    *,
    wave_number: float,
    debye_waller_plateau: float,
    diffusion_coefficient: float,
    tau_alpha: float,
    renewal_delay: float,
    threshold: float = math.exp(-1.0),
) -> dict[str, float]:
    """Infer minimal renewal-cage parameters from common observables.

    The cage plateau of the self-intermediate scattering function gives
    ``A=-2 log(f_c)/k^2``. The long-time diffusion coefficient gives
    ``D=lambda q/2``. The cage-normalized alpha time gives

        lambda tau_d F(tau_alpha/tau_d) [1-exp(-k^2 q/2)] = -log(threshold),

    which determines ``q`` after eliminating ``lambda``. The same equation yields
    a direct falsifiability condition: the dimensionless existence margin below
    must exceed unity for any positive jump variance to satisfy the observables.
    """

    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if not 0.0 < debye_waller_plateau < 1.0:
        raise ValueError("debye_waller_plateau must lie between zero and one")
    if diffusion_coefficient <= 0.0:
        raise ValueError("diffusion_coefficient must be positive")
    if tau_alpha <= 0.0:
        raise ValueError("tau_alpha must be positive")
    if renewal_delay <= 0.0:
        raise ValueError("renewal_delay must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie between 0 and 1")

    cage_variance = -2.0 * math.log(debye_waller_plateau) / (wave_number**2)
    shape = delayed_renewal_shape(tau_alpha / renewal_delay)
    target = -math.log(threshold)
    half_k_squared = 0.5 * wave_number**2
    transport_scale = 2.0 * diffusion_coefficient * renewal_delay * shape
    target_ratio = target / transport_scale
    existence_margin = transport_scale * half_k_squared / target
    if existence_margin <= 1.0:
        raise ValueError(
            "observables fail the scattering-transport existence criterion: "
            "D tau_d F(tau_alpha/tau_d) k^2 / [-log(threshold)] must exceed 1"
        )

    def gamma_over_q(jump_variance: float) -> float:
        return (1.0 - math.exp(-half_k_squared * jump_variance)) / jump_variance

    high = max(cage_variance, 1.0)
    while gamma_over_q(high) > target_ratio:
        high *= 2.0
    low = 0.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        if gamma_over_q(mid) > target_ratio:
            low = mid
        else:
            high = mid
    jump_variance = 0.5 * (low + high)
    renewal_rate = 2.0 * diffusion_coefficient / jump_variance
    inferred_params = DelayedRenewalCageParams(
        cage_variance=cage_variance,
        cage_tau=1.0,
        jump_variance=jump_variance,
        renewal_rate=renewal_rate,
        renewal_delay=renewal_delay,
    )
    peak = dimensionless_peak_prediction(inferred_params)
    reconstructed_tau_alpha = alpha_relaxation_time(wave_number, inferred_params, threshold=threshold)
    return {
        "wave_number": wave_number,
        "threshold": threshold,
        "cage_variance": cage_variance,
        "jump_variance": jump_variance,
        "jump_to_cage_variance": jump_variance / cage_variance,
        "renewal_rate": renewal_rate,
        "renewal_delay": renewal_delay,
        "lambda_tau_delay": renewal_rate * renewal_delay,
        "delayed_renewal_shape_at_tau_alpha": shape,
        "existence_margin": existence_margin,
        "reconstructed_debye_waller_plateau": math.exp(-0.5 * wave_number**2 * cage_variance),
        "reconstructed_diffusion_coefficient": long_time_diffusion_coefficient(inferred_params),
        "reconstructed_tau_alpha": reconstructed_tau_alpha,
        "predicted_ngp_peak_time": peak["peak_time"],
        "predicted_ngp_peak": peak["peak_ngp"],
    }


def infer_parameters_from_full_observables(
    *,
    wave_number: float,
    debye_waller_plateau: float,
    diffusion_coefficient: float,
    tau_alpha: float,
    peak_time: float,
    peak_ngp: float,
    threshold: float = math.exp(-1.0),
) -> dict[str, float]:
    """Infer ``A, q, lambda, tau_d`` without externally supplying ``tau_d``.

    The plateau gives ``A``. The plateau NGP peak height gives ``q/A=4 alpha*``.
    The long-time diffusion coefficient gives ``lambda=2D/q``. The peak time
    then fixes ``tau_d`` through ``lambda tau_d F(t*/tau_d)=A/q``. The supplied
    alpha time is held out as a consistency check rather than used as a fit
    parameter.
    """

    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if not 0.0 < debye_waller_plateau < 1.0:
        raise ValueError("debye_waller_plateau must lie between zero and one")
    if diffusion_coefficient <= 0.0:
        raise ValueError("diffusion_coefficient must be positive")
    if tau_alpha <= 0.0:
        raise ValueError("tau_alpha must be positive")
    if peak_time <= 0.0:
        raise ValueError("peak_time must be positive")
    if peak_ngp <= 0.0:
        raise ValueError("peak_ngp must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie between 0 and 1")

    cage_variance = -2.0 * math.log(debye_waller_plateau) / (wave_number**2)
    jump_variance = 4.0 * peak_ngp * cage_variance
    target_renewal_count = cage_variance / jump_variance
    renewal_rate = 2.0 * diffusion_coefficient / jump_variance
    peak_timing_ratio = target_renewal_count / (renewal_rate * peak_time)
    if not 0.0 < peak_timing_ratio < 1.0:
        raise ValueError("peak timing is incompatible with positive delayed-renewal onset")

    low = 0.0
    high = 1.0

    def shape_ratio(scaled_time: float) -> float:
        return delayed_renewal_shape(scaled_time) / scaled_time

    while shape_ratio(high) < peak_timing_ratio:
        high *= 2.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        if shape_ratio(mid) < peak_timing_ratio:
            low = mid
        else:
            high = mid
    scaled_peak_time = 0.5 * (low + high)
    renewal_delay = peak_time / scaled_peak_time
    inferred_params = DelayedRenewalCageParams(
        cage_variance=cage_variance,
        cage_tau=1.0,
        jump_variance=jump_variance,
        renewal_rate=renewal_rate,
        renewal_delay=renewal_delay,
    )
    reconstructed_tau_alpha = alpha_relaxation_time(wave_number, inferred_params, threshold=threshold)
    reconstructed_peak = dimensionless_peak_prediction(inferred_params)
    return {
        "wave_number": wave_number,
        "threshold": threshold,
        "cage_variance": cage_variance,
        "jump_variance": jump_variance,
        "jump_to_cage_variance": jump_variance / cage_variance,
        "renewal_rate": renewal_rate,
        "renewal_delay": renewal_delay,
        "lambda_tau_delay": renewal_rate * renewal_delay,
        "target_renewal_count": target_renewal_count,
        "scaled_peak_time": scaled_peak_time,
        "peak_timing_ratio": peak_timing_ratio,
        "reconstructed_debye_waller_plateau": math.exp(-0.5 * wave_number**2 * cage_variance),
        "reconstructed_diffusion_coefficient": long_time_diffusion_coefficient(inferred_params),
        "reconstructed_tau_alpha": reconstructed_tau_alpha,
        "reconstructed_peak_time": reconstructed_peak["peak_time"],
        "reconstructed_peak_ngp": reconstructed_peak["peak_ngp"],
        "log_tau_alpha_residual": math.log(reconstructed_tau_alpha / tau_alpha),
    }


def temperature_scan(
    temperatures: np.ndarray,
    law: TemperatureLawParams,
    *,
    wave_number: float,
) -> list[dict[str, float]]:
    """Evaluate temperature-dependent transport and relaxation diagnostics."""

    temperatures = np.asarray(temperatures, dtype=float)
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("temperatures must be a nonempty one-dimensional array")
    if np.any(temperatures <= 0.0):
        raise ValueError("temperatures must be positive")

    rows: list[dict[str, float]] = []
    baseline_product = None
    for temperature in temperatures:
        params = temperature_dependent_params(float(temperature), law)
        diffusion = long_time_diffusion_coefficient(params)
        tau_alpha = alpha_relaxation_time(wave_number, params)
        se_product = diffusion * tau_alpha
        if baseline_product is None:
            baseline_product = se_product
        peak = dimensionless_peak_prediction(params)
        rows.append(
            {
                "temperature": float(temperature),
                "cage_variance": params.cage_variance,
                "jump_variance": params.jump_variance,
                "jump_to_cage_variance": params.jump_variance / params.cage_variance,
                "renewal_rate": params.renewal_rate,
                "renewal_delay": params.renewal_delay,
                "lambda_tau_delay": params.renewal_rate * params.renewal_delay,
                "diffusion_coefficient": diffusion,
                "tau_alpha": tau_alpha,
                "stokes_einstein_product": se_product,
                "normalized_stokes_einstein_product": se_product / baseline_product,
                "predicted_ngp_peak_time": peak["peak_time"],
                "predicted_ngp_peak": peak["peak_ngp"],
            }
        )
    exponents = fractional_stokes_einstein_exponents(
        np.array([row["diffusion_coefficient"] for row in rows]),
        np.array([row["tau_alpha"] for row in rows]),
    )
    for row, exponent in zip(rows, exponents):
        row["fractional_stokes_einstein_exponent"] = float(exponent)
    activation_energies = apparent_alpha_activation_energies(
        temperatures,
        np.array([row["tau_alpha"] for row in rows]),
    )
    for row, activation_energy in zip(rows, activation_energies):
        row["apparent_alpha_activation_energy"] = float(activation_energy)
        row["local_fragility_index"] = float(activation_energy / (row["temperature"] * math.log(10.0)))
    return rows


def gamma_exchange_temperature_scan(
    temperatures: np.ndarray,
    cage_law: TemperatureLawParams,
    exchange_law: FacilitatedExchangeLawParams,
    *,
    wave_number: float,
) -> list[dict[str, float]]:
    """Evaluate temperature-dependent finite-exchange heterogeneity diagnostics."""

    temperatures = np.asarray(temperatures, dtype=float)
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("temperatures must be a nonempty one-dimensional array")
    if np.any(temperatures <= 0.0):
        raise ValueError("temperatures must be positive")
    _validate_temperature_law(cage_law)
    _validate_facilitated_exchange_law(exchange_law)
    if not math.isclose(cage_law.reference_temperature, exchange_law.reference_temperature):
        raise ValueError("cage and exchange laws must share a reference temperature")

    rows = []
    for temperature in temperatures:
        params = temperature_dependent_params(float(temperature), cage_law)
        heterogeneity = temperature_dependent_gamma_exchange(float(temperature), exchange_law)
        diagnostics = gamma_exchange_asymptotic_diagnostics(wave_number, params, heterogeneity)
        rows.append(
            {
                "temperature": float(temperature),
                "shape": heterogeneity.shape,
                "exchange_renewal_count": heterogeneity.exchange_renewal_count,
                "heterogeneity_ratio": diagnostics["heterogeneity_ratio"],
                "late_ngp_renewal_amplitude": diagnostics["late_ngp_renewal_amplitude"],
                "late_alpha_decay_per_renewal": diagnostics["late_alpha_decay_per_renewal"],
                "poisson_alpha_decay_per_renewal": diagnostics["poisson_alpha_decay_per_renewal"],
                "alpha_rate_renormalization": diagnostics["alpha_rate_renormalization"],
                "late_alpha_rate": diagnostics["late_alpha_rate"],
                "poisson_alpha_rate": diagnostics["poisson_alpha_rate"],
                "renewal_rate": params.renewal_rate,
                "renewal_delay": params.renewal_delay,
                "lambda_tau_delay": params.renewal_rate * params.renewal_delay,
                "jump_to_cage_variance": params.jump_variance / params.cage_variance,
            }
        )
    return rows


def glass_phenomenon_audit(
    temperatures: np.ndarray,
    cage_law: TemperatureLawParams,
    exchange_law: FacilitatedExchangeLawParams,
    *,
    wave_number: float,
    threshold: float = math.exp(-1.0),
) -> dict[str, float | list[dict[str, float]]]:
    """Audit which glassy dynamical signatures follow from one parameter law.

    The audit deliberately distinguishes dynamical consequences of the
    delayed-renewal cage model from phenomena it does not derive. In particular,
    a thermodynamic transition is marked unsupported: the model supplies an
    effective renewal clock, not an equilibrium singularity.
    """

    temperatures = np.asarray(temperatures, dtype=float)
    if temperatures.ndim != 1 or temperatures.size < 2:
        raise ValueError("temperatures must contain at least two points")
    if np.any(temperatures <= 0.0):
        raise ValueError("temperatures must be positive")
    _validate_temperature_law(cage_law)
    _validate_facilitated_exchange_law(exchange_law)
    if wave_number <= 0.0:
        raise ValueError("wave_number must be positive")
    if not math.isclose(cage_law.reference_temperature, exchange_law.reference_temperature):
        raise ValueError("cage and exchange laws must share a reference temperature")

    rows: list[dict[str, float]] = []
    baseline_se_product = None
    for temperature in temperatures:
        params = temperature_dependent_params(float(temperature), cage_law)
        heterogeneity = temperature_dependent_gamma_exchange(float(temperature), exchange_law)
        diffusion = long_time_diffusion_coefficient(params)
        tau_alpha_poisson = alpha_relaxation_time(wave_number, params, threshold=threshold)
        tau_alpha_exchange = gamma_exchange_alpha_relaxation_time(
            wave_number,
            params,
            heterogeneity,
            threshold=threshold,
        )
        se_product = diffusion * tau_alpha_exchange
        if baseline_se_product is None:
            baseline_se_product = se_product
        peak = dimensionless_peak_prediction(params)
        asymptotic = gamma_exchange_asymptotic_diagnostics(wave_number, params, heterogeneity)

        time_grid = np.geomspace(
            max(1e-4, 0.02 * params.cage_tau),
            max(20.0 * tau_alpha_exchange, 20.0 * params.renewal_delay, 20.0 * params.cage_tau),
            700,
        )
        decay = gamma_exchange_normalized_alpha_decay(wave_number, time_grid, params, heterogeneity)
        local_beta = local_alpha_stretching_exponent(time_grid, decay)
        alpha_window = (-np.log(decay) > 0.3) & (-np.log(decay) < 2.0) & np.isfinite(local_beta)
        if np.any(alpha_window):
            median_beta = float(np.nanmedian(local_beta[alpha_window]))
        else:
            median_beta = math.nan
        chi4_peak = float(np.max(gamma_exchange_scattering_susceptibility(wave_number, time_grid, params, heterogeneity)))
        late_time = 20.0 * tau_alpha_exchange
        later_time = 60.0 * tau_alpha_exchange
        late_ngp = float(gamma_exchange_ngp_1d(np.array([late_time]), params, heterogeneity)[0])
        later_ngp = float(gamma_exchange_ngp_1d(np.array([later_time]), params, heterogeneity)[0])

        rows.append(
            {
                "temperature": float(temperature),
                "inverse_temperature_shift": 1.0 / float(temperature) - 1.0 / cage_law.reference_temperature,
                "diffusion_coefficient": diffusion,
                "tau_alpha_poisson": tau_alpha_poisson,
                "tau_alpha_exchange": tau_alpha_exchange,
                "normalized_stokes_einstein_product": se_product / baseline_se_product,
                "ngp_peak_time": peak["peak_time"],
                "ngp_peak_height": peak["peak_ngp"],
                "heterogeneity_ratio": asymptotic["heterogeneity_ratio"],
                "late_ngp_renewal_amplitude": asymptotic["late_ngp_renewal_amplitude"],
                "alpha_rate_renormalization": asymptotic["alpha_rate_renormalization"],
                "median_alpha_window_beta": median_beta,
                "chi4_peak": chi4_peak,
                "late_ngp": late_ngp,
                "later_ngp": later_ngp,
                "gaussian_recovery_ratio": later_ngp / late_ngp,
            }
        )

    diffusion = np.array([row["diffusion_coefficient"] for row in rows])
    tau_exchange = np.array([row["tau_alpha_exchange"] for row in rows])
    peak_times = np.array([row["ngp_peak_time"] for row in rows])
    se_products = np.array([row["normalized_stokes_einstein_product"] for row in rows])
    heterogeneity_ratios = np.array([row["heterogeneity_ratio"] for row in rows])
    beta_values = np.array([row["median_alpha_window_beta"] for row in rows])
    chi4_peaks = np.array([row["chi4_peak"] for row in rows])
    recovery_ratios = np.array([row["gaussian_recovery_ratio"] for row in rows])
    activation_energies = apparent_alpha_activation_energies(temperatures, tau_exchange)
    fragility_indices = activation_energies / (temperatures * math.log(10.0))
    for row, activation, fragility in zip(rows, activation_energies, fragility_indices):
        row["apparent_alpha_activation_energy"] = float(activation)
        row["local_fragility_index"] = float(fragility)

    flags = {
        "diffusion_slowdown": float(np.all(np.diff(diffusion) < 0.0)),
        "alpha_slowdown": float(np.all(np.diff(tau_exchange) > 0.0)),
        "ngp_peak_shift": float(np.all(np.diff(peak_times) > 0.0)),
        "stokes_einstein_violation": float(se_products[-1] > 1.5 * se_products[0]),
        "fragility_growth": float(fragility_indices[-1] > fragility_indices[0]),
        "heterogeneity_growth": float(np.all(np.diff(heterogeneity_ratios) > 0.0)),
        "stretched_alpha_window": float(np.nanmin(beta_values) < 0.9),
        "chi4_peak_growth": float(chi4_peaks[-1] > chi4_peaks[0]),
        "gaussian_recovery": float(np.all(recovery_ratios < 1.0)),
        "thermodynamic_transition": 0.0,
    }
    dynamic_keys = [
        "diffusion_slowdown",
        "alpha_slowdown",
        "ngp_peak_shift",
        "stokes_einstein_violation",
        "fragility_growth",
        "heterogeneity_growth",
        "stretched_alpha_window",
        "chi4_peak_growth",
        "gaussian_recovery",
    ]
    return {
        **flags,
        "supported_dynamic_signatures": float(sum(flags[key] for key in dynamic_keys)),
        "tested_dynamic_signatures": float(len(dynamic_keys)),
        "rows": rows,
    }


def glass_signature_phase_diagram(
    temperatures: np.ndarray,
    base_cage_law: TemperatureLawParams,
    base_exchange_law: FacilitatedExchangeLawParams,
    *,
    wave_number: float,
    delay_barrier_gaps: list[float],
    exchange_barrier_sums: list[float],
) -> list[dict[str, float]]:
    """Scan barrier controls for simultaneous glass-dynamics signatures.

    ``delay_barrier_gap`` is ``E_d-E_lambda`` and controls growth of
    ``lambda*tau_d``. ``exchange_barrier_sum`` is the total cooling barrier that
    broadens and slows mobility exchange, controlling growth of
    ``R_x/kappa_0``. A grid point passes the joint dynamic-signature criterion
    only when all audited dynamical signatures are supported while the thermodynamic
    transition flag remains zero.
    """

    _validate_temperature_law(base_cage_law)
    _validate_facilitated_exchange_law(base_exchange_law)
    if not delay_barrier_gaps:
        raise ValueError("delay_barrier_gaps must not be empty")
    if not exchange_barrier_sums:
        raise ValueError("exchange_barrier_sums must not be empty")
    if any(gap < 0.0 for gap in delay_barrier_gaps):
        raise ValueError("delay_barrier_gaps must be nonnegative")
    if any(total < 0.0 for total in exchange_barrier_sums):
        raise ValueError("exchange_barrier_sums must be nonnegative")
    base_exchange_sum = base_exchange_law.shape_broadening_barrier + base_exchange_law.exchange_slowing_barrier
    if base_exchange_sum > 0.0:
        shape_fraction = base_exchange_law.shape_broadening_barrier / base_exchange_sum
    else:
        shape_fraction = 0.5

    rows: list[dict[str, float]] = []
    for delay_gap in delay_barrier_gaps:
        cage_law = TemperatureLawParams(
            reference_temperature=base_cage_law.reference_temperature,
            cage_variance_ref=base_cage_law.cage_variance_ref,
            cage_tau_ref=base_cage_law.cage_tau_ref,
            jump_to_cage_ref=base_cage_law.jump_to_cage_ref,
            renewal_rate_ref=base_cage_law.renewal_rate_ref,
            renewal_delay_ref=base_cage_law.renewal_delay_ref,
            rate_activation=base_cage_law.rate_activation,
            delay_activation=base_cage_law.rate_activation + delay_gap,
            cage_stiffening=base_cage_law.cage_stiffening,
            jump_to_cage_growth=base_cage_law.jump_to_cage_growth,
            cage_tau_activation=base_cage_law.cage_tau_activation,
        )
        for exchange_sum in exchange_barrier_sums:
            exchange_law = FacilitatedExchangeLawParams(
                reference_temperature=base_exchange_law.reference_temperature,
                shape_ref=base_exchange_law.shape_ref,
                exchange_renewal_count_ref=base_exchange_law.exchange_renewal_count_ref,
                shape_broadening_barrier=shape_fraction * exchange_sum,
                exchange_slowing_barrier=(1.0 - shape_fraction) * exchange_sum,
            )
            audit = glass_phenomenon_audit(
                temperatures,
                cage_law,
                exchange_law,
                wave_number=wave_number,
            )
            audit_rows = audit["rows"]
            cold = audit_rows[-1]
            hot = audit_rows[0]
            complete = (
                audit["supported_dynamic_signatures"] == audit["tested_dynamic_signatures"]
                and audit["thermodynamic_transition"] == 0.0
            )
            rows.append(
                {
                    "delay_barrier_gap": float(delay_gap),
                    "exchange_barrier_sum": float(exchange_sum),
                    "complete_dynamic_closure": 1.0 if complete else 0.0,
                    "supported_dynamic_signatures": float(audit["supported_dynamic_signatures"]),
                    "tested_dynamic_signatures": float(audit["tested_dynamic_signatures"]),
                    "diffusion_slowdown": float(audit["diffusion_slowdown"]),
                    "alpha_slowdown": float(audit["alpha_slowdown"]),
                    "ngp_peak_shift": float(audit["ngp_peak_shift"]),
                    "stokes_einstein_violation": float(audit["stokes_einstein_violation"]),
                    "fragility_growth": float(audit["fragility_growth"]),
                    "heterogeneity_growth": float(audit["heterogeneity_growth"]),
                    "stretched_alpha_window": float(audit["stretched_alpha_window"]),
                    "chi4_peak_growth": float(audit["chi4_peak_growth"]),
                    "gaussian_recovery": float(audit["gaussian_recovery"]),
                    "thermodynamic_transition": float(audit["thermodynamic_transition"]),
                    "cold_tau_alpha_ratio": cold["tau_alpha_exchange"] / hot["tau_alpha_exchange"],
                    "cold_se_product_ratio": cold["normalized_stokes_einstein_product"],
                    "cold_heterogeneity_growth_ratio": cold["heterogeneity_ratio"] / hot["heterogeneity_ratio"],
                    "cold_chi4_peak_ratio": cold["chi4_peak"] / hot["chi4_peak"],
                    "minimum_alpha_window_beta": min(row["median_alpha_window_beta"] for row in audit_rows),
                }
            )
    return rows


def _cooling_inverse_temperature_interval(hot_temperature: float, cold_temperature: float) -> float:
    if hot_temperature <= 0.0 or cold_temperature <= 0.0:
        raise ValueError("temperatures must be positive")
    interval = 1.0 / cold_temperature - 1.0 / hot_temperature
    if interval <= 0.0:
        raise ValueError("cold_temperature must be lower than hot_temperature")
    return interval


def barrier_amplification_laws(
    *,
    hot_temperature: float,
    cold_temperature: float,
    delay_barrier_gap: float,
    exchange_barrier_sum: float,
) -> dict[str, float]:
    """Closed cooling amplification laws for barrier-controlled diagnostics."""

    if delay_barrier_gap < 0.0:
        raise ValueError("delay_barrier_gap must be nonnegative")
    if exchange_barrier_sum < 0.0:
        raise ValueError("exchange_barrier_sum must be nonnegative")
    interval = _cooling_inverse_temperature_interval(hot_temperature, cold_temperature)
    lambda_tau_growth = math.exp(delay_barrier_gap * interval)
    heterogeneity_growth = math.exp(exchange_barrier_sum * interval)
    return {
        "hot_temperature": hot_temperature,
        "cold_temperature": cold_temperature,
        "inverse_temperature_interval": interval,
        "delay_barrier_gap": delay_barrier_gap,
        "exchange_barrier_sum": exchange_barrier_sum,
        "lambda_tau_delay_growth": lambda_tau_growth,
        "heterogeneity_ratio_growth": heterogeneity_growth,
        "combined_slowing_growth": lambda_tau_growth * heterogeneity_growth,
    }


def minimal_barrier_requirements(
    *,
    hot_temperature: float,
    cold_temperature: float,
    target_lambda_tau_delay_growth: float,
    target_heterogeneity_ratio_growth: float,
) -> dict[str, float]:
    """Invert target cooling amplifications into minimum barrier controls."""

    if target_lambda_tau_delay_growth < 1.0:
        raise ValueError("target_lambda_tau_delay_growth must be at least one")
    if target_heterogeneity_ratio_growth < 1.0:
        raise ValueError("target_heterogeneity_ratio_growth must be at least one")
    interval = _cooling_inverse_temperature_interval(hot_temperature, cold_temperature)
    required_delay_gap = math.log(target_lambda_tau_delay_growth) / interval
    required_exchange_sum = math.log(target_heterogeneity_ratio_growth) / interval
    return {
        "hot_temperature": hot_temperature,
        "cold_temperature": cold_temperature,
        "inverse_temperature_interval": interval,
        "target_lambda_tau_delay_growth": target_lambda_tau_delay_growth,
        "target_heterogeneity_ratio_growth": target_heterogeneity_ratio_growth,
        "target_combined_growth": target_lambda_tau_delay_growth * target_heterogeneity_ratio_growth,
        "required_delay_barrier_gap": required_delay_gap,
        "required_exchange_barrier_sum": required_exchange_sum,
        "required_combined_barrier": required_delay_gap + required_exchange_sum,
    }


def poisson_weights(mean: float, *, max_count: int) -> np.ndarray:
    """Stable Poisson weights from n=0 to max_count with tail folded into max_count."""

    if mean < 0.0:
        raise ValueError("mean must be nonnegative")
    if max_count < 1:
        raise ValueError("max_count must be at least 1")
    weights = np.zeros(max_count + 1, dtype=float)
    weights[0] = math.exp(-mean)
    for idx in range(max_count):
        weights[idx + 1] = weights[idx] * mean / (idx + 1)
    tail = max(0.0, 1.0 - float(np.sum(weights)))
    weights[-1] += tail
    return weights


def radial_van_hove_3d(
    radius: np.ndarray,
    *,
    time: float,
    params: DelayedRenewalCageParams,
    max_count: int = 80,
) -> np.ndarray:
    """Radial van Hove density for |Delta r| in the 3D renewal cage model."""

    _validate(params)
    radius = np.asarray(radius, dtype=float)
    if np.any(radius < 0.0):
        raise ValueError("radius values must be nonnegative")
    if time < 0.0:
        raise ValueError("time must be nonnegative")

    local = float(local_cage_variance(np.array([time]), params)[0])
    renewal = float(delayed_poisson_mean(np.array([time]), params)[0])
    weights = poisson_weights(renewal, max_count=max_count)
    density = np.zeros_like(radius)

    for count, weight in enumerate(weights):
        variance = local + count * params.jump_variance
        if variance <= 0.0:
            continue
        prefactor = math.sqrt(2.0 / math.pi) / (variance ** 1.5)
        density += weight * prefactor * radius**2 * np.exp(-(radius**2) / (2.0 * variance))

    return density


def gaussian_radial_3d(radius: np.ndarray, *, coordinate_variance: float) -> np.ndarray:
    """Radial density for an isotropic 3D Gaussian with per-coordinate variance."""

    radius = np.asarray(radius, dtype=float)
    if np.any(radius < 0.0):
        raise ValueError("radius values must be nonnegative")
    if coordinate_variance <= 0.0:
        raise ValueError("coordinate_variance must be positive")
    prefactor = math.sqrt(2.0 / math.pi) / (coordinate_variance**1.5)
    return prefactor * radius**2 * np.exp(-(radius**2) / (2.0 * coordinate_variance))


def _find_time_for_renewal_count(target: float, params: DelayedRenewalCageParams) -> float:
    low = 0.0
    high = max(params.renewal_delay, target / params.renewal_rate + 2.0 * params.renewal_delay)
    while delayed_poisson_mean(np.array([high]), params)[0] < target:
        high *= 2.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if delayed_poisson_mean(np.array([mid]), params)[0] < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def dimensionless_peak_prediction(params: DelayedRenewalCageParams) -> dict[str, float]:
    """Plateau-regime peak estimate q R(t*) = A and alpha*=q/(4A)."""

    _validate(params)
    target_renewal = params.cage_variance / params.jump_variance
    return {
        "target_renewal_count": target_renewal,
        "peak_time": _find_time_for_renewal_count(target_renewal, params),
        "peak_ngp": params.jump_variance / (4.0 * params.cage_variance),
    }


def generalized_delay_ngp_short_time(
    params: DelayedRenewalCageParams,
    *,
    delay_exponent: float,
) -> dict[str, float]:
    """Short-time NGP law for r(t)=lambda[1-exp(-t/tau_d)]^m.

    Since L(t) ~ A t/tau_c and R(t) ~ lambda t^(m+1)/[(m+1) tau_d^m],
    alpha_2(t) ~ C t^(m-1). The default model uses m=2, the smallest integer
    exponent for which alpha_2 starts continuously from zero.
    """

    _validate(params)
    if delay_exponent <= 0.0:
        raise ValueError("delay_exponent must be positive")
    prefactor = (
        params.jump_variance**2
        * params.renewal_rate
        * params.cage_tau**2
        / ((delay_exponent + 1.0) * params.cage_variance**2 * params.renewal_delay**delay_exponent)
    )
    return {
        "delay_exponent": delay_exponent,
        "power": delay_exponent - 1.0,
        "prefactor": prefactor,
    }


def classify_delay_exponent(delay_exponent: float) -> str:
    """Classify the NGP origin behavior implied by a delayed renewal exponent."""

    if delay_exponent <= 0.0:
        raise ValueError("delay_exponent must be positive")
    if delay_exponent < 1.0:
        return "singular_origin"
    if math.isclose(delay_exponent, 1.0):
        return "finite_origin"
    return "regular_zero_origin"


def delayed_renewal_shape(scaled_time: float) -> float:
    """Dimensionless delayed-renewal integral F(s) for R(t)=lambda*tau_d*F(t/tau_d)."""

    if scaled_time <= 0.0:
        raise ValueError("scaled_time must be positive")
    return (
        scaled_time
        - 2.0 * (1.0 - math.exp(-scaled_time))
        + 0.5 * (1.0 - math.exp(-2.0 * scaled_time))
    )


def plateau_peak_diagnostics(
    *,
    peak_ngp: float,
    peak_time: float,
    renewal_delay: float,
) -> dict[str, float]:
    """Infer plateau-regime parameters from the observed NGP peak.

    The approximation q R(t*) = A and alpha*(t*) = q/(4A) gives
    q/A = 4 alpha* and R(t*) = 1/(4 alpha*). If the delayed renewal shape is
    assumed and tau_d is known or fit independently, the same peak time estimates
    the asymptotic renewal rate.
    """

    if peak_ngp <= 0.0:
        raise ValueError("peak_ngp must be positive")
    if peak_time <= 0.0:
        raise ValueError("peak_time must be positive")
    if renewal_delay <= 0.0:
        raise ValueError("renewal_delay must be positive")

    jump_to_cage_variance = 4.0 * peak_ngp
    target_renewal_count = 1.0 / jump_to_cage_variance
    scaled_peak_time = peak_time / renewal_delay
    shape = delayed_renewal_shape(scaled_peak_time)
    renewal_rate = target_renewal_count / (renewal_delay * shape)
    return {
        "jump_to_cage_variance": jump_to_cage_variance,
        "target_renewal_count": target_renewal_count,
        "scaled_peak_time": scaled_peak_time,
        "renewal_shape": shape,
        "renewal_rate": renewal_rate,
        "renewal_rate_times_peak_time": renewal_rate * peak_time,
    }


def plateau_ngp_branches(
    *,
    jump_to_cage_variance: float,
    observed_ngp: float,
) -> dict[str, float]:
    """Invert alpha = beta y/(1+y)^2 into its early and late branches.

    Here beta=q/A and y=beta R in the plateau approximation. The admissible
    range is 0 < alpha <= beta/4. Values below the maximum have two roots:
    an early branch y<1 before the NGP peak and a late branch y>1 after it.
    """

    if jump_to_cage_variance <= 0.0:
        raise ValueError("jump_to_cage_variance must be positive")
    if observed_ngp <= 0.0:
        raise ValueError("observed_ngp must be positive")
    beta = jump_to_cage_variance
    discriminant = beta * beta - 4.0 * beta * observed_ngp
    tolerance = 1e-14 * beta * beta
    if discriminant < -tolerance:
        raise ValueError("observed_ngp exceeds the plateau peak bound")
    discriminant = max(0.0, discriminant)
    root = math.sqrt(discriminant)
    early_y = (beta - 2.0 * observed_ngp - root) / (2.0 * observed_ngp)
    late_y = (beta - 2.0 * observed_ngp + root) / (2.0 * observed_ngp)
    return {
        "early_y": early_y,
        "late_y": late_y,
        "peak_y": 1.0,
        "peak_ngp_bound": beta / 4.0,
    }


def observable_consistency_diagnostics(
    *,
    peak_ngp: float,
    peak_time: float,
    renewal_delay: float,
    late_time: float,
    late_ngp: float,
) -> dict[str, float]:
    """Compare peak-inferred and long-time-inferred renewal rates.

    The plateau peak gives q/A and lambda from alpha_2(t*) and t*. A later NGP
    value gives two independent checks: the simple asymptotic estimate
    1/[t alpha_2(t)] and a finite-time correction obtained by inverting the
    exact plateau NGP formula alpha = beta y/(1+y)^2 with beta=q/A and y=beta R.
    Their logarithmic ratios are data-level falsification residuals.
    """

    if late_time <= 0.0:
        raise ValueError("late_time must be positive")
    if late_ngp <= 0.0:
        raise ValueError("late_ngp must be positive")
    peak = plateau_peak_diagnostics(
        peak_ngp=peak_ngp,
        peak_time=peak_time,
        renewal_delay=renewal_delay,
    )
    beta = peak["jump_to_cage_variance"]
    branches = plateau_ngp_branches(jump_to_cage_variance=beta, observed_ngp=late_ngp)
    late_branch_y = branches["late_y"]
    late_renewal_count = late_branch_y / beta
    late_shape = delayed_renewal_shape(late_time / renewal_delay)
    late_renewal_rate_exact = late_renewal_count / (renewal_delay * late_shape)
    late_renewal_rate_asymptotic = 1.0 / (late_time * late_ngp)
    exact_rate_ratio = late_renewal_rate_exact / peak["renewal_rate"]
    asymptotic_rate_ratio = late_renewal_rate_asymptotic / peak["renewal_rate"]
    return {
        "jump_to_cage_variance": peak["jump_to_cage_variance"],
        "target_renewal_count": peak["target_renewal_count"],
        "peak_renewal_rate": peak["renewal_rate"],
        "late_renewal_count": late_renewal_count,
        "late_renewal_rate_exact": late_renewal_rate_exact,
        "late_renewal_rate_asymptotic": late_renewal_rate_asymptotic,
        "exact_rate_ratio": exact_rate_ratio,
        "asymptotic_rate_ratio": asymptotic_rate_ratio,
        "log_exact_rate_residual": math.log(exact_rate_ratio),
        "log_asymptotic_rate_residual": math.log(asymptotic_rate_ratio),
    }
