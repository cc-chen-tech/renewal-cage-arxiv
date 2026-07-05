from __future__ import annotations

from dataclasses import dataclass
import math

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
