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


def _validate_facilitated_exchange_law(law: FacilitatedExchangeLawParams) -> None:
    for name in ("reference_temperature", "shape_ref", "exchange_renewal_count_ref"):
        value = getattr(law, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("shape_broadening_barrier", "exchange_slowing_barrier"):
        value = getattr(law, name)
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")


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
    ``R_x/kappa_0``. A grid point has complete dynamic closure only when all
    audited dynamical signatures are supported while the thermodynamic
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
