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


def _validate(params: DelayedRenewalCageParams) -> None:
    for name in ("cage_variance", "cage_tau", "jump_variance", "renewal_rate", "renewal_delay"):
        value = getattr(params, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")


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
