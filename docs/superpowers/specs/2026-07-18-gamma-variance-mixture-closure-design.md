# Gamma Variance-Mixture Langevin Closure Design

## Question

The transport-clock quotient leaves a T=0.45 fixed-MSD residual in NGP and
`Fs(k=7.25)`. This diagnostic asks whether that residual can be represented by
one positive scalar mobility field in a continuous single-particle Langevin
dynamics. It does not add finite exchange, spatial facilitation, or a fitted
hidden state to the effective theory.

## Microscopic candidate

Let `m` independent mobility coordinates obey stationary Ornstein-Uhlenbeck
dynamics,

```text
dz_a = -(z_a / tau_D) dt + sqrt(2 Dbar / (m tau_D)) dW_a,
D(t) = sum_a z_a(t)^2,
dR_i = sqrt(2 D(t)) dB_i,  i=1,2,3.
```

This is a continuous Markovian Langevin system with positive diffusivity. At
stationarity, `D` is gamma distributed with shape `nu=m/2`, mean `Dbar`, and
relative variance `1/nu`. In the slow-environment limit `tau_D >> t`, the
integrated variance is `A_t ~= D t`, so

```text
M = <|Delta R|^2> = 6 <A_t>,
alpha_2 = Var(A_t) / <A_t>^2 = 1 / nu,
Fs(k) = <exp(-k^2 A_t)>
      = (1 + alpha_2 k^2 M / 6)^(-1/alpha_2).
```

The `alpha_2 -> 0` limit is the Gaussian relation `exp(-k^2 M/6)`. Thus a
linear Gaussian GLE is a strict null model, while this scalar-mobility model is
the minimal non-Gaussian extension.

An independent OU cage displacement contributes a deterministic variance
coordinate `a_c(t)=D_c tau_c [1-exp(-t/tau_c)]`. Writing total variance as
`B=a_c+G`, with fixed mean `mu=M/6` and variance
`v=alpha_2 mu^2`, gives

```text
b = mu - a_c,
nu = b^2 / v,
theta = v / b,
Fs(k) = exp(-k^2 a_c) (1 + k^2 theta)^(-nu).
```

For the nested cage diagnostic, infer `a_c in [0,mu]` from observed `Fs(k=2)`
only when an exact bracketed root exists; do not extrapolate. Then predict
`Fs(k=4)` and `Fs(k=7.25)` without another parameter.

## Frozen real-data protocol

Use the committed segment-splice full-path controls only:

- T=0.45: `L=250`, 3 restart labels, 7 lags;
- T=0.58: `L=37`, 5 restart labels, 5 lags.

Select the within-particle model after requiring cross-particle full-path
agreement at absolute `1e-12`. Validate the exact provenance contract: one
parent sample per temperature, distinct restart frames and velocity seeds, and
zero independent parent replicates. Require finite positive MSD and NGP and all
source claim flags closed.

The total-gamma prediction uses heldout MSD and heldout NGP as diagnostic
inputs. The cage diagnostic additionally uses heldout `Fs(k=2)`. Neither is a
blind heldout prediction. Scattering errors retain the frozen absolute
tolerance `0.03`.

## Decision rules

Report per-temperature maxima for the Gaussian null and total-gamma closure at
`k={2,4,7.25}`. `scalar_mobility_shape_closure_supported_exploratory=1` requires
all three total-gamma maxima to be at most one tolerance unit, exact provenance,
and source stationarity. T=0.58 remains a canary even if its shape metrics pass.

The cage diagnostic requires an exact `k=2` root for at least 80% of rows in
every replicate before any cage-plus-mobility support can be set. Below that
coverage, report its supported-row errors descriptively and fail closed.

Keep all of these zero:

```text
blind_prediction_claim_allowed
finite_exchange_resolved
static_environment_resolved
spatial_facilitation_resolved
activated_cage_geometry_resolved
microdynamic_closure_claim_allowed
thermodynamic_claim_allowed
```

## Langevin simulation validation

Simulate the stationary squared-OU mobility system in three dimensions for
`m=4` (`nu=2`, target `alpha_2=0.5`) using exact OU updates and endpoint
sampling conditional on the integrated variance. Use fixed seeds and compare
`tau_D/t={1,10,100}`. The `tau_D/t=100` row must reproduce analytic MSD, NGP,
and `Fs(k={0.5,1,2})` within predeclared Monte Carlo tolerances. The other rows
show finite-environment averaging and are not fitted to KA data.

## Outputs

- `scripts/summarize_ka_gamma_variance_mixture.py`
- `tests/test_ka_gamma_variance_mixture.py`
- `data/renewal_cage_ka_gamma_variance_mixture_rows.csv`
- `data/renewal_cage_ka_gamma_variance_mixture_gate.csv`
- `data/renewal_cage_gamma_variance_mixture_langevin_validation.csv`
- `figures/renewal_cage_ka_gamma_variance_mixture.svg`
- `docs/microscopic-gamma-variance-mixture.md`

