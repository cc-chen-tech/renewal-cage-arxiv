# Microscopic Gamma Variance-Mixture Diagnostic

## What this step tests

The transport-clock quotient showed that matching calibration and heldout MSD
removes most T=0.45 error at `k=2` and `k=4`, but not all NGP and `k=7.25`
error. The narrow question here is whether one continuous positive scalar
mobility field can generate the remaining displacement shape.

This is a post-run diagnostic: heldout MSD and NGP are diagnostic inputs, not a
blind prediction. The optional cage decomposition also uses heldout `Fs(k=2)`.
No macro parameter is fitted to all three scattering curves.

## Exact variance-mixture identities

Let the displacement conditional on an integrated scalar mobility `A_t` be
isotropic Gaussian:

```text
Delta R | A_t ~ Normal(0, 2 A_t I_3).
```

Then

```text
M(t) = <|Delta R|^2> = 6 <A_t>,
<|Delta R|^4> = 60 <A_t^2>,
alpha_2(t) = 3 <|Delta R|^4> / [5 M(t)^2] - 1
           = Var(A_t) / <A_t>^2,
Fs(k,t) = <exp(-k^2 A_t)>.
```

Thus `Fs` is the Laplace transform of the integrated-mobility distribution.
A linear Gaussian GLE is an exact null: its displacement is Gaussian, so at a
fixed MSD it requires

```text
alpha_2 = 0,
Fs(k) = exp(-k^2 M / 6).
```

No memory kernel in a linear Gaussian GLE can change that fixed-MSD identity.

If `A_t` is gamma distributed with shape `nu` and scale `theta`, then

```text
<A_t> = nu theta,
alpha_2 = 1 / nu,
Fs(k) = (1 + k^2 theta)^(-nu)
      = (1 + alpha_2 k^2 M / 6)^(-1/alpha_2).
```

MSD and NGP therefore determine every wave number without a fit parameter.

## Continuous Langevin origin

Use `m` stationary OU mobility coordinates:

```text
dz_a = -(z_a / tau_D) dt + sqrt[2 Dbar / (m tau_D)] dW_a,
D(t) = sum_(a=1)^m z_a(t)^2,
dR_i = sqrt[2 D(t)] dB_i,  i=1,2,3.
```

At stationarity, each `z_a` is Gaussian with variance `Dbar/m`. Consequently
`D` is gamma distributed with shape `nu=m/2`, mean `Dbar`, and relative
variance `2/m=1/nu`. In the slow-environment limit `tau_D >> t`,
`A_t=int_0^t D(s)ds` approaches `D(0)t`, producing the gamma formula above.
At finite `tau_D/t`, temporal averaging narrows `A_t` and lowers NGP.

Ito's formula closes the mobility coordinates themselves into a positive CIR
diffusivity process:

```text
dD = (2/tau_D)(Dbar-D)dt
   + sqrt[8 Dbar D/(m tau_D)] dW_D.
```

Its stationary Fokker-Planck solution has gamma shape `m/2`, so positivity and
the variance-mixture law follow from the continuous Langevin generator rather
than from an imposed jump clock.

This provides a concrete Markovian single-particle Langevin realization of
fluctuating diffusivity. It is related in spirit to fluctuating-diffusivity GLE
models ([Miyaguchi, Phys. Rev. Research 4, 043062
(2022)](https://doi.org/10.1103/PhysRevResearch.4.043062)) and transient
potential coarse-graining from microscopic overdamped Langevin dynamics
([Uneyama, Phys. Rev. E 101, 032106
(2020)](https://doi.org/10.1103/PhysRevE.101.032106)). The present KA verdict,
however, comes only from the committed trajectory tables and the equations
above.

## Adding a harmonic cage

For a cage OU coordinate,

```text
dy = -(y/tau_c)dt + sqrt(2 D_c)dW,
a_c(t) = D_c tau_c [1-exp(-t/tau_c)].
```

The displacement variance coordinate becomes `B=a_c+G`, where `G` is gamma.
At fixed `mu=M/6` and `v=alpha_2 mu^2`, choosing a cage variance `a_c` fixes
all remaining gamma parameters:

```text
b = mu-a_c,
nu = b^2/v,
theta = v/b,
Fs(k) = exp(-k^2 a_c) (1+k^2 theta)^(-nu).
```

The frozen diagnostic solves `Fs(k=2)` for `a_c in [0,mu]` only if an exact
bracketed root exists, then predicts `k=4` and `k=7.25`. It never extrapolates
outside the physical cage interval.

## Real-trajectory verdict

The maximum normalized errors use the existing absolute `Fs` tolerance 0.03:

| T | model | k=2 | k=4 | k=7.25 |
|---|---:|---:|---:|---:|
| 0.45 | Gaussian null | 1.6620 | 7.2911 | 9.6807 |
| 0.45 | gamma mobility | 0.0774 | 0.6167 | 1.3854 |
| 0.58 | Gaussian null | 0.9866 | 2.4065 | 3.3176 |
| 0.58 | gamma mobility | 0.1130 | 0.1896 | 0.3674 |

The scalar mobility field explains the T=0.45 low/intermediate-k shape but
fails at the cage-scale wave number. This is a localized model failure, not a
failure of fluctuating mobility at all scales.

The shifted cage+gamma model has exact `k=2` roots for 11/21 T=0.45 rows, but
the worst replicate has only 3/7 support, below the frozen 80% requirement.
T=0.58 has 14/25 total roots and a worst-replicate support of 1/5. On supported
T=0.45 rows the maximum normalized `k=7.25` error is 0.8764, but the missing
roots prohibit a closure claim.

T=0.58 remains a canary because its source stationarity gate fails. It cannot
establish a cooling trend.

## Langevin simulation check

The deterministic simulation uses `m=4`, so the slow-limit target is
`alpha_2=0.5`. With 80,000 trajectories and 100 exact OU steps:

| tau_D/t | MSD | NGP | max absolute multi-k Fs error versus slow limit |
|---:|---:|---:|---:|
| 1 | 5.9978 | 0.2841 | 0.0501 |
| 10 | 5.9788 | 0.4618 | 0.0125 |
| 100 | 5.9961 | 0.4974 | 0.00039 |

At `tau_D/t=100`, the predeclared MSD, NGP, and three-wave-number Monte Carlo
tolerances all pass. The simulation therefore validates the microscopic
Langevin-to-gamma limiting step. It does not validate the model against KA at
the failed cage-scale wave number.

## Claim boundary and next model

The evidence supports an exploratory statement: a scalar mobility coordinate
contains real low/intermediate-k shape information, but it is insufficient for
the complete T=0.45 displacement distribution. It does not identify whether
the residual is non-gamma mobility tails, activated cage geometry, path recoil,
or a spatially facilitated field.

The next discriminating model should retain the continuous Langevin variables
but replace the scalar gamma tail with an activated transient potential or a
non-gamma mobility generator, and predict `k=7.25` without using it as an input.
Independent parent trajectories are still required.

```text
blind_prediction_claim_allowed = 0
finite_exchange_resolved = 0
static_environment_resolved = 0
spatial_facilitation_resolved = 0
activated_cage_geometry_resolved = 0
microdynamic_closure_claim_allowed = 0
thermodynamic_claim_allowed = 0
```
