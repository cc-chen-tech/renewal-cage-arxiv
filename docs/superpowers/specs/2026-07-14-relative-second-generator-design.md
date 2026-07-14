# Relative Second-Generator Design

## Objective

Test whether the non-Gaussian residual left by the microscopic smooth-cage
state `(u,p,Lp)` is caused by one omitted instantaneous generator coordinate,
`L2p`, rather than by an empirical non-Gaussian kick distribution.

The input remains the full 4096-particle Kob-Andersen (KA) underdamped
Langevin trajectory. No diffusion, scattering, event, or thermodynamic
observable is fitted.

## Microscopic Identity

For unit mass,

```text
dR = V dt
dV = [F(R) - gamma V] dt + sqrt(2 gamma T) dW,

L = V . grad_R + [F - gamma V] . grad_V + gamma T Delta_V.
```

Let `u(R)` be the differentiable force-support cage-relative coordinate,
`J = grad_R u`, and `p = Lu = J V`. Its first generator is

```text
b = Lp = J[F - gamma V] + H_u[V,V].
```

The next coordinate is evaluated from the generator identity

```text
c = L2p
  = D_R b[V] + D_V b[F - gamma V] + gamma T Delta_V b.
```

Because only `H_u[V,V]` is quadratic in velocity,

```text
Delta_V b = 2 Delta_R u.
```

The conservative-force tangent entering `D_R b[V]` is not estimated from a
trajectory history:

```text
LF = (V . grad_R)F = -H_U V.
```

It is computed from the same C3-switched KA pair Hessian as the force.

## Numerical Realization

### Sparse force-Hessian action

The dense reference implementation forms every selected-particle pair block.
For repeated full-trajectory evaluation, add a periodic linked-cell
implementation that visits only cells intersecting the largest KA cutoff.
It must match the dense force and `LF` outputs on random configurations and
both KA species before it is used for scientific data.

### Phase-space directional derivative

At a saved microscopic state `(R,V,F)`, with `A=F-gamma V`, evaluate

```text
D_R b[V] approximately
  [b(R+h_R V, V, F+h_R LF) - b(R-h_R V, V, F-h_R LF)]/(2 h_R),

D_V b[A] approximately
  [b(R, V+h_V A, F) - b(R, V-h_V A, F)]/(2 h_V).
```

The force linearization is sufficient for a centered first directional
derivative; the remaining truncation error is second order in `h_R`.

### Ito trace

Because the velocity dependence of `b` is exactly quadratic, avoid a second
difference of nearly equal drift values. For fixed Rademacher trace probes,
evaluate the algebraically equivalent Hessian action:

```text
H_u[z_q,z_q] approximately
  [J(R+h_I z_q)z_q - J(R-h_I z_q)z_q]/(2 h_I),

Delta_V b approximately 2 mean_q H_u[z_q,z_q],
E[z_q z_q^T] = I.
```

The probe seed is fixed before validation. Discovery reports convergence with
probe counts `4,8,16,32` and at least three finite-difference steps. The final
validation uses one frozen probe count and step; it does not select them from
held-clone closure scores.

On the first real-frame canary, varying the phase-space step from `3e-6` to
`3e-5` and the cage directional step from `5e-6` to `2e-5` changed `L2p` by
at most `4.6e-8` in relative L2 norm. The Ito term had RMS `1.29`, compared
with total `L2p` RMS `532.11`; it is retained for generator correctness even
though it is not the dominant contribution at this state point.

## Preliminary Proxy No-Go

A causal finite-difference proxy

```text
[Lp(t)-Lp(t-w)]/(w Delta t),  w in {1,2,4,8,16,32},
```

did not reduce the held `Lp` residual excess kurtosis or squared-residual
correlation. On the two independent validation clones, the baseline maximum
absolute `Lp` kurtosis was `1.9409`; all proxy windows remained at or above
`1.9329`, and the maximum squared-residual correlation through 40 frames
remained at or above `0.2292` versus `0.2304` at baseline.

This proxy is algebraically contained in the existing 40-frame `Lp` history,
so the negative result rules out another history derivative as a new state.
It does not rule out the instantaneous configuration observable `L2p`.

## Validation Protocol

1. Unit-test the sparse Hessian action against the dense KA implementation.
2. On small systems, compare `L2p` with the short-time conditional definition
   `E[b(X_dt)-b(X_0)]/dt` and verify finite-difference convergence.
3. Use four discovery clones to choose only numerical step and probe controls.
4. Freeze those controls and evaluate the two independent validation clones.
5. Fit the same discrete Mori memory order `40` and VAR bath order `16` used by
   the `(u,p,Lp)` baseline, now on `(u,p,Lp,L2p)`.

The extension is supported only if every validation clone satisfies:

```text
VAR spectral radius < 1
maximum white-residual correlation through 40 frames <= 0.05
maximum squared-white-residual correlation through 40 frames <= 0.05
maximum absolute white-residual excess kurtosis <= 0.35
stationary covariance maximum error <= 0.25
target (u,p) correlation maximum error <= 0.25.
```

The comparison against `(u,p,Lp)` uses identical targets, time windows,
normalization, memory order, VAR order, and Monte Carlo seed sets.

## Claim Boundary

Passing establishes a finite discrete microscopic Markov embedding for the
tested smooth cage-relative state at `T=0.58`. It does not establish a
continuous Gaussian thermal bath, Kramers escape, a renewal event clock,
cross-temperature transfer, or a thermodynamic glass transition. The
following remain false until separately demonstrated:

```text
continuous_gaussian_langevin_bath_allowed
kramers_escape_claim_allowed
complete_event_clock_closure_allowed
thermodynamic_claim_allowed
```
