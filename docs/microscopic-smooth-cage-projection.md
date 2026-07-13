# Microscopic Smooth Cage Projection

## Result in one sentence

A smooth force-support cage coordinate extracted from the many-particle C3
Kob--Andersen Langevin state obeys the analytically projected instantaneous
stochastic equation, including its state-dependent noise geometry, on matched
full-particle trajectories. This is not yet an autonomous single-particle GLE
or an event-clock derivation.

## Why a new cage coordinate was needed

Earlier hard-neighbor and frozen-neighbor projections have two different
problems. A hard dynamic graph changes discontinuously when a neighbor crosses
its cutoff. A frozen graph remains differentiable but cannot follow cage
reorganization. Both obstruct a controlled projection from the microscopic
many-particle Langevin dynamics.

The replacement uses the interaction support itself. For tagged particle
`i`, let `d_ij` be the minimum-image vector from `i` to `j`, and let

```text
s_ij = |d_ij| / (2.5 sigma_ij).
```

The compact `wendland_c4` weight is

```text
w(s) = (1-s)^6 (35 s^2 + 18 s + 3),  0 <= s < 1,
w(s) = 0,                              s >= 1.
```

Its support is the fixed C3 KA force support, not a fitted event threshold.
Define

```text
W_i  = sum_j w_ij,
mu_i = sum_j w_ij d_ij / W_i,
C_i  = r_i + mu_i,
u_i  = r_i - C_i = -mu_i.
```

Thus `C_i` is a smooth microscopic cage center and `u_i` is the tagged
particle position relative to it.

## Analytic Jacobian

Set

```text
g_ij = (dw_ij/dr_ij) d_ij/r_ij,
B_ij = [w_ij I + (d_ij-mu_i) g_ij^T] / W_i.
```

The nonzero Jacobian blocks of `u_i(R)` are

```text
J_ii = sum_j B_ij,
J_ij = -B_ij.
```

They satisfy `J_ii + sum_j J_ij = 0`, so the coordinate is exactly
translation invariant. Its cage-relative velocity is

```text
p_i = J_i(R) V = sum_j B_ij (v_i-v_j).
```

Unit tests compare this analytic Jacobian with central finite differences and
check the translation identity and positive-definite noise Gram matrix.

## Exact projected equation

The parent underdamped dynamics are

```text
dR = V dt,
dV = [F(R)-gamma V] dt + sqrt(2 gamma T) dW.
```

Because `R` has finite variation, applying Ito's rule to `u_i(R)` and then to
`p_i=J_iV` gives

```text
du_i = p_i dt,

dp_i = b_i(R,V) dt + sqrt(2 gamma T) J_i(R) dW,

b_i = J_i F + Hess(u_i):(V tensor V) - gamma p_i.
```

The `dp_i` equation is an exact instantaneous coordinate projection. With

```text
G_i = J_i J_i^T,
M_i = G_i^-1,
```

its conditional noise covariance is

```text
Cov[dp_i | R,V]/dt = 2 gamma T G_i.
```

After multiplication by `M_i`, the matrix friction and projected thermostat
noise satisfy the instantaneous matrix FDT, `Cov[M_i J_i dW]/dt = M_i`.
This does not eliminate the hidden many-particle dependence of `b_i`, `G_i`,
or the future orthogonal force.

## Common-noise tangent test

For matched microscopic paths at `+epsilon` and `-epsilon`, driven by the same
Wiener increments, define

```text
delta p = (p+ - p-) / (2 epsilon),
delta b = (b+ - b-) / (2 epsilon),
delta J = (J+ - J-) / (2 epsilon).
```

The linearized projected dynamics predict

```text
d(delta p) = delta b dt + sqrt(2 gamma T) delta J dW,

Cov[d(delta p)-delta b dt | paths]/dt
  = 2 gamma T delta J delta J^T.
```

This term is absent for a constant-J tagged coordinate. The deliberate null
model that sets `delta J=0` therefore predicts zero tangent-noise energy,
whereas the observed residual energy is `3.1991` and the integrated
`delta J` prediction is `3.7827`. The zero-covariance null is rejected.

## Numerical protocol and result

The test used the C3-switched KA parent at `T=0.58`, `gamma=1`, microscopic
step `0.001 tau`, 8 independent common-noise members, and both
`epsilon=0.001` and `0.002`. Every path contains 201 saved frames. The primary
stride is five saved steps, so each epsilon contributes `320/320` valid
intervals.

| Saved-step stride | Physical interval | Trace ratio | Mean squared Mahalanobis | Max absolute whitened lag-1 |
|---:|---:|---:|---:|---:|
| 1 | `0.001 tau` | `0.4997` | `1.542` | `0.497` |
| 2 | `0.002 tau` | `0.7402` | `2.297` | `0.187` |
| 5 | `0.005 tau` | `0.8457` | `2.755` | `0.0405` |

At the preregistered stride 5, both epsilon values pass the covariance trace,
three-dimensional Mahalanobis, and whitened lag-1 gates. Their integrated
covariances agree to relative L2 error `7.98e-5` with correlation above
`0.999999996`. The maximum observed condition number of `G_i` is `2.029`.

The stride-1 and stride-2 failures are retained as part of the result. At the
raw integrator step the finite-step Langevin splitting leaves correlated
innovations, while by `0.005 tau` the measured tangent residual approaches the
continuum projected-SDE covariance. The present evidence therefore validates
the projection on the resolved interval; it does not assert exact finite-step
equality at `0.001 tau`.

No diffusion, NGP, scattering function, event rate, or thermodynamic
observable was used to fit the cage weight or the projected covariance.

## What has and has not been derived

Established at this checkpoint:

1. A differentiable cage coordinate is defined directly from microscopic
   particle positions and fixed pair-interaction parameters.
2. Its Jacobian, projected drift, multiplicative noise, and instantaneous FDT
   geometry follow from the many-particle Langevin equation.
3. Matched microscopic trajectories support the predicted `delta J` tangent
   covariance and reject the constant-noise null.

Still unestablished:

1. `(u_i,p_i,G_i,b_i)` is a sufficient state for future cage escape.
2. First-passage readiness is exponential or the conditioned escape rate is a
   Kramers rate.
3. Eliminating the remaining particles yields a transferable autonomous GLE.
4. The projected process predicts diffusion, NGP, multi-k scattering, or SE
   violation without macro-observable fitting.

The next test must use a fixed nonrecrossing `p_hop` definition on long
trajectories, report censored first-passage survival, and measure leave-one-
parent-out hazard skill from the microscopic projected state. Only a positive
held-out event-clock result warrants estimating a conditional memory kernel.

The machine-readable claim boundary remains:

```text
event_clock_claim_allowed = 0
autonomous_single_particle_gle_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Primary literature context

- Ayaz, Scalfi, Dalton, and Netz, *Generalized Langevin equation with a
  nonlinear potential of mean force and nonlinear memory friction from a
  hybrid projection scheme*, Phys. Rev. E 105, 054138 (2022),
  https://doi.org/10.1103/PhysRevE.105.054138.
- Vroylandt and Monmarche, *Position-dependent memory kernel in generalized
  Langevin equations: Theory and numerical estimation*, J. Chem. Phys. 156,
  244105 (2022), https://doi.org/10.1063/5.0094566.
- Mazoyer, Ebert, Maret, and Keim, *Dynamics of particles and cages in an
  experimental 2D glass former*, Europhys. Lett. 88, 66004 (2009),
  https://doi.org/10.1209/0295-5075/88/66004.
