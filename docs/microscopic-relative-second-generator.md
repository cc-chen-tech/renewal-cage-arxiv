# Microscopic Relative Second-Generator Checkpoint

## Question

The smooth cage projection previously left a stable but non-Gaussian white
driving term after a memory-40 Mori model and an order-16 Markov bath.  This
checkpoint tests a specific microscopic explanation: the bath may look
non-Gaussian because the resolved state `(u,p,Lp)` omitted the instantaneous
second generator coordinate `L2p`.

This is a discovery canary, not an independent C3 validation.  It uses four
existing `ka_lj_cut` trajectories, 200 frames per clone, the same fixed target
particles, and `trace_probe_count = 4`.  No diffusion, scattering, event, or
thermodynamic observable is fitted.

## Exact Generator Hierarchy

For the full many-particle underdamped Langevin state `X=(R,V)`, with unit
mass,

```text
dR = V dt
dV = [F(R)-gamma V] dt + sqrt(2 gamma T) dW,

L = V.grad_R + [F-gamma V].grad_V + gamma T Delta_V.
```

Let `u(R)` be the differentiable force-support cage-relative coordinate and
`J=grad_R u`.  The first two projected generator coordinates are

```text
p = Lu = J V,

b = Lp = J[F-gamma V] + H_u[V,V],

c = L2p
  = D_R b[V] + D_V b[F-gamma V] + gamma T Delta_V b.
```

Only the Hessian term in `b` is quadratic in velocity, so

```text
Delta_V b = 2 Delta_R u.
```

The force tangent in `D_R b[V]` is computed from the microscopic KA pair
Hessian,

```text
LF = (V.grad_R)F = -H_U V,
```

rather than estimated from neighboring saved frames.  Thus `L2p` contains new
instantaneous configuration information that cannot be reconstructed from a
finite difference of the existing `Lp` history.

The next exact stochastic equation also shows why adding `L2p` need not end
the hierarchy:

```text
dc = L3p dt + sqrt(2 gamma T) [grad_V(L2p)] dW.
```

Even though the full-state driving `dW` is Gaussian, the projected diffusion
matrix of `c` is state dependent.  Ignoring `grad_V(L2p)` can therefore leave
a correlated scale mixture in a reduced-state residual.

## Numerical Construction

The implementation has three independently checked parts.

1. A periodic linked-cell KA evaluator returns the conservative force and
   `LF` for all particles without forming the dense pair list.  Random-system
   tests match the dense implementation for both `ka_lj_cut` and the C3
   switched protocol.
2. Centered phase-space directional derivatives evaluate `D_R b[V]` and
   `D_V b[F-gamma V]`.
3. Fixed antithetic Rademacher probes evaluate the Ito trace through cage
   Hessian actions rather than subtracting nearly equal second derivatives.

On the real-frame convergence canary, changing the phase-space step over
`3e-6` to `3e-5` and the cage directional step over `5e-6` to `2e-5` changed
`L2p` by at most `3.35e-8` in relative L2 norm.  Four discovery caches are
complete at `200/200` frames.  The maximum absolute reconstructed projected
force mismatch against the older drift cache is `0.02577`; the raw LAMMPS
`fx` column is not used as the conservative force because `fix langevin`
contributes thermostat forces to that dump.

The four-probe canary is sufficient to decide whether the coordinate has any
signal.  It is not the preregistered `4,8,16,32` probe-convergence study, so it
cannot authorize a final C3 validation claim.

## Matched Closure Test

For every leave-one-clone-out fold, the baseline and extension use identical
targets, frames, training-only cage bias, training-only normalization, Mori
memory order 40, VAR bath order 16, lag window 40, simulation count 2000, and
random seed schedule:

```text
baseline:  g = (u-bias, p, Lp)
extension: g = (u-bias, p, Lp, L2p).
```

Drift and second-generator caches are paired by trajectory SHA256, not by
file name.  The held residual diagnostic separately measures ordinary
correlation, squared-residual correlation, and marginal excess kurtosis.
This distinguishes an apparently white residual from one with hidden
volatility memory.

The frozen Gaussian gates are

```text
VAR spectral radius < 1
maximum residual correlation <= 0.05
maximum squared-residual correlation <= 0.05
maximum absolute excess kurtosis <= 0.35
stationary covariance maximum error <= 0.25
target (u,p) correlation maximum error <= 0.25.
```

## Discovery Result

Adding `L2p` improves the old `Lp` residual in every held fold:

| held metric | `(u,p,Lp)` | `(u,p,Lp,L2p)` |
|---|---:|---:|
| worst `Lp` squared-residual correlation | 0.32917 | 0.19946 |
| worst absolute `Lp` excess kurtosis | 3.17998 | 1.68713 |
| worst ordinary residual correlation, all coordinates | 0.04073 | 0.02814 |
| worst stationary covariance error | 0.20260 | 0.20142 |
| worst target correlation error | 0.12718 | 0.15710 |

This is positive microscopic evidence that `L2p` carries omitted information.
It is not a Gaussian closure.  The newly resolved coordinate has the largest
remaining defects:

```text
worst squared residual correlation including L2p = 0.37876
worst absolute residual excess kurtosis including L2p = 5.02707.
```

Both exceed the preregistered limits by large factors.  The finite second
generator truncation moves much of the non-Gaussianity from `Lp` into `L2p`
instead of eliminating it.

The reduced tables therefore record

```text
l2p_improves_lp_shape_on_aggregate = 1
finite_discrete_gaussian_l2p_closure_supported = 0
continuous_gaussian_langevin_bath_allowed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Physical Verdict And Next Exact Object

The result supports a microscopic generator hierarchy, not a finite
constant-noise Langevin closure at second order.  It also explains why a
history derivative proxy failed earlier: a history derivative stays inside
the old memory span, whereas `L2p` adds instantaneous force-curvature and cage
geometry information.

The next principled test is not to fit another arbitrary latent clock.  It is
to derive and evaluate the conditional diffusion of the `L2p` coordinate,

```text
Q_c(X) = 2 gamma T grad_V(L2p) grad_V(L2p)^T,
```

and ask whether whitening by this exact state-dependent covariance removes
the `L2p` kurtosis and squared-residual memory.  Only if that fails is an
explicit `L3p` drift coordinate justified.  Because the present second-order
state already fails its own Gaussian gate, generating the expensive C3
independent validation trajectories at this truncation is not authorized.
