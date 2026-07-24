# L2p Conditional-Diffusion Gate Design

## Objective

Test the next exact object in the smooth-cage generator hierarchy: whether the
state-dependent thermostat diffusion of `c = L2p` explains the non-Gaussian
and volatility-correlated residual left by the finite `(u,p,Lp,L2p)` Markov
bath.

This is a microscopic diagnostic, not a fit of the renewal model.  It may
identify `c` and its diffusion geometry as a candidate environment coordinate
`z`, but it cannot by itself establish autonomous cage escape, Kramers rates,
macroscopic closure, or thermodynamics.

## Why This Gate Is Not The Existing Projected-Noise Audit

For the center-relative velocity state `(w,p)`, the exact projected covariance
`Q(R)` is already validated.  At saved-frame spacing `0.01 tau`, a constant
three-parameter correlated Gaussian covariance differs from exact `Q(R)` by
at most `0.00170` over the tested metrics.  Configuration dependence of that
first projected covariance is therefore not the missing environment variable.

The second-generator checkpoint found a different, unresolved signal.  Adding
the exact coordinate `c=L2p` reduced the worst held `Lp` squared-residual
correlation from `0.32917` to `0.19946` and the worst `Lp` excess kurtosis from
`3.17998` to `1.68713`.  The new `c` coordinate itself retained squared-residual
correlation `0.37876` and excess kurtosis `5.02707`.  The present gate tests the
microscopic explanation stated before those results were known: omitted
state-dependent diffusion of `c`.

## Microscopic Parent And Generator Hierarchy

The full unit-mass many-particle dynamics are

```text
dR = V dt,
dV = [F(R)-gamma V] dt + sqrt(2 gamma T) dW,

L = V.grad_R + [F-gamma V].grad_V + gamma T Delta_V.
```

For the fixed smooth force-support cage coordinate `u(R)`, define

```text
p = Lu,
b = Lp,
c = L2p.
```

The exact Ito equation for `c` is

```text
dc = L3p dt + sqrt(2 gamma T) A_c(X) dW,
A_c(X) = grad_V c(X),
Q_c(X) = 2 gamma T A_c(X) A_c(X)^T.
```

`Q_c` is a `3 x 3` covariance-rate matrix for each target particle.  It is
computed from the full particle state and the same KA force, cage map,
directional steps, and fixed Ito trace probes used to construct `c`.  No event
label or macroscopic observable enters `c` or `Q_c`.

## Probe Construction And Numerical Status

The implementation evaluates velocity directional derivatives

```text
D_V c(X)[eta]
  = [c(R,V+h eta)-c(R,V-h eta)]/(2h) + O(h^2)
```

with fixed antithetic Rademacher vectors `eta`.  For `M` probes,

```text
Q_c^(M) = 2 gamma T M^-1 sum_a y_a y_a^T,
y_a = D_V c[eta_a].
```

This is a deterministic, seed-recorded Hutchinson estimate conditional on the
stored state.  It is not called exact at finite `M`.  The primary estimate uses
`M=32`; nested prefixes `M={4,8,16,32}` and velocity steps
`h={3e-6,1e-5,3e-5}` are retained for convergence.  The same inner trace probes
must be reused in every `+h/-h` pair so that their stochastic trace error does
not contaminate the derivative.

The primary analysis is authorized only if, in every held clone:

- all matrices are finite and positive semidefinite to numerical tolerance;
- the median relative Frobenius difference `Q_c^(16)` versus `Q_c^(32)` is at
  most `0.10`;
- the 95th-percentile relative Frobenius difference is at most `0.25`;
- changing `h` from `1e-5` to either sensitivity value changes the median
  `Q_c^(32)` by at most `0.10` and the 95th percentile by at most `0.25`.

Failure of this numerical gate leaves the physical comparison unevaluable.  A
larger probe count may be preregistered in a later revision, but the current
thresholds must not be retuned using closure results.

## Frozen Data Split

Use the same four `T=0.58`, 10-tau Langevin clones, 64 fixed A particles,
`0.01 tau` frame spacing, 200-frame second-generator discovery window, target
seed, trajectory hashes, cage definition, memory order 40, and VAR order 16 as
the existing relative second-generator checkpoint.

Each leave-one-clone-out fold fits every statistical quantity on the other
three clones.  The held clone supplies only scores.  Pooled cancellation cannot
turn a failed held clone into a pass.

If the raw clone, drift, or second-generator caches have been removed, they are
regenerated from their recorded parent restart and pinned simulation protocol.
Regeneration must preserve trajectory hashes or produce a separately named
rerun protocol; rerun data cannot silently replace the original four-clone
checkpoint.

## Residual And Covariance Quotient

Use the already frozen `(u,p,Lp,L2p)` Mori-40 plus VAR-16 fit.  Let `e_c,n` be
the three-component white innovation belonging to `c` after applying the
training-fold operators to a path.  The covariance quotient is diagnostic:
because the finite-memory filter combines several microscopic increments, it
does not assert `Cov(e_c,n|X_n) = dt Q_c(X_n)` exactly.

For each training fold, fit only the scalar normalization and isotropic floor

```text
Sigma_c,n = a dt Q_c(X_n) + delta I,
a >= 0, delta >= 0,
```

by Gaussian likelihood of training `e_c`.  This two-parameter calibration is a
microscopic residual-law fit.  It may not use event labels, MSD, NGP,
`F_s(k,t)`, diffusion, persistence, exchange, or event rate.  Report `a` and
`delta`; a dominant floor is evidence against `Q_c` sufficiency.

## Frozen Nulls

Evaluate the same held innovations under:

1. `constant_full`: a full constant `3 x 3` covariance fitted on training
   innovations;
2. `constant_isotropic`: one constant scalar variance;
3. `trace_only`: `a dt tr(Q_c)/3 I + delta I`, retaining only the local scale;
4. `exact_tensor`: `a dt Q_c + delta I`, retaining scale and orientation;
5. `permuted_tensor`: the `Q_c` sequence permuted within target and clone with
   a fixed seed, retaining its marginal distribution but destroying state-time
   alignment.

The paired `trace_only` versus `exact_tensor` comparison distinguishes scalar
volatility from tensor orientation.  The paired `exact_tensor` versus
`permuted_tensor` comparison tests whether instantaneous microscopic alignment,
not merely a broad covariance distribution, carries information.

## Held-Clone Metrics

For every model and held clone report:

- mean negative Gaussian log likelihood per three-component sample;
- mean squared Mahalanobis value divided by 3;
- maximum absolute whitened mean;
- maximum absolute whitened covariance error;
- maximum absolute component excess kurtosis;
- maximum absolute whitened correlation at lags `1..40`;
- maximum absolute squared-whitened correlation at lags `1..40`;
- energy correlation between `|e_c|^2` and `tr(Q_c)`;
- fitted `a`, fitted `delta`, and floor variance fraction;
- sample count, trajectory hash, numerical-convergence flags, and all claim
  flags.

The distributional metric is the energy distance between held whitened
innovations and a fixed-size, fixed-seed `N(0,I_3)` reference.  It is reported
as a diagnostic; Gaussianity decisions remain tied to the frozen moment and
memory gates so that no post-run reference threshold is introduced.

## Decision Gates

`l2p_conditional_diffusion_supported = 1` only if the numerical probe gate and
all of the following pass in every held clone:

- `exact_tensor` improves held negative log likelihood over `constant_full`;
- the replicate-first mean improvement has a two-sided 95% Student-t interval
  strictly above zero;
- `exact_tensor` reduces maximum squared-residual correlation by at least 25%
  relative to `constant_full`;
- `exact_tensor` has maximum ordinary and squared whitened correlations at
  lags `1..40` no greater than `0.05`;
- maximum absolute component excess kurtosis is no greater than `0.35`;
- mean squared Mahalanobis divided by 3 lies in `[0.8,1.2]`;
- maximum absolute covariance error is no greater than `0.10`;
- `permuted_tensor` does not pass all preceding closure metrics;
- the fitted isotropic floor fraction is no greater than `0.25`.

`l2p_tensor_orientation_required = 1` additionally requires `exact_tensor` to
improve held likelihood over `trace_only` in every clone with a replicate-first
95% interval above zero.  Otherwise only scalar conditional volatility may be
supported.

If `exact_tensor` improves likelihood or volatility memory but misses any
absolute Gaussian closure gate, record
`l2p_conditional_diffusion_informative_but_insufficient = 1` and keep the
closure claim at zero.  That outcome authorizes deriving `L3p`; it does not
authorize adding an arbitrary latent state.

## Claim Boundary

The output always includes

```text
l2p_diffusion_probe_converged = 0/1
l2p_conditional_diffusion_supported = 0/1
l2p_conditional_diffusion_informative_but_insufficient = 0/1
l2p_tensor_orientation_required = 0/1
l3p_derivation_authorized = 0/1
microscopic_environment_coordinate_z_allowed = 0/1
continuous_gaussian_langevin_bath_allowed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

Even a passing result only identifies `z=(c,Q_c)` as a microscopic candidate
inside a local generator hierarchy.  Promotion to
`microscopic_environment_coordinate_z_allowed = 1` requires both conditional
diffusion support and a later held-clone autonomous evolution law for `z`.
Therefore that flag remains zero in this checkpoint.

## Follow-On

- If the tensor gate passes absolutely, derive and test an autonomous Markov
  evolution for `z=(c,Q_c)` driven by the inherited Gaussian thermostat.
- If it is informative but insufficient, derive `L3p` and its diffusion before
  adding any phenomenological variable.
- If it is uninformative, stop the finite generator hierarchy at this branch
  and test a nonlinear position-dependent memory kernel or transient-potential
  projection with the same held-clone discipline.
- Event-clock, Kramers, and macro-observable tests begin only after an
  autonomous microscopic state passes its own trajectory holdout.
