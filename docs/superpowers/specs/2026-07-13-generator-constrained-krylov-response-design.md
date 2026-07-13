# Generator-Constrained Krylov Response Design

## Objective

Derive a tagged-particle extended Langevin model directly from the full
Kob-Andersen (KA) many-particle Langevin generator, then test whether its
microscopic bath modes predict held-out full-system responses before testing
cage-jump clocks or macroscopic glass observables.

This phase does not claim a thermodynamic glass transition. It preserves
`thermodynamic_claim_allowed = 0` throughout.

## Microscopic Starting Point

For unit mass, the underdamped KA dynamics and backward generator are

```text
dr_i = v_i dt
dv_i = [F_i(R) - gamma v_i] dt + sqrt(2 gamma T) dW_i

L = sum_i v_i . grad_(r_i)
    + sum_i [F_i - gamma v_i] . grad_(v_i)
    + gamma T sum_i Delta_(v_i).
```

For the conservative force on a tagged particle,

```text
F_i = -grad_i U,
H_ij = partial^2 U / partial r_i partial r_j,
G_i = L F_i = -sum_j H_ij v_j.
```

For a central pair interaction with pair Hessian block `K_ij`, this becomes

```text
G_i = -sum_(j != i) K_ij (v_i - v_j).
```

The first Krylov force mode is therefore an exact instantaneous observable of
the many-particle configuration and velocities. It is not a finite-difference
history variable or a fitted precursor.

Because `G_i` is linear in all particle velocities, Ito's formula gives

```text
dG_i = (L^2 F_i) dt - sqrt(2 gamma T) sum_j H_ij dW_j,

Cov[dG_i | R,V]
  = 2 gamma T dt sum_j H_ij H_ij^T.
```

The first auxiliary-mode noise is therefore multiplicative and anisotropic,
with covariance fixed by the same microscopic Hessian that defines `G_i`.

## Reduced Model Under Test

The first generator-constrained *tangent-response* model is

```text
delta dot x = delta v
m delta dot v = delta F - gamma delta v
delta dot F = delta G
delta dot G = A_x delta x + A_v delta v
              + A_F delta F + A_G delta G + delta R_2.
```

The first three equations are fixed by microscopic dynamics. Only the
projection of `L^2 F` onto the retained state is estimated. The residual
`R_2` is retained as an orthogonal stochastic force and is not assumed to be
independent, Gaussian, or white until those properties pass direct tests.
The `A` coefficients are three-by-three response blocks, not scalar isotropic
constants unless isotropy passes an explicit comparison.

Eliminating `F` and `G` yields a finite-dimensional generalized Langevin
description for the tagged-coordinate response. Failure at this order triggers
the next generator mode `L^2 F`; it does not trigger an arbitrary latent
variable.

Absolute tagged position cannot enter a homogeneous-liquid restoring law
without violating translation invariance. Consequently, passing this tangent
test is necessary but not sufficient for the final autonomous stochastic
model. That model must resolve either the cage-relative coordinate
`u = x - C` or an explicit cage center `C`, then revalidate the generator
closure in that coordinate before testing cage-to-cage escape.

## Components

### Exact generator observables

Add a vectorized KA routine that returns, for selected tagged particles:

- conservative pair force `F_i`;
- pair Hessian row blocks `H_ij`;
- first generator force mode `G_i = L F_i`;
- infinitesimal conditional covariance rate
  `2 gamma T sum_j H_ij H_ij^T`.

The implementation reuses the established KA epsilon, sigma, cutoff, and
minimum-image conventions. It must agree with the existing force and Hessian
routines on overlapping outputs.

### Generator validation

Use existing full-state force/velocity trajectories to verify

```text
[F_i(t+dt) - F_i(t-dt)] / (2 dt) approximately equals L F_i(t).
```

The current read-only feasibility audit found correlations of
`0.99910--0.99937` and relative L2 differences of `0.0367--0.0432` across
four independent clones at saved-frame interval `0.005 tau`.

For the stochastic law, define

```text
Delta G_i - L^2 F_i dt
```

and compare its conditional covariance with the Hessian-derived covariance.
The current one-clone, 101-increment canary gives a trace variance ratio
`0.862`, mean squared Mahalanobis distance `2.56` versus the three-dimensional
infinitesimal value `3`, and normalized residual mean `0.056`. This is a
canary, not yet an ensemble conclusion.

### Controlled response data

Run matched `+/-` tagged displacements at `T=0.58` using common Langevin noise:

- `epsilon = 0.001, 0.002`;
- eight isoconfigurational noise seeds;
- `0--1 tau`, saved every `0.005 tau`;
- full positions and velocities only as transient extraction inputs;
- retained arrays: `x, v, F, L F`, force/Hessian consistency diagnostics,
  input seeds, and timing metadata.

Disk space is constrained. Each path is generated and reduced sequentially.
Only one audit raw trajectory is retained; other newly generated raw dumps are
removed only after their derived data and validation checks are written
successfully. Peak additional storage should remain below `200 MiB`.

### Held-out closure tests

Fit no diffusion, scattering, event-clock, or other macroscopic observable.
Estimate the final tangent-generator row only from microscopic response data
and test:

- leave-one-clone-out prediction;
- cross-epsilon prediction;
- temporal prediction from an early fit interval;
- stability of the autonomous transition;
- residual correlation with retained state;
- Hessian-derived conditional noise covariance.

The preregistered response gates are:

```text
relative position-response error <= 0.20 at 0.2 tau
relative position-response error <= 0.30 at 1.0 tau
stable transition in every leave-one-clone-out fold
cross-epsilon response mismatch <= 0.02 through 1.0 tau.
```

If the system is outside the linear regime before a requested horizon, that
horizon is reported as unidentified rather than counted as a model failure.

## Escalation Ladder

If `(x,v,F,LF)` fails while the response remains linear:

1. compute and validate `L^2 F` using the full generator, including its
   velocity-Laplacian contribution when required;
2. orthogonalize successive generator modes with the equilibrium covariance
   inner product to form a block Mori-Krylov chain;
3. stop increasing rank when held-out response errors converge or when the
   required rank ceases to be a useful single-particle reduction;
4. only after response closure, estimate the orthogonal-force law and test
   first passage, persistence/exchange times, jump marks, diffusion, NGP,
   multi-k `F_s`, SE violation, and heterogeneity proxies.

State-dependent or age-dependent kernels are a later branch, activated only
if a converged linear generator hierarchy still fails conditional event-clock
tests.

## Scientific Boundaries

Passing the response gates establishes a microscopic reduced dynamical model
for infinitesimal perturbations only over the validated state point and time
range. It does not yet establish an autonomous cage-relative stochastic model,
and does not by itself derive a renewal hazard, Kramers rate, configurational
entropy, Kauzmann temperature, ideal-glass transition, or heat-capacity
anomaly.
