# Same-Label Local Soft-Mode Precursor Residual Test

## Scope

This is the final preregistered test of a scalar instantaneous local
configuration state before moving to explicit dynamical memory. It asks:

> Does a directional local Hessian state explain held-parent smooth-cage
> escape information absent from `(u,G)` and isotropic radial packing?

Earlier local-Hessian tests used frozen-minimum committors or one continuous
trajectory. They were negative, but they did not use the current five-parent,
320-configuration smooth-center labels. This experiment closes that exact
comparison without changing labels, targets, regularization, or validation.

## Fixed microscopic Hessian state

For target `i`, define the active cluster as all particles within `1.5` of the
target in the frame-zero configuration. Construct the exact KA pair Hessian

```text
H_ab = d2 U_KA / (d r_a d r_b)
```

for active particles. Interactions with particles outside the cluster remain
in active-particle diagonal blocks, so the exterior is a pinned microscopic
environment rather than deleted.

Diagonalize the local matrix. For the first and fourth eigenvalues strictly
above `1e-6`, with ranks `q in {0,3}`, define

```text
L_q = log(1 / lambda_q),
P_q = log[(sum_alpha |e_q(i,alpha)|^2) / lambda_q].
```

Append

```text
N_cluster = log(number of active particles),
N_nonpositive = number of eigenvalues <= 1e-6.
```

The resulting six-dimensional state contains directional curvature,
target-mode participation, cluster size, and local instability count. It is a
deterministic function of the many-particle configuration and is identical
across isoconfigurational clones.

The controls are frozen from prior local-Hessian work:

```text
cluster cutoff   = 1.5
mode ranks       = [0,3]
eigenvalue floor = 1e-6
L2 regularization = 1
```

No inherent-structure minimization is added. This test concerns the actual
finite-temperature state from which the Langevin clones depart.

## Data and models

Reuse exactly the radial residual experiment's data and row assembly:

```text
5 parents x 8 clones x 64 targets = 2560 survival rows
5 parents x 64 targets            = 320 propensity rows
events / censors                  = 1731 / 829
T                                 = 0.58
horizon                           = 20 tau
p_hop threshold / half-window     = 0.08 / 8 frames
validation                        = leave one complete parent out
```

Compare:

```text
geometry          = (u,G),                   4 features
softmode          = local Hessian state,     6 features
geometry_softmode = both,                   10 features
```

Use the same censored exponential and aggregated grouped-binomial diagnostics
as the radial experiment.

## Integrity and decision gates

Require exact counts, target equality, five distinct parent hashes,
frame-zero clone invariance within `1e-12`, and exact reproduction of the
geometry checkpoint within `1e-12`.

Set `instantaneous_local_softmode_precursor_allowed = 1` only if all hold:

1. Combined held-parent Brier skill is at least `0.01` above both geometry and
   softmode alone.
2. Combined held-parent Brier skill exceeds `0.026964`.
3. Mean censored likelihood gain is positive and every held-parent total gain
   is nonnegative.
4. Maximum held-parent survival error is at most `0.10`.
5. Combined aggregated binomial Brier skill is positive and exceeds geometry.

Regardless of outcome, keep event-clock, autonomous GLE, Kramers, and
thermodynamic claims false.

## Outcome routing

If the test passes, the six-dimensional Hessian state becomes a measured
precursor candidate. The next test must follow its temporal evolution and ask
whether first passage into a soft set produces the delayed hazard.

If the test fails, combine it with the existing radial, frozen-minimum,
adiabatic-Hessian, and cubic-barrier failures. Stop adding scalar static local
descriptors. The next model must retain dynamical bath information through a
Mori-Zwanzig orthogonal-force state or memory kernel. The first dynamic test
will use only microscopic trajectory correlations and held-parent prediction,
not event labels, to identify the required slow auxiliary coordinates.

## Deliverables

- A shared tested six-dimensional instantaneous local soft-mode descriptor.
- A same-label five-parent analysis and deterministic CSV artifacts.
- A package-gated scientific verdict.
- An explicit transition to dynamic-memory closure if the gate fails.
