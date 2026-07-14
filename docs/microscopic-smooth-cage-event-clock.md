# Microscopic Smooth-Cage Initial Escape Test

## Question

The smooth force-support coordinate already has an exact instantaneous
projection from the many-particle Langevin equation. This experiment asks the
next, stronger question:

> Is the instantaneous projected state `(u,p,G,b)` sufficient to predict the
> first nonrecrossing cage-center escape in an unseen parent configuration?

The answer at this checkpoint is **no**. The state carries a small transferable
signal, but it fails the preregistered sufficiency and ablation gates.

## Microscopic state and event label

For tagged particle `i`, the smooth cage center is

```text
C_i = r_i + sum_j w_ij d_ij / sum_j w_ij,
w_ij = wendland_c4(r_ij / (2.5 sigma_ij)).
```

The instantaneous projected variables follow from the full particle state:

```text
u_i = r_i - C_i,
p_i = J_i V,
G_i = J_i J_i^T,
b_i = J_i F + Hess(u_i):(V tensor V) - gamma p_i.
```

The LAMMPS force at frame zero is contracted directly with the analytic
Jacobian in `J_i F`. The geometric term is the same centered directional
derivative validated in the smooth tangent experiment.

Escape is defined on the unwrapped `C_i(t)` path by the unchanged event rule:

```text
p_hop threshold   = 0.08
half window       = 8 frames = 0.4 tau
recrossing radius = sqrt(0.08)
reduction         = contiguous peak plus recursive adjacent A-B-A removal
horizon           = 20 tau
```

Particles with no retained event are administratively right-censored at the
common horizon. No event threshold or window was changed after inspecting the
result.

## Data and reconstruction

The test uses five independent standard KA parent restarts at `T=0.58`.
For every parent, the first eight independent Langevin clones are used. Each
clone has 401 frames separated by `0.05 tau`.

The original long dumps contain positions but not initial velocities or
forces. Their manifests preserve the parent restart and random seeds. Each
clone's frame-zero full state was therefore reconstructed by

```text
read_restart
velocity all create T velocity_seed
fix nve
fix langevin T T gamma langevin_seed
dump x,v,F
run 0
```

All five restart hashes are distinct. All 40 reconstructed states match the
corresponding long-trajectory frame-zero positions with maximum coordinate
error `0.0` at dump precision. The raw reconstruction dumps were deleted after
reduction.

Exactly 64 A particles were selected once with seed `20260714` and reused in
all clones and parents. The resulting data contain:

```text
5 parents x 8 clones x 64 targets = 2560 observations
1731 observed first escapes
829 right-censored observations
```

Every held parent contains both events and censoring, so no fold is scored by
extrapolating from an empty event class.

## Censored exponential model

For observation `n`, let `t_n` be the observed event time or censoring time and
`delta_n` indicate an observed escape. The microscopic reaction coordinate is

```text
lambda_n = exp(beta_0 + beta dot z_n).
```

The penalized negative log likelihood is

```text
L = sum_n [lambda_n t_n - delta_n log(lambda_n)]
    + (1/2) beta^T P beta.
```

The intercept is unpenalized and standardized feature coefficients use fixed
`L2=1`. Its analytic derivatives are

```text
grad L = X^T(lambda t - delta) + P beta,
Hess L = X^T diag(lambda t) X + P.
```

The model is refitted five times, withholding a complete parent each time.
The baseline in each fold is one constant rate estimated from the other four
parents.

The rotationally invariant models are:

```text
geometry:  log|u|^2 and three log eigenvalues of G                 (4)
kinematic: geometry plus log|p|^2 and cos(u,p)                     (6)
full:      kinematic plus log|b|^2, cos(u,b), and cos(p,b)         (9)
```

## Result

| Model | Mean held-parent Brier skill | Mean likelihood gain per observation | Minimum parent likelihood gain | Maximum survival error |
|---|---:|---:|---:|---:|
| geometry `(u,G)` | `0.00836` | `0.00646` | `1.2947` | `0.06743` |
| kinematic `(u,p,G)` | `0.00771` | `0.00606` | `0.7812` | `0.06729` |
| full `(u,p,G,b)` | `0.00529` | `0.00498` | `-0.7691` | `0.06728` |

The integrity and survival-calibration gates pass. The full state nevertheless
fails three decisive tests:

1. Its Brier skill `0.00529` is below the preregistered difficulty reference
   `0.026964`, taken from the best prior relative-neighbor structural
   committor diagnostic.
2. It performs worse than geometry alone (`0.00836`) and worse than the
   kinematic ablation (`0.00771`). Adding `b` does not reveal readiness.
3. Parent 2 has negative full-state likelihood gain, `-0.7691`, so the result
   is not uniformly transferable across parent configurations.

The five full-state held-parent results are:

| Parent | Events / 512 | Brier skill | Likelihood gain | Survival error |
|---:|---:|---:|---:|---:|
| 1 | `341` | `0.00809` | `1.5761` | `0.02536` |
| 2 | `346` | `-0.00530` | `-0.7691` | `0.04611` |
| 3 | `364` | `0.00020` | `1.5227` | `0.06526` |
| 4 | `328` | `0.00774` | `4.4475` | `0.06728` |
| 5 | `352` | `0.01570` | `5.9691` | `0.05037` |

## Physical interpretation

This failure separates two derivation levels that can otherwise be confused.
The map from the many-particle Langevin state to `(u,p,G,b)` is exact at one
instant. Exactness of the coordinate transformation does not make those
variables a sufficient Markov state after the other particles are eliminated.

Within one parent, `u` and `G` are configuration variables and are shared by
the eight clones. The randomized clone velocities change `p` and the
velocity-dependent part of `b`, but those fast variables relax long before
many of the `20 tau` escapes occur. Their addition therefore contributes more
clone noise than transferable slow readiness. The small geometry signal says
that local cage shape matters, but the missing predictor must contain slower
many-body structure or projected history not encoded by `(u,p,G,b)` at one
time.

The good survival error does not rescue the state-sufficiency claim. It shows
that an exponential family can reproduce coarse survival reasonably well;
the Brier and likelihood ablations show that the proposed microscopic state
does not discriminate which particle will escape.

## Next microscopic test

The next experiment must diagnose missing state without retuning this event
definition. Two evidence-driven candidates are:

1. Add instantaneous many-body structural invariants, such as species-resolved
   radial density and local Hessian soft modes, to the same smooth-center
   labels and ask whether they explain the geometry-model residual across
   parents.
2. Evaluate a physical-time path with projected history variables derived from
   the orthogonal force or slowly filtered local structure, with their memory
   time fixed from measured cage relaxation rather than a fitted delayed
   hazard.

A positive structural residual test would identify a concrete precursor
missing from the single-coordinate projection. A negative result would route
the work to path history and a nonlinear Mori-Zwanzig state rather than adding
another static scalar.

## Claim boundary

The machine-readable verdict is

```text
microscopic_initial_escape_state_allowed = 0
event_clock_claim_allowed = 0
autonomous_single_particle_gle_claim_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

Thus this checkpoint strengthens the microscopic derivation by ruling out an
insufficient closure. It does not yet derive the renewal hazard or the final
single-particle autonomous Langevin dynamics.
