# Transient Periodic Langevin Design

## Research question

The activated cage-geometry quotient established that the calibration one-jump
law is broader than a fixed-length periodic-potential null, but an independent
Poisson count law leaves only `8/21` physical T=0.45 rows and fails the frozen
`F_s(k=7.25)` tolerance.  Existing two-clock and state-conditioned block
kernels already show that a better average clock or a one-block slow/fast
emission law is insufficient at T=0.45.

This step asks a narrower microscopic question:

> Can one continuous, equilibrium extended-coordinate Langevin system produce
> a harmonic cage, activated cage-to-cage escape, count overdispersion, finite
> recoil memory, and state-dependent exit geometry without imposing renewal
> jumps or a discrete hidden-state clock?

The step is a mechanism-capability and falsification study.  It does not fit a
complete KA model and cannot promote a microscopic-closure claim.

## Evidence boundary from the count-overdispersion canary

Let `N` have mean `n` and Fano factor `F`, and let independent isotropic jump
marks have radial second moment `m2`, one-component second moment `m2/3`,
one-component fourth moment `m4x`, and characteristic function `phi_J(k)`.
For an independent Gaussian cage contribution,

```text
kappa_4,x = n m4x + 3 n (F-1) (m2/3)^2,
kappa_4,x = alpha_2 M^2/3.
```

Therefore

```text
n = alpha_2 M^2 / {3 [m4x + 3(F-1)(m2/3)^2]},
a = (M-n m2)/6.
```

For the gamma-Poisson marginal with the same `n` and `F`,

```text
G_N(s) = [1 + (F-1)(1-s)]^[-n/(F-1)],
F_s(k) = exp(-k^2 a) G_N(phi_J(k)).
```

Using the existing calibration two-clock predicted Fano factors restores
physical support from `8/21` to `20/21` T=0.45 rows, but the maximum supported
`F_s(k=7.25)` error remains about `0.0411`, above the unchanged `0.03`
tolerance.  T=0.58 has `25/25` support and about `0.0466` maximum high-k error,
but remains a stationarity-failing canary.  Count overdispersion is therefore
necessary for the moment inversion but not sufficient for cage-scale shape.

The implementation will recompute this result mechanically from committed
tables.  It is a diagnostic quotient because heldout MSD and NGP remain inputs.

## Alternatives considered

### Overdispersed compound-jump model only

This is analytically transparent and fixes most negative-cage roots.  The
canary already shows that it does not close high-k scattering, and the existing
hybrid macro transfer has already rejected an independent count-jump kernel.
It is retained only as a null and boundary-setting result.

### A finer discrete state-conditioned HMM

The repository already conditions full calibration block displacements on a
slow/fast count posterior and propagates them through two clocks.  That model
passes T=0.58 curves but fails T=0.45 NGP and multi-k scattering.  Adding more
discrete states before identifying a continuous microscopic coordinate would
increase flexibility without resolving the physical origin.

### Coupled transient periodic Langevin model

This is the selected route.  It follows the transient-potential coarse-graining
structure derived by Uneyama from microscopic overdamped Langevin dynamics,
while making the local cage, barrier, and retained elastic environment
explicit.  The model is continuous and satisfies fluctuation-dissipation by
construction.  Discrete cage events appear only after trajectory analysis.

## Continuous model

For one Cartesian component, define a tagged coordinate `x`, a slow elastic
environment coordinate `q`, and a slow barrier coordinate `z`.  Let

```text
u = 2 pi x / L,
V(z) = V0 + g z^2,
U(x,q,z) = V(z)[1-cos(u)]/2 + K(x-q)^2/2 + k_z z^2/2.
```

The overdamped Ito dynamics is

```text
gamma_x dx = -partial_x U dt + sqrt(2 gamma_x T) dW_x,
gamma_q dq = -partial_q U dt + sqrt(2 gamma_q T) dW_q,
gamma_z dz = -partial_z U dt + sqrt(2 gamma_z T) dW_z.
```

All noises are independent.  The same scalar temperature appears in every
fluctuation-dissipation pair.  The potential is invariant under
`(x,q) -> (x+L,q+L)`, so the common translation remains diffusive while the
relative coordinate and `z` have a stationary equilibrium distribution.

In `d` dimensions, `x` and `q` are vectors while one scalar `z` is shared by
all components:

```text
U(x,q,z) = sum_i {V(z)[1-cos(2 pi x_i/L)]/2
                  + K(x_i-q_i)^2/2} + k_z z^2/2.
```

The shared `z` represents one local barrier environment and correlates escape
activity across directions.  Observables average over Cartesian components,
matching the existing KA `F_s` convention.  This cubic minimal model is not
claimed to reconstruct isotropic many-body cage geometry.

## Microscopic-to-effective reduction

Near a minimum and for frozen `q,z`, the tagged coordinate is locally OU with

```text
kappa_x(z) = 2 pi^2 V(z)/L^2 + K,
tau_c(z) = gamma_x/kappa_x(z),
variance_x(z) = T/kappa_x(z).
```

For `K L^2 << V(z)`, a frozen-environment Eyring-Kramers approximation gives
forward and backward escape rates

```text
lambda_+/- (delta,z)
  approximately A_+/- (delta,z) exp[-Delta U_+/- (delta,z)/T],
delta = x_well-q.
```

The elastic offset `delta` changes sign and magnitude after a crossing.  Since
`q` relaxes on `tau_q=gamma_q/K`, the old cage exerts a temporary restoring
force: immediate backtracking is enhanced, while repeated forward motion is
suppressed until `q` follows.  This generates finite ordered-path memory from
continuous dynamics rather than from an event-level recoil rule.

Inside a well, `z` is approximately OU with

```text
tau_z = gamma_z/k_z,
Var(z) approximately T/k_z.
```

Because `V(z)=V0+g z^2`, the conditional Kramers rate is a nonlinear positive
functional of a squared OU coordinate.  It produces finite-lifetime dynamic
rate disorder and a non-gamma rate law.  The same `z` also changes cage
curvature, so count and cage channels are coupled rather than factorized.

Adiabatic elimination of fast intra-well `x` yields a semi-Markov cage-index
process conditioned on `(q,z)`.  A renewal clock is recovered only after the
additional approximation that `(q,z)` fully re-equilibrate between escapes.

## Frozen ablations

Four models use identical integration, seeds, and analysis:

| model | `K` | `g` | purpose |
|---|---:|---:|---|
| static periodic | 0 | 0 | ordinary periodic Kramers baseline |
| rate-only | 0 | positive | finite barrier disorder without recoil |
| elastic-only | positive | 0 | recoil/path memory without rate disorder |
| full transient | positive | positive | coupled count, cage, and path memory |

When `K=0`, the irrelevant `q` coordinate is held fixed and omitted from
stability/equilibrium checks.  When `g=0`, `z` may still be integrated as an
uncoupled equilibrium control, but its path cannot enter tagged-particle
predictions.

The ablation gate is directional, not a demand that one hand-selected synthetic
parameter set match all KA numbers.  The expected mechanical distinctions are:

```text
rate-only: count Fano increases relative to static periodic,
elastic-only: lag-one cage-step correlation becomes more negative,
full transient: both effects coexist with continuous trajectories.
```

Persistence/exchange separation, NGP peak, and multi-k `F_s` are reported for
all models but are not preregistered success conditions for a KA claim.

## Numerical protocol

- Use vectorized Euler-Maruyama with explicit seed and fixed `dt`.
- Initialize relative coordinates from a burn-in segment; discard burn-in.
- Record unwrapped `x`, `q`, `z`, and potential energy at a fixed stride.
- Define a raw cage index by nearest periodic minimum.
- Accept a cage transition only after the new index persists for a fixed
  non-recrossing dwell measured in recorded frames.
- Compute continuity, equipartition/OU, event count, Fano, persistence,
  exchange, cage-step correlation, MSD, NGP, and multi-k `F_s` diagnostics.
- Reject unstable runs on nonfinite state, excessive Euler displacement, or an
  unbounded relative coordinate.

The integration test uses modest ensembles and deterministic tolerances.  It
is a reproducibility check, not evidence about real KA trajectories.

## Deliverables

1. A focused module containing parameters, forces, simulation, event
   extraction, and observable summaries.
2. Unit tests for gradients, translational invariance, deterministic seeds,
   equilibrium local variance, event non-recrossing, and ablation direction.
3. A script and CSV gate for the real count-overdispersion geometry quotient.
4. A synthetic ablation CSV and compact SVG showing count Fano and recoil
   separately.
5. A research note with the derivation, numerical verdict, literature
   relation, and next calibration-only KA gate.

## Claim boundary

This stage may establish only that the continuous model is capable of
generating the required classes of local signatures and that count
overdispersion alone is insufficient on committed KA tables.  It does not
identify `q` or `z` in real particle configurations, validate conditional
Kramers rates in KA, or produce blind heldout macro predictions.

The following flags remain zero regardless of the synthetic ablation outcome:

```text
blind_prediction_claim_allowed = 0
finite_exchange_resolved = 0
static_environment_resolved = 0
spatial_facilitation_resolved = 0
activated_cage_geometry_resolved = 0
transient_potential_identified_in_ka = 0
microdynamic_closure_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

The next promotion gate must infer observable proxies for `q,z` from the
calibration half of raw KA trajectories, test conditional escape rates and
recoil, and then predict heldout MSD, NGP, and multi-k `F_s` without using
heldout `k=7.25`.

## Literature relation

- Uneyama, Phys. Rev. E 101, 032106 (2020), derives Langevin equations with a
  transient potential from microscopic overdamped Langevin path probabilities
  and applies the framework to a tagged particle in a supercooled liquid.
- Uneyama, Phys. Rev. E 105, 044117 (2022), derives transient-potential
  dynamics from Hamiltonian projection and shows how approximations produce
  Markovian potential-parameter dynamics.
- Chechkin et al., Phys. Rev. X 7, 021002 (2017), establishes squared-OU
  diffusing diffusivity as a continuous route to non-Gaussian displacement
  statistics.  The present `z` coordinate instead modulates an activated
  barrier and cage curvature.
- Hasyim and Mandadapu, PNAS 121, e2322592121 (2024), couples localized
  rearrangements through retained elastic stress.  The present `q` coordinate
  is a single-particle local-memory analogue and cannot establish spatial
  facilitation.
