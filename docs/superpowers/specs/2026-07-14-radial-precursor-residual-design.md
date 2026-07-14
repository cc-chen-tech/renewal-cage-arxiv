# Species-Resolved Structural Precursor Residual Test

## Status and scope

This is a preregistered follow-up to the five-parent smooth-cage initial
escape test. That test established an exact instantaneous projection of the
many-particle Langevin state but rejected `(u,p,G,b)` as a sufficient escape
state. This follow-up asks one narrower microscopic question:

> Does species-resolved local packing contain transferable initial escape
> information that is absent from the smooth-cage geometry `(u,G)`?

The event definition, target particles, parents, clones, censoring horizon,
regularization, and held-parent split remain fixed. This experiment does not
retune the renewal hazard and does not claim a complete single-particle
Langevin closure.

## Scientific alternatives

Three mechanisms could explain the previous failure.

1. **Missing static many-body structure.** The smooth coordinate compresses
   the environment into a weighted center and a local metric. It can discard
   species-resolved shell packing that controls a local barrier.
2. **Missing inherent-structure softness.** A barrier may be encoded primarily
   in collective low-frequency Hessian modes rather than radial shell density.
3. **Missing path history.** No instantaneous local descriptor may be a
   sufficient state after the bath is eliminated; the correct reduction may
   require a Mori-Zwanzig memory kernel or explicit auxiliary variables.

The first alternative is tested first because it adds a deterministic
function of the same microscopic configuration, requires no new phenomenology,
and can be falsified on the existing isoconfigurational ensemble. A failure
routes the work to nonlocal modes or history rather than another static scalar.

This ordering is consistent with the following primary literature:

- Schoenholz et al., *A structural approach to relaxation in glassy liquids*,
  https://arxiv.org/abs/1506.07772, introduced a local structural softness
  strongly correlated with rearrangements.
- Obadiya and Sussman, *Using fluid structures to encode predictions of
  glassy dynamics*, https://arxiv.org/abs/2211.00604, showed that classifiers
  built from fluid structure recover information about rearrangement barriers.
- Widmer-Cooper et al., *Irreversible reorganization in a supercooled liquid
  originates from localised soft modes*, https://arxiv.org/abs/0901.3547,
  motivates the next Hessian route if radial structure fails.
- Vroylandt and Monmarche, *Position-dependent memory kernel in generalized
  Langevin equations*, https://arxiv.org/abs/2201.02457, provides the formal
  route to a position-dependent GLE if instantaneous state remains
  insufficient.

## Fixed microscopic data

Use exactly the data already frozen by the smooth-cage event-clock test:

```text
temperature        = 0.58
parents            = 5 independent KA restarts
clones per parent  = first 8
targets            = the same 64 A particles, seed 20260714
observations       = 5 x 8 x 64 = 2560
event threshold    = 0.08
event half-window  = 8 frames = 0.4 tau
recrossing radius  = sqrt(0.08)
horizon            = 20 tau
regularization     = L2 = 1, intercept unpenalized
validation         = leave one complete parent out
```

The cached smooth-center first-passage times and the reconstructed frame-zero
positions are authoritative. No trajectory is relabeled and no threshold is
changed after inspecting the structural result.

## Species-resolved radial state

For target particle `i`, species `a` in `{A,B}`, and fixed shell center
`rho_n`, define

```text
R_i(a,n) = sum_{j != i, type(j)=a}
             f_c(r_ij)
             exp[-(r_ij-rho_n)^2 / (2 sigma_r^2)],

f_c(r) = [1 + cos(pi r/r_c)]/2,  r < r_c,
         0,                       r >= r_c.
```

Minimum-image distances are computed from the full 4096-particle initial
configuration. The grid is inherited without tuning from the prior radial
committor diagnostic:

```text
rho_n  = [0.8, 1.05, 1.3, 1.55, 1.8, 2.05, 2.3]
sigma_r = 0.12
r_c     = 2.5
```

This gives 14 deterministic structural coordinates. For each parent-target
pair the radial vector is identical across the eight clones, as required for
an isoconfigurational precursor. Clone velocities must not enter it.

## Frozen model comparison

Fit the same censored exponential family used in the previous test,

```text
lambda_i = exp(beta_0 + beta dot z_i),
L = sum_i [lambda_i t_i - delta_i log(lambda_i)]
    + (1/2) beta^T P beta.
```

The primary models are:

```text
geometry         = (u,G),                         4 features
radial           = R,                            14 features
geometry_radial  = (u,G,R),                      18 features
```

The prior `geometry` result must be reproduced from the same observations.
`full_radial = (u,p,G,b,R)` may be reported only as a diagnostic; it cannot
decide the structural precursor gate because its fast clone-specific terms
already failed the previous ablation.

As a secondary isoconfigurational check, aggregate the eight clones for every
parent-target pair into an escape count at `20 tau` and fit the existing
grouped binomial logistic diagnostic. This produces 320 independent
configuration-target rows and tests propensity without treating repeated
structural vectors as distinct configurations.

## Integrity gates

The run is invalid unless all conditions hold:

1. Exactly 2560 clone-level observations and 320 parent-target rows are used.
2. The event/censor counts reproduce 1731/829 exactly.
3. Every radial feature is finite and clone-invariant within each
   parent-target pair to numerical tolerance `1e-12`.
4. The geometry censored-exponential metrics reproduce the frozen checkpoint
   to absolute tolerance `1e-12`.
5. Parent hashes, target indices, event parameters, and thermodynamic claim
   boundary match the cached manifests.

## Decision gates

Define `static_radial_precursor_allowed = 1` only if all primary conditions
hold on leave-one-parent-out predictions:

1. `geometry_radial` mean held-parent Brier skill is at least `0.01` larger
   than both `geometry` and `radial` alone.
2. Its mean held-parent Brier skill exceeds the frozen structural difficulty
   reference `0.026964`.
3. Its mean log-likelihood gain per observation over the constant-rate null is
   positive and every held parent has nonnegative total likelihood gain.
4. Its maximum held-parent survival-calibration error is at most `0.10`.
5. The aggregated binomial result has positive mean held-parent Brier skill
   and improves on aggregated geometry alone.

These gates require the species packing variables to explain residual
information, not merely duplicate the existing geometry signal.

Regardless of outcome, keep the following claims false in this single-
temperature experiment:

```text
event_clock_claim_allowed = 0
autonomous_single_particle_gle_claim_allowed = 0
kramers_escape_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Interpretation and next branch

If the gate passes, define a measured structural reaction coordinate

```text
S_i = beta_R dot standardize(R_i)
```

and treat it as a candidate microscopic precursor, not yet as a readiness
clock. The next required tests are its physical-time evolution, first-passage
law, and multi-temperature relation between `lambda(S,T)` and an activation
barrier.

If the gate fails, local radial packing is not the missing sufficient state.
Do not tune the radial grid. Compare a preregistered inherent-structure
soft-mode descriptor with a projected-history descriptor. If both static
routes fail, derive and estimate a position-dependent GLE memory kernel from
the many-particle trajectory rather than forcing a Markov single-coordinate
model.

## Deliverables

- A shared, tested species-resolved radial descriptor implementation.
- One analysis script that consumes the frozen initial-state and label caches.
- Clone-level censored-exponential model and survival CSVs.
- Parent-target aggregated committor CSVs.
- A machine-readable summary containing every integrity and claim gate.
- A scientific note that records either the positive precursor or the bounded
  negative result and routes the next microscopic experiment.
