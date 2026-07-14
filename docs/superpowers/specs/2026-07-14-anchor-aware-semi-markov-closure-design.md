# Anchor-Aware Semi-Markov Closure Design

## Purpose

Test whether a directly observed, finite-lifetime reversible cage state is the
minimum calibration-only extension that closes low-temperature Kob-Andersen
dynamics after the one-step recoil Markov null has failed. The model must predict
held-out MSD, NGP, and multi-wave-number self-intermediate scattering from event
statistics without fitting any macro observable.

This is a dynamical closure test. Even if it passes,
`microdynamic_closure_claim_allowed=0`,
`spatial_facilitation_claim_allowed=0`, and
`thermodynamic_claim_allowed=0` remain mandatory because the event definition
and the cage-residual channel are measured from calibration trajectories.

## Existing evidence and scope

The preceding frozen gate established all of the following:

- the calibration-only one-step radial recoil Markov surrogate passes its
  information-preservation checks at `T=0.45` and `T=0.58`;
- it closes held-out MSD, NGP, and `F_s(k,t)` at `T=0.58`;
- at `T=0.45` it fails held-out NGP and multi-k scattering, with a large MSD
  failure that the next model must also repair;
- the complete ordered calibration path is a zero-macro-fit upper bound that
  closes the low-temperature curves;
- the primary cage-return fraction is `0.429-0.439` at `T=0.45` and
  `0.052-0.059` at `T=0.58`, with every low-temperature replicate exceeding
  its length-preserving isotropic null by at least `1.35`.

An additional calibration audit found that return and escape events have
different waiting-time and jump-length laws in every replicate. At `T=0.45`,
mean return waits are about `99` while mean escape waits are `155-161`; at
`T=0.58`, they are about `31-34` and `59-62`. Return probability also decreases
monotonically across waiting-time quartiles. A memoryless binary Markov chain is
therefore not an adequate candidate: the state and holding time must be modeled
jointly.

## Frozen data protocol

- Use the existing independently restarted type-A Kob-Andersen ensembles:
  three replicates at `T=0.45` and five at `T=0.58`.
- Use calibration times `5000` and `750`, respectively, and equal held-out
  halves. No held-out event, displacement, count, MSD, NGP, or scattering value
  may enter the model.
- Reuse the replicate-specific calibration Debye-Waller factors and the existing
  non-recrossing cage-jump extractor with fluctuation half-window `5`.
- Use physical lag time, block size `20`, and wave numbers `k={2,4,7.25}`.
- Use 16 deterministic synthetic realizations per replicate. Seeds must be
  generated from one recorded base seed and the replicate and realization
  indices.
- Aggregate predicted and observed macro curves across independent replicates
  before the primary decision. Preserve replicate-level rows and Monte Carlo
  errors.
- This protocol is frozen after the cage-anchor discovery audit. The result is a
  reproducible confirmatory follow-up, not a prospectively preregistered
  discovery.

## Directly observed state

Events are ordered separately for each particle. For event `n`, let `a_n` and
`b_n` be its measured pre- and post-cage centers and let
`v_n = b_n - a_n`. Let `a_DW` be the square root of that replicate's calibration
Debye-Waller factor.

For every event after the first event of a particle, define

```text
return: |b_n - a_(n-1)| = |v_(n-1) + v_n| <= a_DW
escape: |b_n - a_(n-1)| > a_DW
```

The departure cage `a_n` becomes the geometric anchor for the next event. A
return therefore implements reversible motion between recent cage centers; an
escape advances the local cage topology. No hidden-state inference or macro
observable is used to assign the state.

For each transition record the current and next state, inter-event holding time,
source and target jump radii, source-radius rank bin, relative cosine, and return
closure distance. The first transition of each particle is used only for
initialization and never joined to another particle.

## Semi-Markov kernel

The model is a Markov-renewal process on the observed `return/escape` state. It
samples one empirical joint transition record conditional on the current state
and one of eight equal-count source-radius bins. The sampled record supplies
jointly:

```text
(next_state, normalized_holding_time, target_radius_quantile, geometry)
```

The holding time is normalized by the source particle's calibration mean wait.
When generating particle `i`, it is multiplied by that same particle's mean.
This retains static particle-to-particle mobility dispersion while pooling the
state law strongly enough to represent rare high-temperature returns.

The target radius is obtained by mapping the sampled radial quantile through the
source particle's calibration jump-radius distribution. This preserves each
particle's radial scale without requiring a separately fitted fast/slow state.

For a return transition, the geometry is the sampled closure distance `d`.
Given current radius `r` and target radius `r'`, use

```text
cos(theta) = (d^2 - r^2 - r'^2) / (2 r r')
```

and draw a uniform three-dimensional azimuth. Only empirically supported tuples
with `|cos(theta)| <= 1` are admissible. No clipping is allowed; an empty
conditional support fails the realization.

For an escape transition, use the sampled empirical relative cosine and a
uniform three-dimensional azimuth, as in the validated recoil kernel. Every
generated event advances physical time by its sampled holding time. Simulation
continues through the calibration-duration observation window plus the maximum
scored lag so that every time origin has complete support.

## Particle disorder and initialization

Each particle retains its calibration mean waiting time, empirical jump-radius
quantile function, and initial observed event vector. The pooled kernel contains
only dimensionless state, holding-time, radial-rank, and angular information.
Consequently the comparison does not erase the persistent particle environment
already detected by the waiting-time audit.

Particles with fewer than two events cannot initialize an event path. Their
calibration no-event probability is retained as an immobile component rather
than silently dropping them. The model reports active and immobile particle
counts for every replicate.

## Competing mechanisms

All models use the same calibration and held-out windows, realizations, time
origins, wave numbers, and error metrics.

1. `one_step_radial_recoil_markov` is the already frozen null.
2. `state_schedule_without_anchor_geometry` samples the same semi-Markov state,
   holding-time, and target-radius records, but generates every return-labelled
   vector from the ordinary one-step recoil geometry. It preserves the state
   clock while removing exact cage closure. Its scheduled state remains an
   internal control label; the geometrically measured return fraction is
   reported separately and is not forced to equal that schedule.
3. `anchor_aware_semi_markov` uses the full directly observed kernel above.
4. `contiguous_empirical_path` remains the calibration-only nonparametric upper
   bound. It is not a fitted candidate and cannot be selected as the theory.

This comparison separates three outcomes. If the state-schedule null closes the
curves, semi-Markov timing is sufficient and anchor geometry is not identified.
If only the anchor-aware model closes them, reversible cage geometry is required
within this candidate set. If the anchor-aware model fails, the model is rejected
and spatial or richer path state remains necessary.

## Calibration-only cage residual

Convert each synthetic event sequence to an event-center displacement path and
measure its displacement moments and characteristic functions over physical lag
time. Combine these with the existing replicate-specific calibration residual
MSD, NGP, and characteristic functions through the established
`compound_jump_cage_observables` factorization.

The held-out factorization table supplies observed targets only. The prediction
uses the calibration residual row at the same lag. Every output records
`external_cage_channel_used=1`, `calibration_cage_residual_transfer=1`, and
`heldout_cage_residual_used_in_prediction=0`. This measured residual channel is
why a successful result is event-level dynamical closure rather than a complete
microscopic derivation.

## Calibration quality gates

The anchor-aware model and the state-schedule null are valid only if every
replicate satisfies all of the following shared checks before held-out curves
are examined:

- all particles and all event transitions are accounted for exactly once;
- generated scheduled-state fraction absolute error at most `0.02`;
- generated scheduled `P(return|return)` and `P(return|escape)` errors at most
  `0.03`;
- return and escape mean holding-time relative errors at most `0.05`;
- maximum relative error over holding-time quantiles
  `{0.25,0.50,0.75,0.90}` at most `0.10` for each state;
- radial mean and standard-deviation relative errors at most `0.02`;
- lag-one cosine mean error at most `0.02` and maximum cosine-quantile error at
  most `0.03`;
- no unsupported conditional tuple, clipped cosine, cross-particle transition,
  held-out event, or macro-fit parameter is used.

The anchor-aware model has two additional geometric checks: its geometrically
measured return fraction must have absolute error at most `0.02`, and its return
closure-distance quantiles divided by `a_DW` must have maximum absolute error at
most `0.05`. For the state-schedule null, both geometric quantities are reported
but deliberately not gated because matching them would restore the information
the null is designed to remove.

Failure of any quality gate makes the mechanism unresolved, regardless of its
macro error.

## Held-out observables and decision

Score the unchanged ensemble tolerances over every common lag:

- maximum relative MSD error `0.10`;
- maximum absolute NGP error `0.30`;
- maximum absolute `F_s(k,t)` error `0.03` over all three wave numbers.

Monte Carlo standard errors must be at most relative MSD `0.01`, NGP `0.03`,
and scattering `0.003`. A model closes one temperature only when all quality,
precision, and curve gates pass. Low-temperature replicate-level higher-order
scores

```text
max(NGP_error / 0.30, maximum_Fs_error / 0.03)
```

must improve over the one-step recoil null in every replicate; ties do not count
as improvement.

Classify the combined result as follows:

- `anchor_geometry_required_within_tested_models`: anchor-aware closes both
  temperatures, the state-schedule null fails low-temperature NGP or scattering,
  and every low-temperature replicate improves over one-step recoil;
- `semi_markov_state_clock_sufficient_anchor_not_identified`: both semi-Markov
  models close both temperatures;
- `anchor_aware_model_rejected`: the anchor-aware model fails any quality,
  precision, or held-out curve gate;
- `mechanism_unresolved`: provenance, support, or competing-model completeness
  is insufficient.

The low-temperature MSD gate is mandatory. Closing only NGP or scattering cannot
rescue the model.

## Outputs

The implementation produces:

- one focused core module for observed-state extraction and synthetic transition
  generation;
- one per-temperature analysis script;
- per-replicate transition, quality, macro-curve, summary, and verdict CSVs;
- one two-temperature mechanism-selection CSV;
- one deterministic SVG showing each model's errors normalized by the three
  frozen tolerances;
- one arXiv-package test that recomputes the final verdict from committed input
  artifacts rather than trusting stored pass flags.

Every artifact records temperature, calibration time, Debye-Waller source,
replicate count, state definition, radial-bin count, realization count, seeds,
provenance flags, effective support, and all claim-boundary flags.

## Tests

Pure tests must establish state labels from exact cage-center paths, prevent
cross-particle transitions, preserve particle waiting and radial scales, generate
isotropic noncoplanar azimuths, enforce return closure without clipping, and
reject empty conditional support. Determinism is tested for fixed seeds.

Decision-table tests independently toggle every quality, precision, curve,
replicate-improvement, and competing-null condition. Artifact tests freeze the
real result, verify exact calibration/held-out provenance, reject duplicate or
missing realization rows, recompute numerical maxima, and preserve all three
claim boundaries.

## SOTA comparison boundary

The tested mechanism is aligned with, but more restrictive than, four primary
observations:

- Kob-Andersen trajectories show correlated successive jumps and cooling-induced
  continuous-time-random-walk failure
  (`https://www.nature.com/articles/srep11770`).
- Reversible jumps return particles toward former average positions, with more
  irreversible cage changes at higher temperature
  (`https://arxiv.org/abs/cond-mat/0308601`).
- Rare cage escapes control relaxation deep in the supercooled regime
  (`https://journals.aps.org/prx/abstract/10.1103/7m7x-zxqv`).
- Wave-number-resolved experiment finds that cage-scale secondary motion and
  alpha relaxation are not independent
  (`https://www.nature.com/articles/s41567-026-03320-5`).

Agreement means that one calibration-only event kernel predicts the held-out
transport and relaxation observables with the frozen tolerances. Qualitative
similarity, a successful single wave number, or closure at only one temperature
does not count. The Kob-Andersen result cannot by itself establish universal
Johari-Goldstein behavior in molecular liquids.

## Explicit exclusions

This design does not derive configurational entropy, a Kauzmann temperature,
heat-capacity anomalies, an ideal glass transition, a spatial four-point length,
or facilitation fronts. Static-environment and cross-particle spatial alternatives
remain unexcluded unless the same-data competing tests reject them. No result
from this design permits the statement that the model explains all glass
transition phenomena.
