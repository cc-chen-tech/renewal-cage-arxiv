# Segment-Splice Paired-Excess Baseline Design

## Status and scope

This is a post-run exploratory diagnostic designed after the frozen
segment-splice run exposed an ensemble/full-path baseline inconsistency. It is
not a revision of the approved segment-splice protocol, its tolerances, or its
mechanism verdict. It consumes only the committed `T=0.45` replicate-score and
cell tables from PR #9 and performs no new trajectory simulation or fitting.

The diagnostic asks one narrower question: after centering each finite-horizon
score on the same replicate's full-path score, which short horizons show a
replicate-consistent excess loss of higher-order closure?

## Inputs and exact support

The source tables are:

- `data/renewal_cage_ka_replicates_T045_segment_splice_replicate_scores.csv`;
- `data/renewal_cage_ka_replicates_T045_segment_splice_cells.csv`; and
- `data/renewal_cage_ka_segment_splice_gate.csv`.

The required grid is exactly three replicates, two models, and lengths
`{1,2,5,10,25,50,125,250}`. The full-path length is `B=250`; finite lengths are
the first seven values. Every source row must keep the three original claim
flags at zero. Every cell must preserve the global source-segment schedule, and
the source verdict must keep `mechanism_state=mechanism_unresolved` and
`low_mechanism_identifiable_against_full_path_control=0`.

At `L=250`, within- and cross-particle surrogates represent the same complete
path ensemble. Their replicate scores must agree to absolute tolerance
`1e-12`. The common baseline is their arithmetic mean, which avoids choosing a
model label for numerically equivalent values.

## Paired excess statistic

For model `m`, replicate `r`, and finite segment length `L`, define

```text
b_r = [score_within(r,250) + score_cross(r,250)] / 2
e_m(r,L) = score_m(r,L) - b_r
```

The score is already dimensionless in frozen tolerance units. Positive excess
means that the finite-horizon null loses higher-order information relative to
the complete calibration path from the same replicate. The additive contrast
is primary. A ratio is not reported because the replicate baselines differ by
more than a factor of three and ratio centering would over-weight the smallest
baseline.

For each `(m,L)`, report all three excesses, their arithmetic mean, sample
standard error, and a two-sided Student-t 95% confidence interval with
`df=2` and `t_0.975=4.302652729911275`:

```text
SE = sample_std(e_1,e_2,e_3) / sqrt(3)
CI95 = mean(e) +/- 4.302652729911275 SE
```

Set `paired_degradation_identified=1` only when `CI95_low>0`. A confidence
interval that includes zero is `unresolved`; it is not evidence of equivalence
or noninferiority. No equivalence margin is introduced after observing the
data.

## Owner-identity contrast

For each replicate and finite length, also report

```text
d(r,L) = score_cross(r,L) - score_within(r,L)
```

This contrast algebraically cancels the full-path baseline. Aggregate it by
first averaging the seven lengths inside each replicate and then computing the
three-replicate mean, standard error, and t95 interval. It remains exploratory
information evidence because neither absolute closure nor the replicate-level
full-path baseline passes.

## Fail-closed interpretation

The output must separate four facts:

1. exact paired input and full-path model agreement;
2. whether short-horizon degradation is identified relative to baseline;
3. whether the full-path baseline closes in every independent replicate; and
4. whether any microscopic state addition is allowed.

The first two may be positive while the latter two remain negative. The
classifier must keep all of the following at zero:

```text
paired_excess_equivalence_claim_allowed
independent_replicate_memory_lower_bound_claim_allowed
finite_memory_state_addition_allowed
owner_identity_sufficiency_claim_allowed
microdynamic_closure_claim_allowed
spatial_facilitation_claim_allowed
thermodynamic_claim_allowed
```

It may set `short_horizon_information_loss_supported_exploratory=1` only when
the identified-degradation lengths form a contiguous prefix of the frozen
finite grid for both models. This is an exploratory design constraint for a
later microscopic model, not a validated memory time. It may set
`owner_identity_information_supported_exploratory=1` only when all 21 owner
contrasts are positive and the replicate-first t95 lower bound is positive.

The mechanism state remains `mechanism_unresolved`. The next required action is
`replicate_resolved_full_path_baseline_or_new_trajectory_validation`, not adding
a finite-memory coordinate to the Langevin model.

## Outputs

The implementation produces:

- `scripts/summarize_ka_segment_splice_paired_excess.py`;
- `data/renewal_cage_ka_segment_splice_paired_excess_rows.csv`;
- `data/renewal_cage_ka_segment_splice_paired_excess_gate.csv`;
- `figures/renewal_cage_ka_segment_splice_paired_excess.svg`;
- `docs/segment-splice-paired-excess.md`; and
- focused unit and arXiv artifact recomputation tests.

The SVG plots paired excess and t95 intervals against `tau_L=20L`, includes a
zero-baseline line, labels unresolved intervals explicitly, and states that the
analysis is post-run exploratory and conditional on the preserved global
source-segment schedule.
