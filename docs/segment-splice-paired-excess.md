# Segment-splice paired-excess baseline

## Question

The frozen segment-splice gate found that every finite low-temperature segment
length failed the absolute curve tolerances, while the excluded full-path
control passed only after ensemble averaging. This post-run diagnostic asks a
narrower question: after subtracting each replicate's own full-path score, which
finite horizons still show extra higher-order loss?

For model `m`, replicate `r`, and finite length `L`,

```text
excess_m(r,L) = score_m(r,L) - score_full_path(r)
```

The full-path score is the average of the within-particle and cross-particle
full-path controls. Those two controls agree to `1.89e-15`, so the baseline is
numerically shared.

## Result

The paired inputs are complete and preserve the global source-segment schedule.
The full-path baseline still fails independent-replicate closure: two of three
low-temperature replicates have full-path scores above the unit tolerance.
Therefore the mechanism state remains

```text
mechanism_unresolved
```

The excess analysis identifies a short-horizon information-loss prefix through
`L=10`, corresponding to `tau_L=200`, for both segment nulls. At longer finite
horizons the replicate-first t95 intervals include zero, so they are
unresolved. Owner identity remains an exploratory signal: all 21 cross-minus-
within contrasts are positive, with replicate-first mean `2.071862` and t95
interval `[0.876874, 3.266850]`.

## Interpretation

This result says that very short retained path pieces lose information beyond
the full-path replicate baseline. It does not say that a finite memory state is
sufficient, because the full-path replicate baseline itself does not close. It
also does not separate static particle disorder, finite-exchange environmental
memory, and spatial facilitation.

All claim gates remain closed:

```text
paired_excess_equivalence_claim_allowed = 0
independent_replicate_memory_lower_bound_claim_allowed = 0
finite_memory_state_addition_allowed = 0
owner_identity_sufficiency_claim_allowed = 0
microdynamic_closure_claim_allowed = 0
spatial_facilitation_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

The next required step is
`replicate_resolved_full_path_baseline_or_new_trajectory_validation`.
