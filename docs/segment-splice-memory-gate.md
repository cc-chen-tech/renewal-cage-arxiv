# Segment-splice path-memory gate

## Question and information hierarchy

This gate asks whether held-out displacement observables can be recovered from
finite contiguous pieces of the calibration path. It compares two nulls:

- `within_particle_segment_shuffle` keeps each segment with its original
  particle but permutes the segment order; and
- `cross_particle_segment_splice` also changes the segment owner at every slot
  and forbids adjacent reuse of one source owner.

Both nulls deliberately use one shared permutation of source-segment ordinals
for all particles. The audit records
`global_source_segment_schedule_preserved=1`. Therefore any substantive result
would be conditional on the preserved global source-segment schedule. This
experiment does not destroy calibration-time synchronization across particles
and cannot establish an unconditional single-particle memory closure.

## Frozen protocol

The calibration paths use Type-A particles, block size 20, and no held-out path
as a prediction input. At `T=0.45`, the path has 250 blocks, three independent
replicates, and segment lengths `1,2,5,10,25,50,125,250`. At `T=0.58`, it has
37 blocks, five replicates, and lengths `1,2,4,8,16,32,37`. The targets are
held-out MSD, NGP, and `F_s(k,t)` at `k=2,4,7.25`.

All cells were first evaluated with 16 paired deterministic realizations. At
least one precision gate failed, so the complete two-temperature grid was
rerun with the nested realization set `0..63`. A cell requires relative MSD
error at most 0.10, absolute NGP error at most 0.30, and maximum absolute
scattering error at most 0.03. Its Monte Carlo standard errors must be at most
0.01, 0.03, and 0.003, respectively. The full-path value `L=B` is an upper
control and is never selectable as a finite memory length.

## Real-data result

Every exact provenance gate passed at both temperatures, including one-time
token reuse, block-vector multiset preservation, complete reconstructed paths,
and the preserved global source-segment schedule. All three T=0.45
stationarity controls passed.

The 64-realization T=0.45 precision gate nevertheless failed for the
within-particle model at `L=1` and `L=2`: the maximum ensemble relative MSD
Monte Carlo standard errors were 0.01551 and 0.01183, above 0.01. The remaining
low-temperature cells met the frozen precision limits. At the curve level,
both models failed every non-full length. Only `L=250` passed, and that value is
the excluded full-path control. For example, at `L=125` the within-particle
model had maximum normalized errors corresponding to MSD 0.03755, NGP 0.3470,
and scattering 0.06927; NGP and scattering still exceeded their tolerances.

At T=0.58, all Monte Carlo precision gates passed. Several intermediate cells
passed the raw curve limits, but `L=32` failed the scattering limit for both
models and the full path is excluded, so there is no selectable monotone tail.
More importantly, `early_late` and `early_heldout` stationarity controls had
already failed. The high-temperature crossover control is therefore unresolved
without altering any tolerance.

The committed verdict is consequently:

```text
L_within*(T=0.45) = unresolved
L_cross*(T=0.45)  = unresolved
low_temperature_mechanism_state = mechanism_unresolved
high_temperature_control_resolved = 0
mechanism_state = mechanism_unresolved
```

The full-path control also fails at the independent-replicate level even though
its ensemble cell passes. The full-path higher-order scores are approximately
0.8995, 2.0953, and 3.3236 for replicates 1, 2, and 3, so two of three
replicates fail the unit tolerance. The committed fields therefore set
`low_full_path_control_ensemble_pass=1`,
`low_full_path_control_all_replicates_pass=0`,
`low_full_path_control_failed_replicate_count=2`, and
`ensemble_cancellation_detected=1`. This baseline prevents a paired mechanism
claim: the ensemble pass can arise through replicate averaging, and no
independent-replicate memory lower bound is allowed.

One post-run signal is retained as explicitly exploratory. For all 21 pairs of
three replicates and seven finite lengths, the cross-particle higher-order score
is larger than the within-particle score. Averaging the seven differences
within each replicate first gives a mean difference of 2.071862 tolerance
units, standard error 0.277733, and a two-sided t95 interval
`[0.876874, 3.266850]`. This supports the statement that owner identity carries
additional information within these nulls. It does not establish that owner
identity is sufficient: absolute closure and the full-path replicate baseline
both fail. It also cannot distinguish static particle disorder, finite-exchange
environment memory, or spatial facilitation.

This result does not identify finite single-particle memory. In particular,
the ensemble rejection at the largest non-full horizon (`L=125`,
`tau_L=2500`) is descriptive and is not an independent-replicate lower-bound
claim. The next useful calculation is a paired excess-error gate relative to a
replicate-resolved full-path baseline, followed by a null that independently
randomizes the global source-segment schedule. Any such extension is
exploratory unless preregistered on new trajectories.

## Claim boundary

The segment-splice gate is an information-hierarchy diagnostic, not a
microscopic Langevin derivation. Every source and verdict table keeps
`microdynamic_closure_claim_allowed=0`,
`spatial_facilitation_claim_allowed=0`, and
`thermodynamic_claim_allowed=0`. The verdict also fixes
`owner_identity_sufficiency_claim_allowed=0`,
`independent_replicate_memory_lower_bound_claim_allowed=0`, and
`static_vs_finite_exchange_resolved=0`.
