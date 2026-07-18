# Segment-Splice Memory Gate Design

> **Revision note.** The original approved design was committed as `dff26e2`.
> Later edits only make two claim-boundary corrections explicit: (a) both nulls
> retain one global source-segment schedule, and (b) the primary low-temperature
> decision is reported separately from high-temperature crossover support.
> These corrections do not change trajectory inputs, segment-length grids,
> seeds, the nested 16-to-64 realization protocol, precision limits, or curve
> tolerances. Baseline-identifiability diagnostics and the exploratory paired
> owner-identity effect were observed after the run and are reported only in the
> result CSV and result note; they are not retroactively added as preregistered
> decision rules.

## Purpose

Determine whether the remaining low-temperature closure gap is controlled by a
finite ordered single-particle path memory or by persistent particle/environment
identity. The gate compares two calibration-only segment surrogates that retain
the same local ordered path pieces but differ in whether those pieces stay with
one particle.

This is an information-hierarchy test, not a new fitted theory. It may identify
the minimum information that a later Langevin or generator closure must retain.
It cannot by itself identify a microscopic force variable or prove spatial
facilitation. The mandatory claim flags remain

```text
microdynamic_closure_claim_allowed = 0
spatial_facilitation_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

## Existing evidence

The frozen empirical-path protocol already establishes the following:

- the ordered calibration path closes held-out MSD, NGP, and multi-`k` self
  scattering at `T=0.45` and `T=0.58` without a macro fit;
- a within-particle block shuffle fails at both temperatures, so one-block
  particle-specific statistics are insufficient;
- a direction-randomized path fails, so ordered directional structure matters;
- at `T=0.45`, a null preserving the one-block radial law and complete
  Cartesian two-point spectrum still fails higher-order observables in every
  replicate;
- the two-state anchor-aware semi-Markov closure improves the low-temperature
  higher-order score but fails the complete held-out closure gate; and
- the waiting-time audit selects persistent particle environment at `T=0.45`,
  but that statistic does not separate long single-particle path memory from a
  slowly evolving environment.

The unresolved question is therefore not whether path order matters. It is the
duration and ownership of the required path information.

## Frozen trajectory protocol

- Use the existing independently restarted type-A Kob-Andersen ensembles:
  three replicates at `T=0.45` and five at `T=0.58`.
- Use calibration times `5000` and `750`, respectively, with equal held-out
  halves. No held-out trajectory value may enter a surrogate.
- Reuse unwrapped positions, physical lag time, block size `20`, and wave
  numbers `k={2,4,7.25}` from the empirical-path transfer protocol.
- For particle `i`, define the calibration block path
  `X_i=(x_i,0,...,x_i,B-1)`, where each `x_i,b` is the three-dimensional
  displacement over one block. This gives `B=250` at `T=0.45` and `B=37` at
  `T=0.58`; an incomplete terminal block is excluded.
- Reuse only lags divisible by `20` that are present in the frozen held-out
  factorization tables.
- Aggregate realizations within a replicate before aggregating independent
  replicates. Keep every replicate-level result and Monte Carlo error.
- Reuse the existing early/late, early/held-out, and late/held-out stationarity
  controls. A failed stationarity control makes the memory interpretation
  unresolved.

The segment-length grids are fixed in block units:

```text
T=0.45: L = 1, 2, 5, 10, 25, 50, 125, 250
T=0.58: L = 1, 2, 4, 8, 16, 32, 37
```

The corresponding memory horizon is `tau_L=20 L` trajectory samples. The last
entry at each temperature is the complete-path upper-bound control, not a
finite-memory candidate.

## Segment representation

For a fixed `L`, split every particle path at deterministic boundaries into
ordered segment tokens

```text
S_i,j = (x_i,jL, ..., x_i,min((j+1)L,B)-1).
```

Every token records its source particle, source ordinal, length, and an exact
hash of its ordered vectors. The shorter terminal token, when present, is kept;
no padding, wrapping, truncation, interpolation, reversal, or vector rotation
is allowed.

All observables are measured on the reconstructed complete paths. Windows that
cross a new segment seam are included. Excluding them would hide the dependence
that the null is designed to remove.

## Paired surrogate family

Each `(temperature, replicate, L, realization)` uses one recorded seed and a
common tokenization. Random assignments are constrained permutations, so every
source token is used exactly once.

### Within-particle segment shuffle

`within_particle_segment_shuffle` randomly reorders the segment tokens of each
particle and concatenates them. It therefore retains:

- particle identity for every vector;
- the exact vector multiset of every particle;
- the exact ordered contents of every segment; and
- all adjacent-vector pairs internal to a segment.

It destroys segment order and all original adjacencies across segment seams.
When a particle has more than one segment, its segment permutation must differ
from the identity. For `L=B`, the path is unchanged and is labeled a trivial
upper-bound control.

### Cross-particle segment splice

`cross_particle_segment_splice` starts from the same per-particle sequence of
target segment lengths as the paired within-particle realization. It fills each
target slot with a source token of the same length from the replicate-wide pool,
subject to all of the following constraints:

- the source particle differs from the target particle;
- adjacent target segments do not have the same source particle;
- every source token is used exactly once; and
- every target path contains exactly `B` blocks.

This retains the global vector multiset, exact ordered segment contents, segment
length layout, ensemble block rate, and local path cumulants inside a segment.
It removes persistent ownership of consecutive segments by one particle. A
constrained assignment failure after a fixed maximum of 100 deterministic
restarts invalidates that realization; it must not be repaired by relaxing a
constraint.

At `L=B`, whole particle paths are merely permuted. Ensemble observables must
then equal the contiguous-path upper bound to numerical precision. This row is
an implementation control and cannot support an environment-memory claim.

## Exact information gates

Before any held-out curve is inspected, every randomized realization must pass
the following mechanical checks:

- source token count equals target token count;
- minimum and maximum source-token reuse counts both equal one;
- ordered token-hash multiset is exactly preserved;
- global block-vector hash multiset is exactly preserved;
- segment-length histogram is exactly preserved;
- internal adjacent-pair hash multiset is exactly preserved;
- every reconstructed particle path has exactly `B` finite three-dimensional
  blocks; and
- no held-out path, held-out event, or macro-fit parameter is used.

The within-particle model must additionally preserve each particle's block-vector
multiset exactly. The cross-particle model must have zero same-source assignment
fraction and zero adjacent same-source segment fraction for every nontrivial
`L`.

Both frozen nulls use the same source-segment ordinal schedule for every target
particle. Record `global_source_segment_schedule_preserved=1` mechanically.
Any substantive interpretation is conditional on this preserved global
schedule and is not an unconditional single-particle memory claim.

Report the guaranteed internal-adjacency fraction and verify that it equals the
analytic fraction of pairs internal to unchanged segment tokens. Report any
original adjacency recreated at a randomized seam separately as an accidental
match; it is not counted as guaranteed retained information.

## Monte Carlo protocol

Run 16 deterministic realizations per randomized cell first. If any randomized
cell at either temperature fails a Monte Carlo precision limit, extend every
randomized cell at both temperatures to 64 realizations using a nested seed
sequence whose first 16 seeds are unchanged. No cell may be extended selectively.

The frozen precision limits are:

- maximum ensemble relative MSD Monte Carlo standard error `0.01`;
- maximum ensemble NGP Monte Carlo standard error `0.03`; and
- maximum ensemble `F_s` Monte Carlo standard error `0.003` over every `k`.

A precision failure after 64 realizations makes the complete mechanism gate
unresolved; it is not a curve failure and cannot be used for mechanism
selection.

## Held-out scoring

For each model and `L`, use `cumulative_block_observables` on all contiguous
windows of the reconstructed calibration paths and compare with the unchanged
held-out targets. The curve limits remain:

- maximum relative MSD error `0.10`;
- maximum absolute NGP error `0.30`; and
- maximum absolute `F_s(k,t)` error `0.03` over all three wave numbers.

A model-length cell passes only when its exact information gates, stationarity
gate, Monte Carlo precision gate, and all three held-out curve gates pass.
Replicate-level higher-order scores

```text
max(max_lag NGP_error / 0.30,
    max_lag,k Fs_error / 0.03)
```

are retained for paired comparisons. Ensemble closure is mandatory; a favorable
replicate score cannot rescue a failed curve gate.

## Memory-length selection

For model `m` and temperature `T`, define `L_m*(T)` as the smallest nontrivial
grid value such that `m` passes at that value and at every larger nontrivial
grid value. An isolated pass followed by a failure is not a memory length.

`L=B` is excluded from `L_m*`. A finite length is identified only if at least
one original seam is destroyed and at least two ordered segments remain in each
particle path. If no such monotone pass tail exists, record `L_m*=unresolved`.

Because the fixed grid is evaluated against held-out curves, `L_m*` is a
mechanism-localization result, not a parameter validated for a predictive
theory. A later parametric closure must freeze the selected horizon and test it
on independent trajectories or a second system.

## Decision table

The primary decision uses `T=0.45`; `T=0.58` provides the warmer-liquid
crossover control.

| Frozen outcome | Verdict | Interpretation |
| --- | --- | --- |
| Both finite lengths resolve and `L_cross* = L_within*` | `finite_single_particle_path_memory_sufficient_conditional_on_global_schedule` | Conditional on the preserved global source-segment schedule, local ordered path pieces close the tested observables; persistent ownership is not additionally required for tolerance-level closure. |
| `L_within*` resolves and `L_cross*` is unresolved or larger | `persistent_environment_identity_required_beyond_local_path` | Keeping segments with one particle carries information beyond a local path horizon. |
| Both finite lengths are unresolved | `longer_or_richer_path_state_required` | The tested finite segment horizons do not close; a longer path state, nonstationarity, or omitted collective variable remains. |
| `L_cross*` resolves while `L_within*` is unresolved, or `L_cross* < L_within*` | `null_family_pathology_unresolved` | The nominally more destructive null performs better, so the intended information ordering is not identified. |
| Any provenance, exactness, stationarity, support, or precision gate fails | `mechanism_unresolved` | The surrogate family cannot support a physical conclusion. |

For either substantive low-temperature verdict, the selected model cells must
have higher-order score at most one in all three low-temperature replicates. The
persistent-environment verdict additionally requires the within-particle score
to be strictly smaller than the cross-particle score in every low-temperature
replicate at `L_within*`; ties do not count. The result must also report whether
each selected low-temperature horizon is strictly longer than its
high-temperature counterpart. Absence of that crossover blocks a cooling-induced
memory claim but does not erase the temperature-specific result.

No result from this table enables a spatial-facilitation claim. In particular,
`persistent_environment_identity_required_beyond_local_path` groups static or
slow particle disorder, evolving local environment, and spatially cooperative
facilitation until a later observable separates them.

The stored output separates `low_temperature_mechanism_state` from the global
preregistered `mechanism_state`. A failed T=0.58 support gate sets
`high_temperature_control_resolved=0` and blocks cooling-memory growth, but it
does not erase a complete temperature-specific T=0.45 verdict. A failed T=0.45
support or precision gate still makes the low-temperature state unresolved.

## Outputs

The implementation will produce:

- one focused segment-permutation module under `src/`;
- one two-temperature analysis script and one verdict/figure summarizer;
- per-realization exact-quality CSVs;
- per-replicate prediction and paired-score CSVs;
- per-temperature length-scan summaries and verdicts;
- one two-temperature mechanism-selection CSV; and
- one deterministic SVG showing normalized MSD, NGP, and maximum multi-`k`
  scattering errors versus `tau_L` for both surrogate families.

Every artifact records temperature, calibration time, block size, segment
length, complete block count, replicate count, realization count, seed,
stationarity status, all provenance fields, and all three claim-boundary flags.
The arXiv-package test must recompute the final verdict from committed source
tables rather than trusting stored pass flags.

## Tests

Pure tests must cover exact token accounting, variable terminal-segment lengths,
within-particle multiset preservation, constrained cross-particle assignment,
adjacent-source exclusion, deterministic nested seeds, inclusion of seam-crossing
windows, and the `L=B` degenerate control. Invalid dimensions, nonfinite vectors,
impossible assignments, duplicate tokens, and incomplete paths must fail loudly.

Analysis tests must cover replicate-first aggregation, all information and
precision gates, monotone-tail length selection, every decision-table branch,
the warmer-liquid crossover label, held-out exclusion, zero macro-fit parameters,
and immutable microscopic, spatial, and thermodynamic claim boundaries.

## Explicit non-goals

This change will not add a new latent state, fit a memory kernel, infer a local
softness variable, run a new many-particle simulation, modify the Langevin
equation, or update the manuscript's central claim. Those steps are conditional
on this gate identifying which information survives the low-temperature nulls.
