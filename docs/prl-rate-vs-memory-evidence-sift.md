# PRL Rate-Versus-Memory Evidence Sift

## Decision question

The PRL-level question is narrower than whether any event-clock model can fit a
curve:

> Does low-temperature failure come from an incorrect mean jump rate, or from
> removing ordered cage-path and persistent-environment memory?

Evidence is eligible for the primary decision only when it uses independently
restarted trajectories at `T=0.45`, keeps calibration and held-out windows
separate, predicts MSD, NGP, and multi-`k` self scattering without a macro fit,
and includes an information-preserving null.

## Backlog audit

The uncommitted `codex/prl-event-clock-closure` backlog contains 583 files:
509 generated CSVs at `T=0.58`, 73 exploratory scripts, and one microscopic
response note. It contains no `T=0.45` artifact. None of that backlog is
eligible for the primary low-temperature mechanism decision.

Several entries remain useful microscopic canaries but should not be imported
into the PRL evidence set:

- the structural-rate marked-Poisson null fails held-out diffusion, NGP, and
  high-`k` scattering, but uses only `T=0.58` and held structural rates;
- the frozen-minimum response predicts most short-time cage-center increments
  from the KA pair potential, but does not test long-time low-temperature
  observables;
- active-bath and pair-memory experiments show omitted collective state and
  colored residuals, but use a small `T=0.58` target/clone sample;
- frozen-cage event clocks test diffusion only and do not close every replicate.

These are route-finding results, not PRL claim support.

## Eligible evidence

| Test | Information retained | Frozen result | Interpretation |
| --- | --- | --- | --- |
| One-step recoil surrogate, `T=0.58` | calibration rate, radial law, one-step recoil | passes: MSD `0.036`, NGP `0.096`, maximum `F_s` error `0.017` | the reduced clock is adequate in the warmer liquid |
| One-step recoil surrogate, `T=0.45` | same information and protocol | fails: MSD `3.40`, NGP `1.98`, maximum `F_s` error `0.537` | cooling creates missing information, not merely Monte Carlo error |
| Within-particle time shuffle, `T=0.45` | particle identity and one-block vector distribution; destroys temporal order | fails: MSD `6.67`, NGP `2.07`, maximum `F_s` error `0.660` | preserving particle-specific rates and jump statistics is insufficient |
| Direction-randomized path, `T=0.45` | every block length and its time order; destroys recoil directions | fails: MSD `9.83`, NGP `2.00`, maximum `F_s` error `0.714` | directional path organization is required |
| Contiguous calibration path, `T=0.45` | complete ordered single-particle block path | passes: MSD `0.056`, NGP `0.030`, maximum `F_s` error `0.021` | an ordered-path upper bound transfers without a macro fit |
| Radial plus full two-point spectral null, `T=0.45` | one-block radial law and complete Cartesian spectrum | passes MSD but fails higher-order observables in all three replicates | the missing channel includes a connected nonlinear path cumulant |
| Anchor semi-Markov versus schedule-matched geometry null | identical state clock, waits, radial draws, and particle mean-wait scales | exact anchor geometry improves low-temperature ensemble errors but both models fail | reversible returns matter, but a two-state anchor closure is not sufficient |

The source tables are
`renewal_cage_ka_replicates_T045_empirical_path_verdict.csv`,
`renewal_cage_ka_replicates_T045_nonlinear_path_verdict.csv`, both recoil verdicts,
and `renewal_cage_ka_anchor_semi_markov_gate.csv`.

## Current conclusion

The present data reject **mean event rate as the controlling low-temperature
variable**. This is stronger than observing anticorrelated jumps: nulls that
preserve particle identity, one-block statistics, and even the complete
two-point path spectrum fail while the ordered calibration path transfers.

The data do not yet prove that path memory alone is the complete microscopic
cause. At `T=0.45`, the waiting-time audit selects
`persistent_particle_environment_required`; its median persistent-environment
excess is `3.01 +/- 0.11`, versus `0.218 +/- 0.094` at `T=0.58`. The rejected
anchor closure also shows that one binary return/escape state cannot represent
all of this information.

The defensible statement is therefore:

> Low-temperature anomalies require ordered nonlinear cage-path memory beyond
> the mean rate, one-block jump law, and two-point spectrum. The remaining
> closure gap may contain longer single-particle memory, persistent environment
> memory, or spatial facilitation; the current tests do not separate them.

All microscopic-closure, spatial-facilitation, and thermodynamic-glass claim
flags remain zero.

## Next decisive experiment

Use the existing `T=0.45` and `T=0.58` replicate protocol and add one paired
segment-splice family. Split each calibration particle path into contiguous
segments of fixed length `L`, retain the ordered vectors inside every segment,
and randomly reassign segments among particles within the same one-block radial
stratum. This preserves local path cumulants up to `L` and the ensemble event
rate while destroying persistent particle identity beyond `L`.

Compare, with common seeds and unchanged held-out targets:

1. within-particle time shuffle: rate and identity without path order;
2. segment splice over a fixed `L` grid: local path memory without long-lived
   environment identity;
3. contiguous empirical path: path and environment upper bound.

If a finite `L` closes all observables, the controlling object is a finite path
memory kernel. If every splice fails while the contiguous path passes, persistent
environment identity is required. Only after this gate should a new microscopic
state or Langevin embedding be fitted.
