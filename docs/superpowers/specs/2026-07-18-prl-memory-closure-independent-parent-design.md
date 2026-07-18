# PRL Memory Closure Independent-Parent Preregistered Design

## Frozen scientific question and candidate claim

The primary question is fixed verbatim:

> What microscopic information is minimally required to reconstruct low-temperature glassy relaxation?

The candidate claim is fixed verbatim and may be enabled only by the final gate
defined below:

> In the low-temperature Kob-Andersen glass former, relaxation dynamics cannot
> be reconstructed from the mean event rate, one-step jump law, or two-point path
> spectrum. Accurate reconstruction requires ordered cage-path memory and a
> persistent particle-level environment.

This is a preregistration. No result may change the wording, parent rule,
observables, lag grid, calibration budget, tolerances, Monte Carlo rule, or
ablation logic below.

## Scope and immutable claim boundaries

The target is an event/block-path dynamical reconstruction claim for the
committed Kob-Andersen trajectories. It is not a first-principles microscopic
closure, a spatial-facilitation identification, or a thermodynamic glass
transition result.

The following fields remain zero for every row and every outcome:

```text
complete_microscopic_closure_claim_allowed = 0
spatial_facilitation_claim_allowed = 0
thermodynamic_glass_transition_claim_allowed = 0
```

The shorter machine-readable compatibility aliases
`microdynamic_closure_claim_allowed`, `spatial_facilitation_claim_allowed`, and
`thermodynamic_claim_allowed` also remain zero. A positive memory-closure result
would identify the information required within the tested single-particle model
family. It would not exclude cross-particle coupling, derive the cage-event
mapping, or explain a thermodynamic transition.

## Audit of evidence available before this preregistration

All numerical statements in this section are frozen historical inputs, not new
independent-parent confirmation.

| Evidence family | Frozen result | What it supports | What it does not prove |
| --- | --- | --- | --- |
| rate versus memory sift / one-step recoil | T=0.45 one-step recoil fails with maximum ensemble errors MSD 3.3982, NGP 1.9760, Fs 0.53665; T=0.58 passes its raw curves | mean-rate and one-step information are inadequate in the low-temperature restart ensemble | independent-parent failure, a unique missing mechanism |
| empirical path transfer | at T=0.45 contiguous calibration paths pass 0.05580/0.02965/0.02128 while within-particle shuffle fails 6.6652/2.0747/0.66007 | ordered calibration paths contain transferable information beyond particle-specific one-block marginals | a finite parametric closure or parent-independent effect |
| nonlinear path / two-point spectrum null | the T=0.45 radial multivariate surrogate preserves the radial law and spectrum within frozen quality limits, passes MSD, and fails higher-order observables in all three restart labels | the missing information is beyond the one-block radial law and two-point spectrum | independent-parent nonlinear-memory necessity |
| memory hierarchy | historical artifact selects `particle_conditioned_finite_exchange_ordered_path_kernel` within frozen single-particle alternatives | motivates the candidate architecture | its restart-level confidence intervals are not independent-parent evidence |
| segment splice | all finite T=0.45 lengths fail; only the excluded full path passes after ensemble averaging; two of three restart labels fail even at full path | short pieces and owner identity contain information; ensemble averaging can conceal failures | a finite memory length, owner sufficiency, static-versus-finite exchange, or an independent-parent lower bound |
| paired excess | a correlated-parent short-horizon excess prefix extends through L=10 and tau_L=200 | a descriptive correlated-parent canary | an independent-sample interval or positive memory closure |
| anchor semi-Markov | exact anchor geometry improves every T=0.45 restart label, but both anchor and schedule-matched models fail | binary return/escape geometry is insufficient | a complete environment or ordered-path model |
| gamma/variance mixture | gamma mobility improves low/intermediate-k shape but fails T=0.45 k=7.25 with normalized error 1.3854; shifted cage support is incomplete | scalar mobility carries partial shape information | blind prediction, finite exchange, activated geometry, or memory closure |

The historical `renewal_cage_ka_memory_hierarchy.csv` remains a record of the
old analysis. This change does not rewrite it. The new parent audit and claim
ledger supersede it only for the independent-parent PRL claim.

## Frozen parent provenance and statistical unit

The parent trajectory is the statistical unit. A restart from a different
frame or velocity seed of the same source parent is a correlated child, not an
independent sample. Randomized surrogate realizations and time origins are Monte
Carlo or within-trajectory units and never increase the parent count.

The committed provenance has one source SHA at T=0.45 and one source SHA at
T=0.58. Therefore the available parent counts are:

| temperature | restart labels | source parents | required independent parents | missing independent parents |
| ---: | ---: | ---: | ---: | ---: |
| 0.45 | 3 | 1 | 3 | 2 |
| 0.58 | 5 | 1 | 5 | 4 |

The required counts are inherited from the existing frozen protocols, which
required three low-temperature and five warm-control replicates. This design
corrects the unit from restart to parent without changing the count.

The provenance ledger must record, for every child trajectory: temperature,
restart label, parent identifier, source DOI and SHA256, parent frame, velocity
seed, production length, calibration and held-out windows, stationarity state,
available observables, and whether the row contributes an independent parent.
Rows sharing a parent identifier must be grouped before any interval or gate.

Within one parent, predictions and observations are first averaged across its
child restarts at each lag. Errors and pass/fail are then computed on that
parent-level curve. Restart rows remain visible so a failed child cannot be
hidden by the ensemble figure. Across parents, the primary result is the list of
parent pass/fail decisions; an ensemble mean is secondary.

## Frozen data and stationarity protocol

T=0.45 is the primary low-temperature evidence. It uses calibration time 5000,
an equal held-out window, block size 20 trajectory samples, and lags
`20,100,200,500,1000,2000,3000`.

T=0.58 is only a control/canary. It uses calibration time 750, an equal held-out
window, block size 20, and lags `20,100,200,400,600`. It is eligible only when
all frozen `early_late`, `early_heldout`, and `late_heldout` stationarity rows
pass. The committed T=0.58 source fails the first two comparisons, so it is
ineligible and must remain a canary in this run.

Every model uses Type-A calibration block displacements only. Held-out paths,
events, MSD, NGP, and multi-k Fs may not enter any model parameter, environment
lifetime, state, seed, or sampling choice. Held-out tables supply targets only.

The scored observables are MSD, three-dimensional NGP, and self-intermediate
scattering at `k={2,4,7.25}`. The frozen curve limits are:

```text
maximum relative MSD error = 0.10
maximum absolute NGP error = 0.30
maximum absolute Fs error over all k = 0.03
```

For stochastic predictions, the frozen Monte Carlo limits are:

```text
maximum relative MSD Monte Carlo SE = 0.01
maximum absolute NGP Monte Carlo SE = 0.03
maximum absolute Fs Monte Carlo SE = 0.003
```

The realization schedule is nested: evaluate deterministic realization indices
`0..15`; if any model/temperature cell fails precision, extend every evaluated
cell together to indices `0..63`. The base seed is `20260718`, inherited from the
segment-splice gate. The existing spectral ablation retains its frozen base seed
`211003` and iteration count.

## Positive model and nested ablations

All path models operate on the same calibration block array with shape
`particle x block x 3`. The particle-level environment state is the identity of
the calibration source particle supplying the current block. Its finite exchange
lifetime is not fitted to held-out curves: for each child trajectory it is the
block-size-20 calibration identity-correlation e-folding time already defined by
the environment-crossover analysis. The per-block exchange probability is fixed
as

```text
p_exchange = 1 - exp(-block_size / tau_environment).
```

At exchange, a different source particle is chosen uniformly. Its starting
calibration block is chosen uniformly. Without exchange, the environment state
persists. Every target particle starts in its own calibration identity. Source
choices and block choices use deterministic model/parent/restart/realization
seeds.

The six frozen ablations are:

1. `mean_rate_null`: independent isotropic Gaussian block vectors with the
   calibration global mean block-vector variance. It retains only the mean
   one-block transport scale.
2. `one_step_jump_law`: independent resampling from the global empirical
   calibration block-vector distribution. It retains the complete one-block law
   but no identity or order.
3. `two_point_path_spectrum`: the existing radial multivariate spectral
   surrogate. It retains the one-block radial law and two-point Cartesian path
   spectrum while destroying nonlinear ordered-path structure.
4. `static_particle_environment`: each target particle draws independent blocks
   from its own calibration path forever. It retains static identity and the
   particle-conditioned one-block law but no order or exchange.
5. `finite_exchange_environment`: the source-particle state follows the fixed
   finite-exchange process, while blocks are sampled independently from the
   current source path. It retains persistent finite-lifetime environment but no
   ordered recoil history.
6. `full_candidate`: the same finite-exchange process supplies contiguous
   calibration blocks while the state persists. At exchange it starts a new
   contiguous source run. It is the particle-conditioned finite-exchange
   ordered-path kernel.

Wrapping a source path is forbidden. Reaching its terminal calibration block
forces a recorded environment exchange. No target-particle path shares a global
source-segment schedule with another target path.

The contiguous empirical calibration path is retained as a nonselectable upper
control. It is not the positive model.

## Replicate-first and parent-first gate

For each model, restart, realization, lag, and observable, write the prediction,
target, error, and Monte Carlo contribution. Aggregate realizations within a
restart. Aggregate child restarts within a parent. Compute the curve gate only
after those two steps. The machine gate must never label restart count as
independent replicate count.

A parent passes one model only when every scored lag satisfies all three curve
limits and every stochastic precision limit. The full candidate must pass every
eligible T=0.45 parent. Ensemble averaging cannot rescue a parent failure.

The positive candidate claim may open only if all of the following are true:

- the ledger contains at least three independently prepared, stationarity-passing
  T=0.45 parents with complete calibration and held-out observables;
- the full candidate passes every T=0.45 parent;
- mean-rate, one-step, two-point-spectrum, static-environment, and
  finite-exchange-without-order ablations each fail at least one higher-order
  observable on at least two of the three required parents;
- the full candidate has a strictly smaller parent-level higher-order score than
  each required ablation on at least two of the three parents;
- the contiguous upper control passes every T=0.45 parent;
- if T=0.58 is used as a control rather than a canary, all stationarity rows pass,
  at least five independent parents are present, and the full candidate passes
  all five.

The higher-order score is unchanged from the prior gates:

```text
max(NGP_error / 0.30, maximum_Fs_error / 0.03).
```

No positive claim is allowed when parent provenance is incomplete, even if all
correlated restart curves pass.

## Fail-closed outcomes and failure localization

The gate emits exactly one primary state:

- `positive_memory_closure_supported_within_tested_family` only when every
  condition above passes;
- `blocked_independent_parent_validation` when the required parent count or
  independence provenance is missing;
- `blocked_stationarity_control` when a temperature is requested as evidence but
  its frozen stationarity comparisons fail;
- `candidate_rejected` when provenance and stationarity are eligible but the full
  candidate fails any parent curve or precision gate;
- `ablation_pattern_unresolved` when the full candidate passes but the required
  ablations do not fail in the preregistered pattern.

When the candidate is rejected, assign only the following diagnostic, never a
mechanism claim:

- `data_volume_or_support`: a calibration support or Monte Carlo precision gate
  fails;
- `environment_lifetime_or_state`: the static-environment/contiguous controls
  pass but finite-exchange/full models fail;
- `ordered_path_kernel`: finite exchange without order and the contiguous upper
  control outperform the failing full candidate;
- `cross_particle_or_unmodeled_coupling`: stationarity, support, and precision
  pass but even the contiguous single-particle upper control fails;
- `multiple_unresolved`: more than one pattern applies.

`cross_particle_or_unmodeled_coupling` does not prove spatial facilitation.

## Machine-readable outputs and reproducibility

The implementation must produce:

- `data/renewal_cage_ka_prl_parent_provenance.csv`;
- `data/renewal_cage_ka_prl_parent_blockers.csv`;
- restart/realization rows, restart summaries, parent summaries, model verdicts,
  and one final gate CSV under the prefix
  `data/renewal_cage_ka_prl_memory_closure`;
- `data/renewal_cage_ka_prl_memory_closure_claim_ledger.csv`;
- a deterministic secondary ensemble SVG that visually marks every parent
  failure rather than showing only an average;
- a CLI that can rebuild the ledger and gate from explicit trajectory manifests,
  calibration environment rows, held-out target rows, stationarity rows, and
  provenance;
- focused tests, package recomputation tests, and a result note.

If trajectories for enough independent parents are unavailable, the committed
blocker CSV must state the temperature, required parent count, available parent
count, missing count, required production length, required split, stationarity
requirement, observable list, and next action. Shared-parent resampling cannot
satisfy it.

## Engineering and evidence boundaries

Python 3.12 or newer is required by the current `origin/main` source surface.
The unmodified starting point passes 979 tests under Python 3.12.13. Python 3.9
fails on `zip(strict=True)`, and Python 3.11 cannot parse the current PEP 701
f-string; those are environment facts, not scientific results.

Focused tests establish implementation behavior. The full local suite and arXiv
package rebuild establish repository integration. Remote GitHub Actions is a
separate state and must be reported separately. None of those engineering checks
can convert a blocked scientific gate into a positive claim.
