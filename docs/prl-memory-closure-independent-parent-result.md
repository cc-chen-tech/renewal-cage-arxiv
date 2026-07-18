# PRL Memory Closure: Independent-Parent Result

## Frozen question and claim

The preregistered question is:

> What microscopic information is minimally required to reconstruct
> low-temperature glassy relaxation?

The candidate claim was frozen before the positive-model run:

> In the low-temperature Kob-Andersen glass former, relaxation dynamics cannot
> be reconstructed from the mean event rate, one-step jump law, or two-point
> path spectrum. Accurate reconstruction requires ordered cage-path memory and
> a persistent particle-level environment.

The complete frozen design, including the thresholds and claim boundaries, is
[`superpowers/specs/2026-07-18-prl-memory-closure-independent-parent-design.md`](superpowers/specs/2026-07-18-prl-memory-closure-independent-parent-design.md).
No threshold, lag, observable, calibration budget, realization rule, or claim
wording was changed after inspecting the positive-model curves.

## Parent-provenance decision

The parent trajectory, not a restart label, is the statistical unit. The three
`T=0.45` restart labels all descend from the same DOI/SHA256 source parent; the
five `T=0.58` labels likewise descend from one source parent. Different source
frames and velocity seeds do not turn those children into independently
prepared parents.

| Temperature | Role | Required parents | Available parents | Missing parents | Stationarity |
| --- | --- | ---: | ---: | ---: | --- |
| `0.45` | primary low-temperature evidence | 3 | 1 | 2 | pass |
| `0.58` | control/canary only | 5 | 1 | 4 | fail (`early_late` and `early_heldout`) |

The machine gate is therefore
`blocked_independent_parent_validation`. Shared-parent resampling cannot clear
it. The exact acquisition blockers are recorded in
`data/renewal_cage_ka_prl_parent_blockers.csv`.

## Evidence frozen before the positive model

The audit preserves the distinctions among earlier tests:

- the `T=0.45` one-step recoil, within-particle shuffle, and randomized-direction
  nulls fail, whereas the corresponding warmer one-step test passes;
- the radial multivariate spectral surrogate retains the one-block radial law
  and Cartesian two-point spectrum but fails higher-order `T=0.45` transfer;
- segment splice rejects every finite `T=0.45` segment length, while two of the
  three child restart labels still fail the excluded full-path control;
- the paired-excess prefix is a shared-parent exploratory canary, not an
  independent-sample confidence statement;
- anchor semi-Markov and gamma/variance-mixture diagnostics improve selected
  observables but do not close the frozen low-temperature curve gate.

Those results support a need for information beyond rate, one-step statistics,
and a two-point spectrum. They do not independently establish the positive
memory-closure claim.

## Positive-model diagnostic

The implemented candidate is a particle-conditioned finite-exchange
ordered-path kernel. Its only added state is calibration-source particle
identity with the pre-existing block-size-20 environment e-folding lifetime;
while that identity persists it emits contiguous calibration blocks. Source
endpoints force a recorded exchange rather than wrapping. Heldout MSD, NGP, and
multi-`k` scattering are joined only after each calibration-only prediction is
complete.

The nested family is:

1. `mean_rate_null`;
2. `one_step_jump_law`;
3. `two_point_path_spectrum` (the frozen existing spectral surrogate);
4. `static_particle_environment`;
5. `finite_exchange_environment` without order;
6. `full_candidate` with finite exchange and ordered blocks.

The contiguous empirical calibration path is a nonselectable upper control.
Pass/fail is computed for every child restart before parent aggregation, so a
parent or ensemble average cannot conceal a child failure.

The final correlated-parent diagnostic uses the frozen nested realization grid
`0..63`; the exact model scores and precision flags are in
`data/renewal_cage_ka_prl_memory_closure_model_verdicts.csv`:

| Model | Failed child restarts | Worst child higher-order score | Precision pass |
| --- | ---: | ---: | ---: |
| mean-rate null | 3/3 | 25.679 | no |
| one-step jump law | 3/3 | 25.640 | no |
| two-point path spectrum | 3/3 | 5.289 | yes |
| static particle environment | 3/3 | 25.309 | no |
| finite-exchange environment, no order | 3/3 | 25.589 | no |
| full finite-exchange ordered-path candidate | 3/3 | 14.923 | yes |
| contiguous empirical upper control | 2/3 | 3.324 | yes |

The score is `max(NGP error / 0.30, maximum Fs error / 0.03)`. The full
candidate's maximum child errors are MSD `1.698`, NGP `1.785`, and Fs `0.448`,
whereas its maximum Monte Carlo errors are relative MSD `0.00387`, NGP
`0.00366`, and Fs `0.000614`. Its rejection is therefore a curve failure, not a
Monte Carlo precision artifact.

The positive candidate fails the frozen low-temperature curve gate, as do all
selectable ablations. The contiguous upper control also fails replicate-first
transfer, because child restarts 2 and 3 fail even though their parent-averaged
curve is closer. This localizes the observed closure gap only to
`cross_particle_or_unmodeled_coupling` within the preregistered diagnostic
vocabulary. That label does **not** establish spatial facilitation.

## Claim decision and boundary

The candidate positive-memory claim remains closed for two independent reasons:

1. only one of three required `T=0.45` parents exists; and
2. the full candidate fails the replicate-first correlated-parent diagnostic.

The result does not establish complete microscopic closure, spatial
facilitation, or a thermodynamic glass transition. Every corresponding claim
flag remains zero in
`data/renewal_cage_ka_prl_memory_closure_claim_ledger.csv`.

## Reproduction

The audit-only mode writes the parent ledger and fail-closed gate without
touching trajectory files. Full mode reads an explicit ensemble manifest,
calibration prefixes, heldout target table, environment-crossing table, and the
recomputable spectral-null artifact. The command and all output paths are
documented in
[`superpowers/plans/2026-07-18-prl-memory-closure-independent-parent.md`](superpowers/plans/2026-07-18-prl-memory-closure-independent-parent.md).

Validation is reported in three separate layers:

- scientific result: parent gate blocked and full candidate rejected as above;
- engineering validation: focused tests, full local suite, deterministic
  artifact recomputation, syntax checks, and arXiv package rebuild;
- remote CI: reported from the draft PR and never used as evidence for a
  scientific pass.
