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

| Temperature | Role | Required parents | Available parents | Missing parents | Restart-first stationarity | Auxiliary-input lineage |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `0.45` | primary low-temperature evidence | 3 | 1 | 2 | fail; 0/3 child restarts pass all comparisons | fail; derived tables lack embedded parent keys |
| `0.58` | control/canary only | 5 | 1 | 4 | fail; only child restart 4 passes all comparisons | fail; derived tables lack embedded parent keys |

The machine gate is therefore
`blocked_independent_parent_validation`. Shared-parent resampling cannot clear
it. The old temperature-averaged stationarity rows are retained only as
historical evidence; they cannot qualify a parent. The exact acquisition,
stationarity, and lineage blockers are recorded in
`data/renewal_cage_ka_prl_parent_blockers.csv`, with the restart-level audit in
`data/renewal_cage_ka_prl_parent_stationarity.csv` and cryptographic input audit
in `data/renewal_cage_ka_prl_input_lineage.csv`.

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
complete. For every restart and realization, the finite-exchange/no-order and
full ordered models now consume the exact same precomputed source-particle and
exchange schedule; their only difference is within-source block order.

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

The automatic nested procedure first evaluated `0..15`, detected a precision
failure, and extended every generated stochastic model to `0..63`. The frozen
spectral surrogate retains its eight realizations `0..7`, and the contiguous
upper control is deterministic. The exact model scores and precision flags are in
`data/renewal_cage_ka_prl_memory_closure_model_verdicts.csv`:

| Model | Failed child restarts | Worst child higher-order score | Precision pass |
| --- | ---: | ---: | ---: |
| mean-rate null | 3/3 | 25.679 | no |
| one-step jump law | 3/3 | 25.640 | no |
| two-point path spectrum | 3/3 | 5.289 | yes |
| static particle environment | 3/3 | 25.309 | no |
| finite-exchange environment, no order | 3/3 | 25.606 | no |
| full finite-exchange ordered-path candidate | 3/3 | 14.895 | yes |
| contiguous empirical upper control | 2/3 | 3.324 | yes |

The score is `max(NGP error / 0.30, maximum Fs error / 0.03)`. The full
candidate's maximum child errors are MSD `1.698`, NGP `1.781`, and Fs `0.447`,
whereas its maximum Monte Carlo errors are relative MSD `0.00415`, NGP
`0.00320`, and Fs `0.000617`. Its rejection is therefore a curve failure, not a
Monte Carlo precision artifact.

The positive candidate fails the frozen low-temperature curve gate, as do all
selectable ablations. The contiguous upper control also fails replicate-first
transfer, because child restarts 2 and 3 fail even though their parent-averaged
curve is closer. This localizes the observed closure gap only to
`cross_particle_or_unmodeled_coupling` within the preregistered diagnostic
vocabulary. That label does **not** establish spatial facilitation.

## Claim decision and boundary

The candidate positive-memory claim remains closed for three independent reasons:

1. only one of three required `T=0.45` parents is currently scientifically
   eligible; two new independent acquisition attempts have complete outputs but
   fail closed because their original launcher did not persist an exit code;
2. the available `T=0.45` parent fails restart-first stationarity;
3. the full candidate fails the replicate-first correlated-parent diagnostic.

The earlier auxiliary-input blocker is now resolved as an engineering result:
every heldout, environment, and spectral row embeds its exact parent ID, source
hash, complete-trajectory hash, byte size, and hash scope, and all eight
restart-specific joins pass. This does not turn correlated restarts into
independent parents and therefore does not open the scientific claim.

The result does not establish complete microscopic closure, spatial
facilitation, or a thermodynamic glass transition. Every corresponding claim
flag remains zero in
`data/renewal_cage_ka_prl_memory_closure_claim_ledger.csv`.

## Reproduction

The passive completion watcher is
`scripts/watch_ka_parent_completion.py`. It never signals a process or edits a
run directory and has no credential option. Its remote snapshot found both new
PIDs already exited. Each parent has 10001 frames, timesteps 0 through
10000000, exactly 1000 atoms per frame, a final restart, zero matched LAMMPS
error signatures, and complete-file SHA256 hashes. Because the historical
launcher did not save `wait` status, both explicit exit codes are unavailable;
`data/renewal_cage_ka_prl_T045_parent_acquisition_completion.json` therefore
records `blocked_missing_observed_exit_code` rather than inferring success from
the files.

`scripts/import_ka_independent_parent_acquisition.py` is the one-click importer.
It validates completion and parent hashes before opening any trajectory. For an
eligible bundle it loads each parent separately, takes 0--5000 tau as
calibration and 5000--10000 tau as heldout, writes parent-keyed MSD, NGP and
multi-k Fs targets, runs all three stationarity comparisons per parent,
estimates the environment lifetime from calibration only, builds the frozen
two-point spectral surrogate, and executes the 64-realization six-ablation
gate. On the current completion artifact it stops before trajectory I/O and
writes the machine-readable fail-closed bundle under
`data/renewal_cage_ka_prl_T045_parent_acquisition_import/`. No trajectory was
downloaded merely to bypass the missing-exit blocker; if the transient
authenticated SSH session expires before a later authorized transfer, an SSH
key or new interactive authentication is required and no password is stored.

`scripts/bind_ka_prl_input_lineage.py` verifies ensemble and child manifests,
hashes each complete trajectory, and deterministically binds that identity into
all heldout, environment, and spectral rows. Then
`scripts/audit_ka_prl_parent_inputs.py` rebuilds restart-specific stationarity
and lineage from both raw ensembles, their manifests, and explicit auxiliary
tables. Audit-only mode then writes the parent ledger and fail-closed gate
without touching trajectory files. Full mode reads an explicit ensemble
manifest, calibration prefixes, heldout target table, environment-crossing
table, and spectral-null artifact; its 16-to-64 escalation is automatic.
Derived-only mode re-audits provenance/stationarity/lineage and rebuilds every
summary, verdict, gate, claim row, and SVG from the frozen realization rows. The
full CLI re-hashes the actual runtime manifests, auxiliary files, and complete
trajectory files against the lineage audit before reading trajectory blocks. A
stored 64-realization grid is accepted only if its own `0..15` prefix reproduces
the frozen precision trigger.
The commands and all output paths are documented in
[`superpowers/plans/2026-07-18-prl-memory-closure-independent-parent.md`](superpowers/plans/2026-07-18-prl-memory-closure-independent-parent.md).

Validation is reported in three separate layers:

- scientific result: parent gate blocked and full candidate rejected as above;
- engineering validation: 62 focused acquisition, completion, ingestion,
  lineage, and memory-closure tests pass; the complete local Python 3.12 suite
  reports `Ran 1090 tests in 25.691s — OK`; the new fail-closed acquisition
  import bundle rebuilds byte-identically; six lineage-bound input artifacts, the two
  raw-audit artifacts, and eight downstream artifacts rebuild
  byte-identically; runtime input hashes, Python syntax, and `git diff --check`
  pass; and `scripts/build_arxiv_package.py` exits zero with
  `dist/renewal-cage-arxiv-source.zip`;
- remote CI: pending when this result note was committed; its later state is
  reported on the draft PR and is never used as evidence for a scientific
  pass.
