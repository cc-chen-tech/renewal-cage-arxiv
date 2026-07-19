# PRL acquisition completion and ingestion design

## Scope

This is an execution addendum to the frozen independent-parent acquisition and
memory-closure specifications.  It changes no claim wording, thresholds, lag
grid, calibration budget, held-out observable, realization count, or ablation
expectation.  All claim fields remain zero until the existing parent-first gate
opens them from scientifically eligible data.

## Passive completion watcher

The watcher may read PIDs, LAMMPS logs, trajectories, manifests, and restart
files.  It may write only a separate completion artifact.  It must not signal a
process, edit a run directory, restart a job, or infer an exit code from output
completeness.

For every parent it records the observation time, PID state, explicit exit code
when an exit-code file exists, frame count, first and last production timestep,
atom-count consistency, final-restart presence, error-signature count, file
sizes, and complete-file SHA256 hashes.  Scientific ingestion is allowed only
when an explicit exit code is zero and every frozen completion check passes.
An absent historical exit code is
`blocked_missing_observed_exit_code`, even when all output checks pass.

The watcher has no password option and reads no credential file.  Remote
deployment uses an already authenticated SSH session.  If that session is no
longer available when trajectory transfer is required, the repo artifact must
report `requires_ssh_key_or_interactive_reauthentication`; it must not store a
password.

## One-click independent-parent ingestion

The CLI consumes the frozen acquisition manifest, one completion artifact, and
one or more parent output directories.  It first verifies parent ID and hashes,
then refuses scientific processing for every completion-ineligible parent.

For an eligible T=0.45 trajectory, frames 0 through 5000 tau form calibration
and frames 5000 through 10000 tau form heldout.  The shared boundary frame is
the coordinate origin of the held-out displacement window; no held-out target
value is used as model input.  Type-A displacement paths are blocked at the
already frozen 20-tau interval.  Held-out MSD, three-dimensional NGP, and
axis-averaged Fs at k=2, 4, and 7.25 are evaluated on the frozen lag grid.
Stationarity is decided separately for `early_late`, `early_heldout`, and
`late_heldout`, per parent, with the existing curve tolerances.

The same calibration paths drive the frozen 64-realization family:

- `mean_rate_null`;
- `one_step_jump_law`;
- `two_point_path_spectrum` (eight frozen spectral surrogates, as already
  preregistered inside the six-family gate);
- `static_particle_environment`;
- `finite_exchange_environment`;
- `full_candidate`.

The contiguous empirical path is retained only as the pre-existing upper
control.  Parent is the statistical unit; realization and time-origin averages
never increase the independent-parent count.  The output bundle contains a
parent ledger, completion blockers, held-out targets, stationarity rows,
calibration-only environment rows, spectral rows, realization rows, model
verdicts, gate row, and claim ledger.  Tests may use explicitly labelled
synthetic fixtures, but fixture outputs always keep scientific claims closed.

## Failure states

Completion, lineage, stationarity, grid completeness, and model failures remain
separate machine-readable states.  Missing exit status or missing remote
authentication blocks import.  Failed stationarity blocks the model claim.
Failure of the full candidate on three eligible T=0.45 parents is preserved as
an independent negative result; it does not authorize a post-hoc spatial term.
