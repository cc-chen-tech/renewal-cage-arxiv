# PRL T=0.45 Independent-Parent Acquisition Design

## Status and relationship to the frozen claim

This design is an acquisition addendum to
`2026-07-18-prl-memory-closure-independent-parent-design.md`. It does not
change the scientific question, claim wording, observable set, lag grid,
calibration budget, curve tolerances, Monte Carlo rule, ablation family, or
claim boundaries. Acquisition can remove a data blocker; it cannot turn a
failed stationarity, lineage, or model gate into a pass.

T=0.45 is primary. The two missing low-temperature parents are acquired before
any T=0.58 work. T=0.58 remains a canary and its four-parent deficit does not
delay these runs.

## Considered acquisition routes

1. Reuse two more frames or velocity seeds from the existing public parent.
   This is rejected because it creates correlated children of the same parent.
2. Prepare two complete KA systems through separately seeded high-temperature
   mixture histories, independent cooling ramps, and independent target-
   temperature holds. This is selected because every production parent has a
   separate preparation path while retaining the repository's KA potential and
   NVT production convention.
3. Locate two external published equilibrium trajectories. This would be
   acceptable with complete provenance, but no such local source is presently
   available and waiting for it would not execute the approved acquisition.

## Frozen system and preparation protocol

Each parent contains 1000 equal-mass particles at number density 1.2 in a
periodic cubic box. The mixture is exactly 800 type-A and 200 type-B particles.
The initial spatial template is a 10 by 10 by 10 simple-cubic lattice; each
parent uses a different exact random type assignment and a different Maxwell
velocity draw. The lattice is only a reproducible nonoverlapping initialization
template, not evidence of equilibrium.

The standard shifted Kob-Andersen Lennard-Jones potential is unchanged:

```text
pair_style lj/cut 2.5
pair_modify shift yes
pair_coeff 1 1 1.0 1.0 2.5
pair_coeff 1 2 1.5 0.8 2.0
pair_coeff 2 2 0.5 0.88 2.2
```

Every stage uses timestep 0.001 tau and `fix nvt` with thermostat damping 10
tau. Each parent follows its own uninterrupted trajectory through:

1. 1000 tau at T=1.0 to melt the initialization template;
2. 4000 tau of linear cooling from T=1.0 to T=0.45;
3. 5000 tau held at T=0.45 before the evidence trajectory begins;
4. 10000 tau production at T=0.45.

The preparation stages are excluded from calibration and heldout observables.
Production time 0 through 5000 tau is calibration; production time 5000 through
10000 tau is held out. Positions and image flags are written every 1 tau. The
expected production dump contains 10001 ordered frames, including endpoints.
Restarts are written every 100 tau, including a preparation-end restart and a
final restart.

The two frozen parent labels and random controls are:

| parent_id | type-assignment seed | velocity seed |
| --- | ---: | ---: |
| `ka-t045-independent-p02-20260719` | 4502001 | 4502002 |
| `ka-t045-independent-p03-20260719` | 4503001 | 4503002 |

No alternate seed may silently replace a failed or interrupted parent. A
replacement requires a committed manifest amendment and remains a new parent
attempt.

## Independence and eligibility

Distinct seeds and histories establish independently prepared attempts, not an
automatic scientific pass. Before either attempt contributes to the parent
count, the audit must verify:

- exact 800:200 composition, density 1.2, potential parameters, and full stage
  lengths;
- distinct type assignments, velocity draws, preparation trajectories, and
  production trajectory hashes;
- absolute pairwise initial-configuration Fs at k=7.25 no greater than the
  existing frozen 0.1 decorrelation limit, including comparison with every
  eligible existing parent where its production-start configuration exists;
- complete trajectory and manifest SHA256 provenance;
- all frozen restart-specific/parent-specific stationarity comparisons pass;
- heldout, environment, and spectral inputs pass exact parent-lineage joins.

If a parent fails any item, it remains a visible failed attempt. Ensemble
averaging, time-origin resampling, or a sibling parent cannot rescue it.

## Machine-readable acquisition contract

Before simulation starts, the repository must contain a JSON manifest that
freezes, for each parent: parent ID, seeds, initialization and equilibration
stages, temperature, production and split lengths, dump interval, particle and
box parameters, potential parameters, implementation commit, LAMMPS version and
binary SHA256, generated initial-data SHA256, generated input SHA256, remote
output directory, and expected completion checks.

The implementation commit is made before the manifest so that the manifest can
refer to an immutable code revision without a self-referential Git hash. The
LAMMPS build may occur before the simulation; its binary hash must be committed
to the manifest before either production job is launched.

## Parent-keyed input lineage

Every new heldout-target, environment-lifetime, and spectral artifact row must
carry these immutable keys:

```text
temperature
parent_id
trajectory_sha256
parent_manifest_sha256
```

Restart-bearing artifacts also carry `replicate`. The lineage audit requires
every row used for one parent to match its ledger exactly. A global table hash,
file path, or temperature-only join is insufficient. Legacy tables without the
keys remain diagnostic-only and keep their current lineage failure.

Heldout MSD, NGP, and Fs values are targets only. They may not enter model
parameters, exchange lifetimes, seeds, realization escalation, or any
calibration choice.

## Remote execution and completion

The authorized host has 2 vCPUs, 1.8 GiB RAM, 4 GiB swap, and 22 GiB free disk
at preregistration time. Each LAMMPS process is serial and launched with reduced
CPU priority so existing user work remains preferred. Both parent jobs may run
concurrently only after input validation; their combined projected output is
below 4 GiB.

A launched job is auditable only when its status record contains the remote
PID, exact command without credentials, output directory, stdout/stderr log,
start time, code commit, manifest SHA256, input hashes, and binary hash. Job
completion requires exit code zero, the expected 10001 production frames from
0 to 10000 tau, 1000 atoms in every frame, final restart presence, finite log
thermodynamics, and complete-file SHA256 hashes. Scientific eligibility is
decided later by the frozen stationarity, lineage, and 64-realization gates.

## Negative-result transition boundary

If the full candidate is rejected on at least three independently prepared,
stationarity-passing, lineage-complete T=0.45 parents, that result is frozen as
an independent negative result for this candidate. Neighbor-conditioned or
cross-particle coupling may then be considered only under a separate written
preregistration. No parameter or spatial term may be added to this model after
seeing the new heldout curves.
