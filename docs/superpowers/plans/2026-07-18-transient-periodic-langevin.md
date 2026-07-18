# Transient Periodic Langevin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a continuous equilibrium transient-periodic Langevin canary that separates barrier-rate disorder from elastic recoil memory, while mechanically proving that count overdispersion alone does not close the committed KA cage-scale scattering residual.

**Architecture:** Add one focused numerical module for the extended-coordinate potential, integration, cage-event extraction, and observables. Keep the real-data quotient in a separate summarizer that consumes only committed CSVs. A deterministic analysis script runs four frozen synthetic ablations and emits claim-limited CSV/SVG artifacts.

**Tech Stack:** Python 3.12, NumPy, standard-library `csv`/`dataclasses`/`math`, `unittest`, repository-native deterministic SVG generation.

## Global Constraints

- Work only in `/Users/luicy/AI/renewal-cage-arxiv/.worktrees/transient-periodic-langevin` on `codex/transient-periodic-langevin`.
- Keep the existing event inputs and `F_s` absolute tolerance `0.03` unchanged.
- Heldout MSD and NGP in the quotient are diagnostic inputs, not blind predictions.
- Use one scalar barrier coordinate `z` shared by all dimensions.
- Every stochastic output uses an explicit integer seed.
- All eight strong claim flags in the design remain `0`.
- T=0.58 remains a stationarity-failing canary.
- Add no external dependencies.

---

### Task 1: Count-Overdispersed Geometry Quotient

**Files:**
- Create: `scripts/summarize_ka_count_overdispersed_geometry.py`
- Create: `tests/test_ka_count_overdispersed_geometry.py`

**Interfaces:**
- Consumes activated-geometry and state-joint macro rows keyed by `(temperature, replicate, lag)`.
- Produces `count_overdispersed_geometry_row(geometry, macro)` and `summarize_temperature(rows, source_stationarity_pass)`.

- [ ] **Step 1: Write failing algebra and boundary tests.** Check the gamma-Poisson PGF, the `F -> 1` Poisson branch, unsupported negative cage variance without clipping, and zero claim flags.
- [ ] **Step 2: Confirm RED.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_ka_count_overdispersed_geometry -v
```

Expected: import failure.

- [ ] **Step 3: Implement the exact transform.** Use

```python
denominator = m4x + 3.0 * (fano - 1.0) * (m2 / 3.0) ** 2
mean_count = alpha2 * msd**2 / (3.0 * denominator)
cage_variance = (msd - mean_count * m2) / 6.0
count_pgf = (1.0 + (fano - 1.0) * (1.0 - phi)) ** (
    -mean_count / (fano - 1.0)
)
```

Set `curve_transfer_pass=1` only if support is at least `0.8` and every `F_s` error is at most `0.03`.

- [ ] **Step 4: Lock committed-table values.** Assert T045 support `20/21`, maximum high-k error `0.04108845937739283`; T058 support `25/25`, maximum high-k error `0.04664285739962798`; both curve gates fail and T058 is canary-only.
- [ ] **Step 5: Run tests and commit.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_ka_count_overdispersed_geometry -v
git add scripts/summarize_ka_count_overdispersed_geometry.py tests/test_ka_count_overdispersed_geometry.py
git commit -m "test count-overdispersed cage geometry"
```

### Task 2: Potential, Forces, and Parameters

**Files:**
- Create: `src/transient_periodic_langevin.py`
- Create: `tests/test_transient_periodic_langevin.py`

**Interfaces:**
- Produces immutable `TransientPeriodicParams`.
- Produces `potential_energy(x, q, z, params)` and `conservative_forces(x, q, z, params)` for `x,q` shape `(trajectory, dimension)` and `z` shape `(trajectory,)`.

- [ ] **Step 1: Write failing validation, central-difference gradient, and joint translation-invariance tests.** Require force errors below `2e-6`; reject nonpositive thermal/friction/barrier/period values and negative couplings.
- [ ] **Step 2: Confirm RED.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_transient_periodic_langevin.TransientPotentialTests -v
```

- [ ] **Step 3: Implement the potential and forces.**

```python
phase = 2.0 * np.pi * x / period
barrier = base_barrier + barrier_coupling * z**2
U = np.sum(0.5 * barrier[:, None] * (1.0 - np.cos(phase))
           + 0.5 * K * (x - q)**2, axis=1) + 0.5 * k_z * z**2
force_x = -barrier[:, None] * np.pi / period * np.sin(phase) - K * (x-q)
force_q = K * (x-q)
force_z = -k_z*z - g*z*np.sum(1.0-np.cos(phase), axis=1)
```

- [ ] **Step 4: Run tests and commit.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_transient_periodic_langevin.TransientPotentialTests -v
git add src/transient_periodic_langevin.py tests/test_transient_periodic_langevin.py
git commit -m "add transient periodic potential"
```

### Task 3: Continuous Integrator

**Files:**
- Modify: `src/transient_periodic_langevin.py`
- Modify: `tests/test_transient_periodic_langevin.py`

**Interfaces:**
- Produces `simulate_transient_periodic_langevin(params, trajectory_count, dimension, dt, burnin_steps, production_steps, record_stride, seed)` returning recorded `positions`, `environment_positions`, `barrier_coordinates`, `record_dt`, maximum Euler displacement, and a finite-state audit.

- [ ] **Step 1: Write failing seed, local equipartition, free-translation, and stability-bound tests.** In the high-barrier `K=g=0` limit, wrapped variance must agree within `15%` with `T/(2*pi^2*V0/L^2)`. Reject `dt*maximum_local_curvature/gamma_x >= 0.2`.
- [ ] **Step 2: Confirm RED.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_transient_periodic_langevin.TransientIntegratorTests -v
```

- [ ] **Step 3: Implement vectorized Euler-Maruyama.** Use FDT amplitude `sqrt(2*T*dt/gamma)`, hold irrelevant `q` fixed for `K=0`, integrate `z` as an uncoupled OU control for `g=0`, burn in before recording, and raise on nonfinite state.
- [ ] **Step 4: Run tests and commit.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_transient_periodic_langevin.TransientIntegratorTests -v
git add src/transient_periodic_langevin.py tests/test_transient_periodic_langevin.py
git commit -m "simulate transient periodic Langevin paths"
```

### Task 4: Event Coarse-Graining and Observables

**Files:**
- Modify: `src/transient_periodic_langevin.py`
- Modify: `tests/test_transient_periodic_langevin.py`

**Interfaces:**
- Produces `stable_cage_events(positions, period, dwell_frames)`, `event_clock_statistics(events, trajectory_count, dimension, duration, count_window)`, and `displacement_observables(positions, lag_frames, wave_numbers)`.

- [ ] **Step 1: Write failing tests.** A cage path `0,1,0,1,1,1` with dwell three accepts only the final transition. Deterministic displacements reproduce direct MSD, NGP, and component cosine values. No-event paths return finite count fields plus explicit unsupported persistence/exchange flags.
- [ ] **Step 2: Confirm RED.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_transient_periodic_langevin.TransientEventTests -v
```

- [ ] **Step 3: Implement nearest-well indices `floor(x/L+0.5)`, non-recrossing acceptance, per-trajectory-window count Fano, persistence/exchange means, and normalized successive-vector dot correlation.**
- [ ] **Step 4: Run tests and commit.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_transient_periodic_langevin.TransientEventTests -v
git add src/transient_periodic_langevin.py tests/test_transient_periodic_langevin.py
git commit -m "coarse grain continuous cage events"
```

### Task 5: Frozen Four-Model Ablation

**Files:**
- Create: `scripts/analyze_transient_periodic_langevin.py`
- Modify: `tests/test_transient_periodic_langevin.py`

**Interfaces:**
- Produces `run_ablation(seed, quick)` for `static_periodic`, `rate_only`, `elastic_only`, and `full_transient` with paired seeds.
- Gate fields: `rate_disorder_count_fano_increase`, `elastic_memory_more_negative_step_correlation`, `full_model_joint_signature_pass`, and `synthetic_capability_only`.

- [ ] **Step 1: Write failing CLI, reproducibility, finite-support, directional-gate, and zero-claim tests.**
- [ ] **Step 2: Confirm RED.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_transient_periodic_langevin.TransientAblationTests -v
```

- [ ] **Step 3: Implement paired ablations.** Start with `T=1,L=1,V0=3,k_z=1,K in {0,1},g in {0,1.5},gamma_x=1,gamma_q=20,gamma_z=20,dt=0.002,record_stride=10,dwell_frames=5`. Production uses at least 384 trajectories, 10,000 burn-in steps, and 40,000 production steps. Change parameters only for numerical stability/event support, record the change, and never tune against heldout KA curves.
- [ ] **Step 4: Run tests and commit.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_transient_periodic_langevin -v
git add scripts/analyze_transient_periodic_langevin.py tests/test_transient_periodic_langevin.py
git commit -m "test transient Langevin ablations"
```

### Task 6: Artifacts, Note, and Package Gate

**Files:**
- Create: `data/renewal_cage_ka_count_overdispersed_geometry_rows.csv`
- Create: `data/renewal_cage_ka_count_overdispersed_geometry_gate.csv`
- Create: `data/renewal_cage_transient_periodic_langevin_ablation.csv`
- Create: `data/renewal_cage_transient_periodic_langevin_gate.csv`
- Create: `figures/renewal_cage_transient_periodic_langevin.svg`
- Create: `docs/microscopic-transient-periodic-langevin.md`
- Modify: `tests/test_arxiv_package.py`
- Modify: `README.md`

**Interfaces:**
- Both CLIs write deterministic CSV outputs; the ablation CLI writes the compact SVG.

- [ ] **Step 1: Write a failing package test.** Recompute byte-identical outputs in a temporary directory; require all SVG coordinates inside the viewBox; require the note to state equations, `20/21`, both high-k failures, four ablations, Uneyama relation, and all zero flags.
- [ ] **Step 2: Confirm RED.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_arxiv_package.ArxivPackageTests.test_transient_periodic_langevin_is_recomputed_and_claim_limited -v
```

- [ ] **Step 3: Run production CLIs with seed `20260718`, write the note, and update README.** Separate synthetic Langevin capability, real KA diagnostic quotient, and unresolved calibration-to-heldout identification.
- [ ] **Step 4: Inspect SVG visually and mechanically.** Record the exact method; do not claim browser verification after coordinate-only audit.
- [ ] **Step 5: Run focused/package tests and commit.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_transient_periodic_langevin tests.test_ka_count_overdispersed_geometry tests.test_arxiv_package.ArxivPackageTests.test_transient_periodic_langevin_is_recomputed_and_claim_limited -v
git add README.md docs/microscopic-transient-periodic-langevin.md data/renewal_cage_ka_count_overdispersed_geometry_*.csv data/renewal_cage_transient_periodic_langevin_*.csv figures/renewal_cage_transient_periodic_langevin.svg scripts tests/test_arxiv_package.py
git commit -m "report transient periodic Langevin canary"
```

### Task 7: Full Verification and Publication

**Files:**
- Modify only deterministic packaging outputs required by verification.

- [ ] **Step 1: Recompute standard outputs.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/generate_renewal_cage_results.py
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/build_arxiv_package.py
```

- [ ] **Step 2: Run full tests and checks.**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s tests -v
git diff --check
git status --short
```

- [ ] **Step 3: Require recomputation to leave existing tracked outputs unchanged.** If `git status --short` shows a pre-existing artifact changed by either standard generator, inspect it as a verification failure and fix the new integration before publication; do not bundle unrelated regenerated output.
- [ ] **Step 4: Push and create a ready stacked PR.**

```bash
git push -u origin codex/transient-periodic-langevin
gh pr create --base codex/activated-cage-geometry --head codex/transient-periodic-langevin --title "Derive a transient periodic Langevin canary" --body "Derive and simulate an equilibrium transient-periodic Langevin canary with separate barrier-disorder and elastic-memory ablations. Mechanically recompute the count-overdispersed KA geometry quotient, keep all strong claims fail-closed, and document the remaining calibration-to-heldout gate."
```

If PR #18 has merged, rebase onto latest `origin/main`, rerun all verification, and target `main`. Do not request escalated permission; report a direct `gh` error if rejected.

- [ ] **Step 5: Watch remote CI and report local tests, CI, PR state, and merge state separately.**

```bash
gh pr checks "$(gh pr view --json number --jq .number)" --watch
```
