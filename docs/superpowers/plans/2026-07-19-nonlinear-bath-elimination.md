# Nonlinear Bath Elimination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive, implement, and remotely validate a thermodynamically consistent nonlinear auxiliary-bath Langevin system whose exact elimination produces a single-particle position-dependent GLE and whose cage first-passage clock is tested against the frozen delayed-hazard effective theory.

**Architecture:** Pure NumPy modules separate exact algebra, diagnostics, and event statistics. A resumable remote-only simulator writes provenance-complete NPZ caches; a separate analyzer recomputes gates and writes CSV/SVG artifacts without changing the frozen parent parameters.

**Tech Stack:** Python 3.11, NumPy, `unittest`, existing CSV/SVG helpers, remote sequential execution through the approved compute node.

## Global Constraints

- Do not run production simulation locally.
- Do not start the nonlinear-bath remote canary or production run until the active L3p remote wrapper chain has exited.
- Use exactly the parameters, seeds, split, lags, thresholds, event definition, and nulls frozen in `docs/superpowers/specs/2026-07-19-nonlinear-bath-elimination-design.md`.
- Keep every real-KA, autonomous-GLE, complete-event-clock, Kramers, spatial-facilitation, and thermodynamic claim flag at zero.
- Fail closed on incomplete caches, changed source hashes, nonfinite arrays, mismatched checkpoints, or missing held trajectories.

---

### Task 1: Exact Extended Langevin Algebra

**Files:**
- Create: `src/nonlinear_bath_gle.py`
- Create: `tests/test_nonlinear_bath_gle.py`

**Interfaces:**
- Produces: `periodic_potential_gradient(u, barrier, period) -> ndarray`.
- Produces: `periodic_coupling(u, amplitudes, modulation, phases, period) -> ndarray` with shape `(..., modes)`.
- Produces: `nonlinear_bath_step(u, p, z, normal_p, normal_z, controls) -> tuple[ndarray, ndarray, ndarray]`.
- Produces: `reconstruct_auxiliary_path(...) -> ndarray` using the identical exact-OU discrete recurrence.
- Produces: `eliminated_memory_kernel(u_left, u_right, lag, controls) -> ndarray`.

- [ ] **Step 1: Write failing algebra tests**

Test periodic force against a centered finite difference of `W`, verify the
antisymmetric coupling conserves `p^2/2 + sum(z^2)/2` at zero thermostat and
zero potential to first order, and verify that reconstruction from supplied
normal increments reproduces every stored auxiliary value to `5e-11`.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m unittest tests.test_nonlinear_bath_gle -v
```

Expected: import failure for `nonlinear_bath_gle`.

- [ ] **Step 3: Implement the frozen equations**

Use

```python
rho = np.exp(-alpha * dt)
phi = -np.expm1(-alpha * dt) / alpha
next_u = u + p * dt
next_p = p + (-potential_gradient - gamma * p
              + np.sum(coupling * z, axis=-1)) * dt
next_p += np.sqrt(2.0 * gamma * temperature * dt) * normal_p
next_z = rho * z - phi * coupling * p[..., None]
next_z += np.sqrt(temperature * (1.0 - rho**2)) * normal_z
```

Validate every shape, scalar control, and claim flag before returning.

- [ ] **Step 4: Verify GREEN and dependencies**

Run:

```bash
python -m unittest tests.test_nonlinear_bath_gle tests.test_ka_generator_ladder_diffusion -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/nonlinear_bath_gle.py tests/test_nonlinear_bath_gle.py
git commit -m "derive exact nonlinear bath elimination"
```

### Task 2: Equilibrium, Bath-Replay, and Event Diagnostics

**Files:**
- Create: `src/nonlinear_bath_diagnostics.py`
- Create: `tests/test_nonlinear_bath_diagnostics.py`

**Interfaces:**
- Consumes: frozen controls and arrays from `nonlinear_bath_gle`.
- Produces: `equilibrium_diagnostics(u, p, z, controls) -> dict`.
- Produces: `bath_replay_covariance(paths, replay_forces, lags, controls) -> dict`.
- Produces: `accepted_periodic_cage_events(u, sample_time, period, persistence) -> list[dict]`.
- Produces: `fit_constant_hazard(training_waits, censored) -> dict`.
- Produces: `fit_delayed_square_hazard(training_waits, censored) -> dict`.
- Produces: `classify_nonlinear_bath_gate(...) -> dict`.

- [ ] **Step 1: Write failing diagnostic tests**

Use analytic Gaussian arrays for equilibrium moments, constructed OU covariance
tables for bath replay, and hand-built crossing/recrossing paths for event
acceptance. Use deterministic censored waiting-time tables to verify that the
delayed model is fitted on training rows only and that every broad claim stays
zero.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m unittest tests.test_nonlinear_bath_diagnostics -v
```

Expected: import failure for `nonlinear_bath_diagnostics`.

- [ ] **Step 3: Implement minimal diagnostics**

Compute the 40-bin Gibbs total-variation distance, fixed equilibrium errors,
lagwise normalized replay RMSE, nonrecrossing cage waits, exact constant-hazard
MLE, and bounded deterministic optimization of `(lambda,tau_d)` using a fixed
logarithmic `tau_d` grid and analytic conditional `lambda` MLE. Do not add SciPy.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m unittest tests.test_nonlinear_bath_diagnostics -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/nonlinear_bath_diagnostics.py tests/test_nonlinear_bath_diagnostics.py
git commit -m "add nonlinear bath physical gates"
```

### Task 3: Resumable Remote-Only Simulator

**Files:**
- Create: `scripts/simulate_nonlinear_bath_elimination.py`
- Modify: `tests/test_nonlinear_bath_gle.py`

**Interfaces:**
- Consumes: frozen equations from Task 1.
- Produces: canary and production NPZ caches with `completed_step_count`, RNG
  state, source SHA256, all frozen controls, downsampled paths, event records,
  and zero-valued broad claim flags.

- [ ] **Step 1: Write failing CLI/provenance tests**

Require `--output-path`, `--mode {canary,production,null-constant-coupling,null-no-bath}`,
`--checkpoint-interval`, and `--resume`. Test argument validation and rejection
of a checkpoint whose source hash, seed, mode, or parameter vector differs.
Do not execute trajectory generation in local tests.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m unittest tests.test_nonlinear_bath_gle -v
```

Expected: missing CLI or provenance helper.

- [ ] **Step 3: Implement atomic checkpointing and vectorized stepping**

Generate all trajectories in one vectorized state array while iterating only
over time. Retain every canary increment, but production only retains the
frozen downsampled state and online event information. Save through a temporary
NPZ followed by atomic rename.

- [ ] **Step 4: Verify CLI without local simulation**

Run:

```bash
python scripts/simulate_nonlinear_bath_elimination.py --help
python -m unittest tests.test_nonlinear_bath_gle -v
```

Expected: help succeeds and all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/simulate_nonlinear_bath_elimination.py tests/test_nonlinear_bath_gle.py
git commit -m "add resumable nonlinear bath simulator"
```

### Task 4: Frozen Analyzer and Artifacts

**Files:**
- Create: `scripts/analyze_nonlinear_bath_elimination.py`
- Create: `tests/test_nonlinear_bath_analysis.py`

**Interfaces:**
- Consumes: complete caches from Task 3 only.
- Produces: `_details.csv`, `_summary.csv`, `_survival.csv`, and an unclipped
  SVG with equilibrium, FDT-replay, and held survival panels.

- [ ] **Step 1: Write failing loader/classifier/SVG tests**

Require source-hash recomputation, exact frozen controls, 128/128 split,
complete null grid, recomputation of every verdict field, and SVG labels that
state all axes and tolerances without clipping.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m unittest tests.test_nonlinear_bath_analysis -v
```

Expected: missing analyzer.

- [ ] **Step 3: Implement fail-closed analysis**

Read no held event score until the cache/numerical/equilibrium/replay gates are
recomputed. Fit hazards on trajectories `0:128`, evaluate only `128:256`, and
bootstrap complete trajectories with seed `20260812`. Write all outcome and
claim flags mechanically.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m unittest tests.test_nonlinear_bath_analysis -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_nonlinear_bath_elimination.py tests/test_nonlinear_bath_analysis.py
git commit -m "gate nonlinear bath elimination"
```

### Task 5: Sequential Remote Canary and Production

**Files:**
- Generate remotely: `nonlinear_bath_canary.npz`, production/null NPZ caches,
  logs, CSV/SVG outputs, environment manifest, and SHA256 manifest.

**Interfaces:**
- Consumes: committed Tasks 1-4 and an exited L3p wrapper chain.
- Produces: authoritative numerical and physical verdicts.

- [ ] **Step 1: Confirm zero active numerical processes after L3p completion**

Run a remote process audit and require no Python cache/analyzer process before
launching the canary.

- [ ] **Step 2: Run canary remotely**

Run the frozen canary and its half-step counterpart sequentially. Stop before
production if algebraic reconstruction, finite-state, or checkpoint provenance
fails.

- [ ] **Step 3: Run primary and nulls sequentially**

Run primary, constant-coupling, and no-bath caches one at a time with identical
base Brownian streams. Run the time-permuted kernel diagnostic only in the
analyzer.

- [ ] **Step 4: Analyze and copy only authoritative artifacts**

Generate verdicts remotely, record environment/source/input/output hashes, and
copy CSV/SVG/log/manifest files into the worktree. Do not copy full raw paths
into git.

- [ ] **Step 5: Mechanically inspect every flag and render the SVG**

Use the CSV loader and an SVG rasterization/view check. Any missing row,
clipped axis, or nonzero broad claim blocks packaging.

### Task 6: Documentation, Package, and PR

**Files:**
- Create: `docs/microscopic-nonlinear-bath-elimination.md`
- Modify: `README.md`
- Modify: `scripts/build_arxiv_package.py`
- Modify: `tests/test_arxiv_package.py`
- Generate: tracked CSV/SVG and packaged PDF if the repository convention
  requires it.

- [ ] **Step 1: Write package tests before documentation edits**

Require the exact equations, frozen outcomes, negative results if any, remote
provenance, literature boundary, and all zero claim flags in the source bundle.

- [ ] **Step 2: Verify package RED**

Run the focused package test and confirm it fails because the new note/artifacts
are absent.

- [ ] **Step 3: Add outcome documentation and package entries**

Report exact measured values. Do not call a synthetic pass evidence for KA,
finite generator closure, Kramers escape, spatial facilitation, or
thermodynamics.

- [ ] **Step 4: Run complete verification**

Run the focused suites, complete `unittest` discovery, package rebuild, archive
listing, `git diff --check`, and CI-equivalent command. Record exact outputs.

- [ ] **Step 5: Commit, push, and open a stacked draft PR**

Push `codex/generator-ladder-diffusion-closure` and create a draft PR stacked
on `codex/l3p-generator-quotient`. Keep it draft until remote artifacts and all
CI checks pass.
