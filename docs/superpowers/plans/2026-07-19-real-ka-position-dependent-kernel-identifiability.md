# Real-KA Position-Dependent Kernel Identifiability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether real microscopic KA cage dynamics identifies a finite-basis position-dependent Mori-Zwanzig kernel and whether that kernel admits the stricter thermodynamic two-position positive-Prony realization.

**Architecture:** Add a focused numerical core for training-only radial bases, regularized causal Volterra inversion, and held residual diagnostics. A separate CLI performs nested whole-clone selection over the frozen model hierarchy and writes fail-closed CSV/SVG verdicts. Production analysis runs only on the remote machine after source-hash validation.

**Tech Stack:** Python 3.11, NumPy, existing KA cache loaders, `unittest`, deterministic CSV/SVG renderers.

## Global Constraints

- Never run the four-clone analysis locally.
- Use exactly four existing `T=0.58` clones, 64 fixed A particles, and `0.01 tau` frame spacing.
- Outer folds are whole-clone leave-one-out; model selection uses only the other three clones.
- Frozen supports are `[4, 16, 40, 100]`; frozen ranks are `[1, 2, 4, 8]`.
- Frozen normalized-design ridges are `[0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]`.
- Positive decay candidates are `logspace(log10(0.05), log10(50), 32)`.
- Event labels and macro observables never enter kernel fitting or model selection.
- Every broad Langevin, event-clock, Kramers, spatial, and thermodynamic claim remains closed.

---

### Task 1: Training-Only Radial Basis And Jacobians

**Files:**
- Create: `src/ka_position_dependent_kernel.py`
- Create: `tests/test_ka_position_dependent_kernel.py`

**Interfaces:**
- Consumes: finite arrays `u[...,3]` from training clones.
- Produces: `fit_radial_basis_scale(position) -> dict`, `radial_vector_basis(position, scale) -> array[...,3,3]`, and `radial_vector_basis_jacobian(position, scale) -> array[...,3,3,3]`.

- [ ] **Step 1: Write the failing basis test**

Verify that held positions cannot change `mu_r2`, `sigma_r2`, or `epsilon_r2`, and compare all three analytic Jacobians with centered finite differences on nonzero vectors.

- [ ] **Step 2: Run the RED test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_ka_position_dependent_kernel.PositionDependentKernelTests.test_training_only_radial_basis_and_jacobian -v
```

Expected: import failure for `ka_position_dependent_kernel`.

- [ ] **Step 3: Implement the basis**

Use

```text
s=(r2-mu_r2)/sigma_r2
h0=u
h1=u*s
h2=u*(s*s-1)
```

with exact Jacobians

```text
grad h0 = I
grad h1 = s I + 2 u u^T/sigma_r2
grad h2 = (s^2-1)I + 4s u u^T/sigma_r2.
```

Reject zero/nonfinite training variance and store the first percentile of positive `r2` as `epsilon_r2`.

- [ ] **Step 4: Run the GREEN test**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ka_position_dependent_kernel.py tests/test_ka_position_dependent_kernel.py
git commit -m "derive training-only radial kernel basis"
```

---

### Task 2: Regularized Finite-Basis Volterra System

**Files:**
- Modify: `src/ka_position_dependent_kernel.py`
- Modify: `tests/test_ka_position_dependent_kernel.py`

**Interfaces:**
- Consumes: training paths `u`, `v`, exact generator drift `a`, a fitted basis scale, memory support, and nonnegative ridge.
- Produces: `assemble_mz_volterra_system(...) -> dict`, `solve_regularized_mz_kernel(system, ridge) -> dict`, and `predict_mz_drift(...) -> array`.

- [ ] **Step 1: Write the failing synthetic recovery test**

Construct a three-basis causal discrete system with a known 4-lag kernel. Require exact recovery at ridge zero for a full-rank noiseless design and finite shrinkage for positive ridge.

- [ ] **Step 2: Run the RED test**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_ka_position_dependent_kernel.PositionDependentKernelTests.test_regularized_mz_system_recovers_causal_kernel -v
```

Expected: missing function failure.

- [ ] **Step 3: Implement one joint causal solve**

Assemble the lower-triangular history features

```text
Phi[t,(lag,j)] = grad h_j(u[t-lag]) v[t-lag]
```

after the support horizon. Fit mean-force and memory coefficients jointly by

```text
theta=(X^T X + ridge D)^(-1)X^T a,
```

with `D=0` on mean-force coefficients and `D=1` on memory coefficients. Use `np.linalg.lstsq` on the augmented system, never an explicit inverse. Return rank, singular values, and condition number.

- [ ] **Step 4: Run the GREEN test**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Add invalid-system tests**

Reject nonfinite paths, mismatched time/particle axes, support outside the path, negative ridge, zero-rank designs, and held arrays passed to the training assembler.

- [ ] **Step 6: Run the focused module**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error::RuntimeWarning -m unittest \
  tests.test_ka_position_dependent_kernel -v
```

Expected: all tests PASS with no warnings.

- [ ] **Step 7: Commit**

```bash
git add src/ka_position_dependent_kernel.py tests/test_ka_position_dependent_kernel.py
git commit -m "solve regularized position-dependent Volterra kernel"
```

---

### Task 3: Real-Pole And Thermodynamic Two-Position Realizations

**Files:**
- Modify: `src/ka_position_dependent_kernel.py`
- Modify: `tests/test_ka_position_dependent_kernel.py`

**Interfaces:**
- Produces: `real_pole_history_features(...)`, `two_position_auxiliary_features(...)`, and `reconstruct_auxiliary_innovations(...)`.

- [ ] **Step 1: Write failing exact-recursion tests**

Use a known positive decay and radial coupling to verify

```text
z[t+1]=rho*z[t]-C(u[t])^T v[t] dt
force[t]=C(u[t])z[t]
```

against direct convolution. Separately prove that signed past-position real-pole coefficients do not open the positive-Prony claim.

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_ka_position_dependent_kernel.PositionDependentKernelTests.test_two_position_auxiliary_recursion_matches_direct_convolution -v
```

Expected: missing function failure.

- [ ] **Step 3: Implement deterministic recursions**

Use exact discrete decay `rho=exp(-alpha*dt)` and the integrated forcing factor `(1-rho)/alpha`. Keep current-position and historical-position coupling factors distinct. Reject nonpositive poles.

- [ ] **Step 4: Run GREEN and the full focused module**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error::RuntimeWarning -m unittest \
  tests.test_ka_position_dependent_kernel -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ka_position_dependent_kernel.py tests/test_ka_position_dependent_kernel.py
git commit -m "construct thermodynamic two-position kernel realization"
```

---

### Task 4: Held Diagnostics And Fail-Closed Classifier

**Files:**
- Modify: `src/ka_position_dependent_kernel.py`
- Modify: `tests/test_ka_position_dependent_kernel.py`

**Interfaces:**
- Produces: `held_kernel_diagnostics(...) -> dict` and `classify_position_dependent_kernel_gate(rows) -> dict`.

- [ ] **Step 1: Write failing classifier tests**

Cover M2 identified/M4 failed, M2 failed, M4 identified with inconsistent rank, and all-pass same-rank cases. Require every broad claim field to equal zero in all outcomes.

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_ka_position_dependent_kernel.PositionDependentKernelTests.test_classifier_separates_mz_real_pole_and_positive_prony_claims -v
```

Expected: missing function failure.

- [ ] **Step 3: Implement mechanical thresholds**

Copy every threshold and claim field exactly from the design spec. Compute replicate-first mean, standard error, and `df=3` 95 percent interval for M2/M1 RMSE ratios. Never pool clone rows before a per-fold pass decision.

- [ ] **Step 4: Run GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ka_position_dependent_kernel.py tests/test_ka_position_dependent_kernel.py
git commit -m "gate real KA kernel identifiability claims"
```

---

### Task 5: Frozen Four-Clone CLI And Artifacts

**Files:**
- Create: `scripts/analyze_ka_position_dependent_kernel.py`
- Create: `tests/test_ka_position_dependent_kernel_analysis.py`
- Modify: `docs/superpowers/specs/2026-07-19-real-ka-position-dependent-kernel-identifiability-design.md`

**Interfaces:**
- Consumes: four decomposed-drift caches and exact provenance paths.
- Produces: `_details.csv`, `_selection.csv`, `_kernel.csv`, `_summary.csv`, `.svg`, and SHA manifest.

- [ ] **Step 1: Write failing CLI/provenance tests**

Require the frozen support/rank/pole grids, exact four-cache completeness, aligned target indices, and recomputed trajectory/drift/source hashes.

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  tests.test_ka_position_dependent_kernel_analysis -v
```

Expected: CLI import or file failure.

- [ ] **Step 3: Implement nested whole-clone selection**

For each outer held clone, select support/ridge/rank/poles only by inner whole-clone folds. Record every candidate, selected hyperparameter, training clone set, and `fit_uses_outer_held_clone=0`.

- [ ] **Step 4: Implement deterministic artifacts**

Render unclipped held RMSE/NLL, residual-correlation, FDT, selected-support/rank, and claim-boundary panels. Include tolerance labels and distinguish failed values rather than clipping them to one top line.

- [ ] **Step 5: Run focused tests and diff audit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error::RuntimeWarning -m unittest \
  tests.test_ka_position_dependent_kernel \
  tests.test_ka_position_dependent_kernel_analysis -v
git diff --check
```

Expected: all tests PASS and no diff errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_ka_position_dependent_kernel.py \
  src/ka_position_dependent_kernel.py \
  tests/test_ka_position_dependent_kernel.py \
  tests/test_ka_position_dependent_kernel_analysis.py \
  docs/superpowers/specs/2026-07-19-real-ka-position-dependent-kernel-identifiability-design.md
git commit -m "implement real KA position-dependent kernel gate"
```

---

### Task 6: Remote Run, Audit, And Claim Update

**Files:**
- Create on remote PASS: frozen CSV/SVG outputs under `/root/renewal-cage-compute-data/output/position-dependent-kernel/`.
- Create after verified download: `docs/microscopic-position-dependent-kernel.md`.
- Modify after verified outcome: `README.md`, package manifest, and package tests.

**Interfaces:**
- Consumes: byte-identical committed source and four complete drift caches.
- Produces: held verdict, per-fold diagnostics, resource log, environment manifest, and artifact SHA-256 values.

- [ ] **Step 1: Verify remote source and environment**

Require exact source hashes, Python/NumPy versions, no conflicting numerical process, and the four input cache hashes before execution.

- [ ] **Step 2: Run one synthetic identity canary remotely**

```bash
/usr/bin/python3.11 -W error::RuntimeWarning -m unittest \
  tests.test_ka_position_dependent_kernel -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run the four-clone analyzer once**

Use `/usr/bin/time -v`, one numerical process, and a remote-only output directory. Do not rerun with changed controls after reading results.

- [ ] **Step 4: Recompute and audit the verdict**

Independently recompute every per-fold threshold and claim flag from raw CSV rows. Confirm all broad claims remain zero.

- [ ] **Step 5: Package only the verified outcome**

Document failures as failures. A nonparametric MZ pass cannot be rewritten as a positive-Prony or autonomous single-particle GLE result.
