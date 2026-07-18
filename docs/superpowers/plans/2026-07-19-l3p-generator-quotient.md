# Microscopic L3p Generator Quotient Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute `L^3p` directly from the many-particle KA Langevin generator and test, on four held clones, whether it removes the `L^2p` residual memory and non-Gaussianity left after exact-`Q_c` whitening.

**Architecture:** A new microscopic evaluator combines a centered zero-temperature position derivative of `L^2p`, the existing deterministic `D_V(L^2p)` matrix, and a paired trace estimate of the cage Laplacian thermal correction. A resumable remote cache records every component and numerical sensitivity. A separate held analyzer appends real, time-permuted, or backward-difference fifth coordinates to the existing Mori-40/VAR-16 state and applies the committed exact-`Q_c` covariance model to the resulting `L^2p` innovation.

**Tech Stack:** Python 3.12, NumPy, `unittest`, existing KA trajectory and smooth-cage utilities, remote Python 3.11 for production numerical runs.

## Global Constraints

- Work only in `/Users/luicy/AI/renewal-cage-arxiv/.worktrees/l3p-generator-quotient`.
- Base the stacked branch on commit `8b7fda696af763f2c9462802e33418a7d1b3766b` until PR #21 merges.
- Do not modify the four trajectory inputs, hashes, target indices, frame count 200, Mori order 40, VAR order 16, lag limit 40, or exact-`Q_c` score tolerances.
- Use trace prefixes `{4,8,16,32}`, trace seed `20260731`, time-null seed `20260802`, and the step ladders frozen in the design.
- Do not read held quotient scores until every clone passes the numerical gate.
- Run production canaries and caches only on `root@47.94.164.38`, sequentially, with at most one numerical process.
- Keep every broad microscopic, autonomous-GLE, event-clock, Kramers, spatial, and thermodynamic claim flag at zero.

---

### Task 1: Implement the exact quotient assembly and paired cage-Laplacian estimator

**Files:**
- Create: `src/ka_l3p_generator.py`
- Create: `tests/test_ka_l3p_generator.py`

**Interfaces:**
- Consumes: `smooth_force_support_cage_batch`, fixed full-state Rademacher probes, the deterministic `l2p_velocity_jacobian`, and component arrays with target-vector trailing shape.
- Produces: `smooth_cage_laplacian_prefixes(...) -> dict` with `laplacian_prefixes` of shape `(prefix_count, targets, 3)`.
- Produces: `assemble_l3p_generator(...) -> dict` with `l3p_prefixes`, `position_transport_term`, `acceleration_response_term`, `thermal_gradient_prefixes`, and `thermal_friction_prefixes`.

- [ ] **Step 1: Write the failing trace-estimator tests**

Add tests that pass the complete scaled coordinate basis
`sqrt(3N) * {e_i}` for a small smooth cage, whose mean outer product is the
identity, and compare the final prefix with an explicit centered divergence of
the analytic cage Jacobian. Require rigid-translation invariance and exact zero
for a two-particle linear relative coordinate.

```python
result = smooth_cage_laplacian_prefixes(
    positions,
    particle_types=particle_types,
    box_lengths=box_lengths,
    target_indices=targets,
    trace_probes=coordinate_probes,
    directional_step=1e-5,
    prefix_counts=(len(coordinate_probes),),
)
np.testing.assert_allclose(
    result["laplacian_prefixes"][-1], exhaustive_laplacian,
    rtol=2e-7,
    atol=2e-8,
)
```

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tests.test_ka_l3p_generator.L3pGeneratorTests.test_cage_laplacian_prefix_matches_exhaustive_basis -v
```

Expected: fail because `ka_l3p_generator` does not exist.

- [ ] **Step 3: Implement the paired Hessian-action estimator**

For each probe `xi`, evaluate

```python
hessian_action = (
    relative_velocity(R + h * xi, velocities=xi)
    - relative_velocity(R - h * xi, velocities=xi)
) / (2.0 * h)
```

and form each prefix mean. Validate sorted unique prefixes, exact probe shape, finite inputs, and `max(prefix_counts) <= len(trace_probes)`.

- [ ] **Step 4: Write the failing quotient-assembly tests**

Use independent random arrays and require

```python
acceleration_term = np.einsum("tanb,nb->ta", A_c, acceleration)
expected = (
    position_term[None]
    + acceleration_term[None]
    + 8.0 * gamma * temperature * laplacian_gradient_prefixes
    - 6.0 * gamma**2 * temperature * laplacian_prefixes
)
```

Also require exact zero thermal terms at `temperature=0` and fail-closed shape validation.

- [ ] **Step 5: Implement the minimal pure assembly and run GREEN**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tests.test_ka_l3p_generator -v
git add src/ka_l3p_generator.py tests/test_ka_l3p_generator.py
git commit -m "derive microscopic L3p quotient components"
```

---

### Task 2: Build the full single-frame microscopic `L^3p` evaluator

**Files:**
- Modify: `src/ka_l3p_generator.py`
- Modify: `tests/test_ka_l3p_generator.py`

**Interfaces:**
- Consumes: one full KA phase-space frame, force protocol, fixed targets, trace probes, `position_step`, `cage_hessian_step`, and `jacobian_step`.
- Produces: `smooth_cage_l3p_generator_batch(...) -> dict` with final `l3p`, all prefix/component arrays, pair count, steps, and zero claim flags.

- [ ] **Step 1: Write the failing harmonic identity test**

For linear `u=JR` and harmonic `F=-KR`, require

```text
L^3p = J[(K^2-gamma^2 K)R + (2 gamma K-gamma^3 I)V].
```

Test the pure helper used by the production evaluator to `rtol=2e-10`, `atol=2e-10`.

- [ ] **Step 2: Write the failing direct-generator test on a small KA frame**

Construct `c0(R,V)` with `smooth_cage_second_generator_batch(..., temperature=0, trace_probes=None)`. Use centered exhaustive phase-space derivatives to evaluate

```text
V.grad_R c0 + [F-gamma V].grad_V c0 + gamma T Delta_V c0
+ 2 gamma T V.grad_R Delta_R u
```

and compare with `smooth_cage_l3p_generator_batch` at `rtol=2e-4`, `atol=2e-6`.

- [ ] **Step 3: Run both tests and verify RED**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest \
  tests.test_ka_l3p_generator.L3pGeneratorTests.test_harmonic_l3p_identity \
  tests.test_ka_l3p_generator.L3pGeneratorTests.test_l3p_matches_exhaustive_phase_space_generator -v
```

- [ ] **Step 4: Implement the centered microscopic evaluator**

At `R +/- h_R V`, call `ka_lj_sparse_force_generator_observables` to recompute both `F` and `DF[V]`, then call the second-generator evaluator at zero temperature. At the base frame, construct `A_c` with `smooth_cage_l2p_velocity_jacobian_batch`, apply it to `F-gamma V`, evaluate paired Laplacian prefixes at `R` and `R +/- h_R V`, and call `assemble_l3p_generator`.

Return the exact keys

```python
{
    "l3p": l3p_prefixes[-1],
    "l3p_prefixes": l3p_prefixes,
    "position_transport_term": position_term,
    "acceleration_response_term": acceleration_term,
    "thermal_gradient_prefixes": thermal_gradient,
    "thermal_friction_prefixes": thermal_friction,
    "laplacian_prefixes": laplacian,
    "laplacian_velocity_derivative_prefixes": laplacian_gradient,
    "thermodynamic_claim_allowed": 0.0,
}
```

- [ ] **Step 5: Verify the independent `A_c a` directional identity**

Compare the cached acceleration term with
`smooth_cage_l2p_velocity_directional_derivative_batch` using
`velocity_direction=F-gamma V` and its exact force-generator direction.

- [ ] **Step 6: Run focused old and new suites and commit**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tests.test_ka_l3p_generator tests.test_ka_l2p_conditional_diffusion tests.test_ka_smooth_cage -v
git add src/ka_l3p_generator.py tests/test_ka_l3p_generator.py
git commit -m "evaluate microscopic L3p on KA frames"
```

---

### Task 3: Add the fail-closed numerical classifier and resumable cache

**Files:**
- Modify: `src/ka_l3p_generator.py`
- Create: `scripts/cache_ka_l3p_generator.py`
- Modify: `tests/test_ka_l3p_generator.py`

**Interfaces:**
- Produces: `classify_l3p_numerical_canary(...) -> dict[str,float|str]` implementing the six frozen numerical gates.
- Produces: `clone_NNN_l3p_generator.npz` with frame-aligned values, sensitivities, provenance, and checkpoint count.

- [ ] **Step 1: Write failing classifier boundary tests**

Build synthetic primary/reference/coarse/prefix arrays exactly on each tolerance boundary, then perturb one metric above its limit. Require `l3p_numerical_gate_pass` to change from 1 to 0 without changing any broad claim flag.

```python
verdict = classify_l3p_numerical_canary(rows)
self.assertEqual(verdict["l3p_numerical_gate_pass"], 1.0)
self.assertEqual(verdict["thermodynamic_claim_allowed"], 0.0)
```

- [ ] **Step 2: Write failing CLI and cache-provenance tests**

Require CLI defaults for all frozen steps, prefixes, seeds, potential protocol, checkpoint interval, frame limit, and target batch size. Reject existing caches with mismatched trajectory SHA, targets, potential, any step, trace probes, seed, or requested frame count.

- [ ] **Step 3: Run tests and verify RED**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest \
  tests.test_ka_l3p_generator.L3pGeneratorTests.test_l3p_numerical_classifier_is_fail_closed \
  tests.test_ka_l3p_generator.L3pGeneratorTests.test_l3p_cache_cli_freezes_provenance -v
```

- [ ] **Step 4: Implement the classifier and atomic checkpoint cache**

Allocate frame-first arrays for final/prefix/component values and numerical errors. Save after every frame with `atomic_savez`. A completed matching cache exits without recomputation. Only the first requested frame pays for primary/reference/coarse and independent directional checks; production frames use the accepted primary settings.

- [ ] **Step 5: Verify checkpoint resume and commit**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tests.test_ka_l3p_generator -v
git add src/ka_l3p_generator.py scripts/cache_ka_l3p_generator.py tests/test_ka_l3p_generator.py
git commit -m "cache fail-closed microscopic L3p generators"
```

---

### Task 4: Implement the held generator-coordinate quotient and nulls

**Files:**
- Create: `src/ka_l3p_quotient.py`
- Create: `scripts/analyze_ka_l3p_generator_quotient.py`
- Create: `tests/test_ka_l3p_quotient.py`

**Interfaces:**
- Produces: `append_generator_coordinate(state, coordinate)`, `backward_l2p_difference(...)`, and `permute_l3p_by_clone_target(...)`.
- Produces: `extract_l3p_quotient_fold(...)` returning aligned training/held `L2p` innovations and source-frame indices for each model.
- Produces: `classify_l3p_generator_quotient(details, convergence) -> dict` with all frozen outcome and claim flags.

- [ ] **Step 1: Write failing state/null tests**

Use a synthetic five-block chain. Require the real fifth block to preserve time alignment, the time null to preserve each target's three-vector marginal exactly while changing its order, and the backward null to use only `c_n-c_{n-1}`. Require all four models to discard the same first source frame.

- [ ] **Step 2: Write a failing held synthetic mechanism test**

Generate `c` innovations whose conditional mean depends on a known `d` signal. Require the real generator model to beat baseline, permuted, and backward nulls in every held fold and set `l3p_generator_coordinate_informative=1`. Replace `d` with independent noise and require the flag to remain zero.

- [ ] **Step 3: Write failing classifier boundary tests**

Cover every-fold NLL ordering, replicate-first t interval, 25% squared-memory reduction, 25% kurtosis reduction, numerical support, and each absolute closure limit independently. Require

```text
finite_l3p_gaussian_closure_supported = 0
microscopic_environment_coordinate_z_allowed = 0
continuous_gaussian_langevin_bath_allowed = 0
autonomous_single_particle_gle_allowed = 0
complete_event_clock_closure_allowed = 0
kramers_escape_claim_allowed = 0
spatial_facilitation_claim_allowed = 0
thermodynamic_claim_allowed = 0
```

for every synthetic outcome.

- [ ] **Step 4: Implement the paired fold analyzer**

Reuse `second_generator_resolved_state`, `discrete_mori_zwanzig_operators`, the VAR whitening functions, exact-`Q_c` cache loading, `fit_scaled_conditional_covariance`, and `conditional_covariance_diagnostic`. Share common-coordinate training means/scales and fit no quantity on a held clone.

- [ ] **Step 5: Add deterministic CSV/SVG output**

Write details, convergence, and summary CSVs. The SVG has four model lines on unclipped axes for NLL improvement and squared-memory/kurtosis ratios, marks tolerances `1.0`, and labels the real coordinate separately from both nulls.

- [ ] **Step 6: Run tests and commit**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tests.test_ka_l3p_quotient tests.test_ka_l3p_generator -v
git add src/ka_l3p_quotient.py scripts/analyze_ka_l3p_generator_quotient.py tests/test_ka_l3p_quotient.py
git commit -m "gate held microscopic L3p generator quotient"
```

---

### Task 5: Run the remote one-frame canary and freeze the numerical outcome

**Files:**
- Modify only after the run: `docs/superpowers/specs/2026-07-19-l3p-generator-quotient-design.md`
- Remote ignored caches: `/root/renewal-cage-compute-data/output/l3p-generator-canary/`

**Interfaces:**
- Consumes: clone 1 raw trajectory, drift cache, second-generator cache, exact-`Q_c` cache, and byte-identical branch source.
- Produces: one-frame primary/reference/coarse/prefix diagnostics and `/usr/bin/time -v` resource evidence.

- [ ] **Step 1: Transfer source by content hash and verify remote dependencies**

Use a binary git patch or explicit file transfer. Compare SHA256 before execution. Do not modify or reset the remote raw-data directories.

- [ ] **Step 2: Run one synthetic remote identity test**

```bash
python3.11 -m unittest tests.test_ka_l3p_generator -v
```

Expected: all tests pass under the remote interpreter.

- [ ] **Step 3: Run the frozen real-frame canary sequentially**

Record wall time, maximum RSS, swap count, all numerical metrics, source hash, trajectory hash, and cache hash. Do not launch the four-clone run if any numerical gate fails.

- [ ] **Step 4: Append only the mechanical numerical outcome and commit**

Do not change steps or tolerances. If the gate fails, record the failed estimator and stop the physical branch. If it passes, record `l3p_numerical_gate_pass=1` and proceed.

---

### Task 6: Generate four remote caches and run the held quotient

**Files:**
- Create on numerical PASS: `data/renewal_cage_ka_l3p_generator_quotient_T058_details.csv`
- Create on numerical PASS: `data/renewal_cage_ka_l3p_generator_quotient_T058_convergence.csv`
- Create on numerical PASS: `data/renewal_cage_ka_l3p_generator_quotient_T058_summary.csv`
- Create on numerical PASS: `figures/renewal_cage_ka_l3p_generator_quotient_T058.svg`
- Create: `docs/microscopic-l3p-generator-quotient.md`

**Interfaces:**
- Consumes: four complete 200-frame L3p caches and four exact-`Q_c` caches.
- Produces: the frozen four-model held verdict with replicate-first uncertainty.

- [ ] **Step 1: Run all caches one clone at a time**

Use checkpoint resume, never run clones concurrently, and confirm no swap for each completed job. Verify all 800 frames and all four trajectory hashes before analysis.

- [ ] **Step 2: Run the held analyzer once**

Do not alter null seeds, model families, fit inputs, or tolerances. Save stdout, command, source commit, and artifact hashes.

- [ ] **Step 3: Mechanically audit every verdict field**

Recompute the classifier from raw detail/convergence rows. Report every-fold values, replicate-first intervals, all four model comparisons, all claim flags, and whether `Q_d` derivation is authorized.

- [ ] **Step 4: Inspect the SVG at exact dimensions**

Render with Playwright or Poppler, verify unclipped axes, visible tolerance labels, no overlapping text, and no implication that clipped values are equal.

- [ ] **Step 5: Write the result document without broadening claims**

Include the exact identity, numerical outcome, held result, null failures, physical interpretation, and all zero claim flags.

---

### Task 7: Package, verify, publish, and update the stacked PR

**Files:**
- Modify: `README.md`
- Modify: `scripts/build_arxiv_package.py`
- Modify: `tests/test_arxiv_package.py`
- Create on PASS: `paper/figures/renewal_cage_ka_l3p_generator_quotient_T058.pdf`

**Interfaces:**
- Produces: deterministic arXiv PDF/CSV inclusion and a test that recomputes the result from committed rows.

- [ ] **Step 1: Write the failing package recomputation test**

Require all scripts/docs/data/figures, exact row counts, mechanical classifier equality, fixed claim flags, PDF/CSV archive entries, and the documented numerical values.

- [ ] **Step 2: Run the package test and verify RED**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tests.test_arxiv_package.ArxivPackageTests.test_l3p_generator_quotient_is_recomputed_and_claim_limited -v
```

- [ ] **Step 3: Implement deterministic package rendering and run GREEN**

Generate the PDF from committed CSV rows with invariant metadata. Add the PDF and three CSVs to the source zip.

- [ ] **Step 4: Run full verification**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s tests -v
git diff --check
git status --short
```

Run `scripts/generate_renewal_cage_results.py` and `scripts/build_arxiv_package.py` on the remote node, not as a local production calculation. Compare committed artifact hashes with remote regeneration.

- [ ] **Step 5: Commit, push, and open/update a stacked draft PR**

```bash
git add README.md src scripts tests docs data figures paper/figures
git commit -m "test microscopic L3p generator quotient"
git push -u origin codex/l3p-generator-quotient
```

Target PR #21's branch until it merges; after merge, rebase safely onto
`origin/main` in this worktree and retarget the PR. Wait for all CI checks and
report the exact head SHA and merge state.
