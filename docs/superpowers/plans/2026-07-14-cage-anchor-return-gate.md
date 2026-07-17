# Cage-Anchor Return Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a calibration-only two-temperature mechanism gate that distinguishes local one-step recoil from memory of an earlier cage center.

**Architecture:** Add two focused statistical kernels to `src/ka_replicates.py`: an analytic consecutive cage-return statistic and a particle-conditioned radial recoil Markov surrogate. Separate scripts extract real calibration data, score held-out macro curves, and combine both evidence streams without fitting macro observables.

**Tech Stack:** Python 3, NumPy, standard-library CSV/JSON/argparse, `unittest`, existing LAMMPS trajectory and cage-jump utilities.

## Global Constraints

- Use only calibration trajectories to estimate return or recoil kernels.
- Keep block size `20`, radial bins `8`, recoil realizations `16`, radius scales `0.5,1.0,1.5`, and primary radius scale `1.0`.
- Preserve curve tolerances `MSD=0.10`, `NGP=0.30`, `F_s=0.03`.
- Keep `microdynamic_closure_claim_allowed=0`, `spatial_facilitation_claim_allowed=0`, and `thermodynamic_claim_allowed=0`.
- Do not edit manuscript claims until the real-data gate and package tests pass.

---

### Task 1: Cage-Anchor Return Statistics

**Files:**
- Modify: `src/ka_replicates.py`
- Modify: `tests/test_ka_replicates.py`

**Interfaces:**
- Consumes: event dictionaries returned by `extract_debye_waller_cage_jumps`.
- Produces: `consecutive_cage_anchor_returns(events, debye_waller_factor, radius_scale) -> dict[str, float]`.

- [ ] **Step 1: Write failing synthetic return tests**

```python
def test_cage_anchor_returns_detect_exact_backtrack_and_analytic_null(self):
    events = synthetic_events(vectors=[[1, 0, 0], [-1, 0, 0], [1, 0, 0]])
    result = consecutive_cage_anchor_returns(
        events, debye_waller_factor=0.04, radius_scale=1.0
    )
    self.assertEqual(result["return_count"], 2.0)
    self.assertGreater(result["return_fraction"], result["isotropic_null_fraction"])
```

- [ ] **Step 2: Run the focused test and verify the missing-function failure**

Run: `python3 -m unittest tests.test_ka_replicates.KAReplicatePreparationTests.test_cage_anchor_returns_detect_exact_backtrack_and_analytic_null -v`

- [ ] **Step 3: Implement validation, analytic isotropic probability, run accounting, and duration quantiles**

```python
def consecutive_cage_anchor_returns(events, *, debye_waller_factor, radius_scale):
    threshold = radius_scale * math.sqrt(debye_waller_factor)
    adjacent = events["particle"][:-1] == events["particle"][1:]
    first = events["jump_vector"][:-1][adjacent]
    second = events["jump_vector"][1:][adjacent]
    returned = np.linalg.norm(first + second, axis=1) <= threshold
    cosine_limit = (
        threshold**2
        - np.sum(first**2, axis=1)
        - np.sum(second**2, axis=1)
    ) / (2.0 * np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1))
    isotropic_probability = np.clip((cosine_limit + 1.0) / 2.0, 0.0, 1.0)
    return summarize_return_mask(returned, isotropic_probability, events, adjacent)
```

- [ ] **Step 4: Add malformed-input, particle-boundary, and geometric-run tests; run all focused tests**

Run: `python3 -m unittest tests.test_ka_replicates.KAReplicatePreparationTests -q`

- [ ] **Step 5: Commit**

```bash
git add src/ka_replicates.py tests/test_ka_replicates.py
git commit -m "measure reversible cage anchor returns"
```

### Task 2: One-Step Radial Recoil Markov Null

**Files:**
- Modify: `src/ka_replicates.py`
- Modify: `tests/test_ka_replicates.py`

**Interfaces:**
- Consumes: finite arrays shaped `(particles, blocks, 3)` and a NumPy generator.
- Produces: `radial_recoil_markov_surrogate(block_displacements, rng, radial_bin_count) -> np.ndarray` and `radial_recoil_markov_quality(reference, candidate) -> dict[str, float]`.

- [ ] **Step 1: Write failing deterministic and information-preservation tests**

```python
def test_radial_recoil_markov_is_deterministic_and_removes_triplet_order(self):
    first = radial_recoil_markov_surrogate(path, np.random.default_rng(7), 2)
    second = radial_recoil_markov_surrogate(path, np.random.default_rng(7), 2)
    np.testing.assert_allclose(first, second)
    self.assertFalse(np.array_equal(first, path))
```

- [ ] **Step 2: Run the focused tests and verify missing-function failures**

Run: `python3 -m unittest tests.test_ka_replicates.KAReplicatePreparationTests.test_radial_recoil_markov_is_deterministic_and_removes_triplet_order -v`

- [ ] **Step 3: Implement equal-count source-radius bins and rotationally invariant transition sampling**

```python
def radial_recoil_markov_surrogate(block_displacements, rng, *, radial_bin_count):
    radii = np.linalg.norm(block_displacements, axis=2)
    source_order = np.argsort(radii[:, :-1], axis=1)
    source_groups = np.array_split(source_order, radial_bin_count, axis=1)
    output = initialize_from_empirical_blocks(block_displacements, rng)
    for block_index in range(1, block_displacements.shape[1]):
        transition = sample_radius_conditioned_transition(
            block_displacements, radii, source_groups, output[:, block_index - 1], rng
        )
        output[:, block_index] = rotate_with_isotropic_azimuth(transition, rng)
    return output
```

- [ ] **Step 4: Implement quality metrics and degenerate-vector validation; run preparation tests**

Run: `python3 -m unittest tests.test_ka_replicates.KAReplicatePreparationTests -q`

- [ ] **Step 5: Commit**

```bash
git add src/ka_replicates.py tests/test_ka_replicates.py
git commit -m "add local recoil Markov null"
```

### Task 3: Calibration and Held-Out Analysis Scripts

**Files:**
- Create: `scripts/analyze_ka_cage_anchor_returns.py`
- Create: `scripts/analyze_ka_recoil_markov_transfer.py`
- Modify: `tests/test_ka_replicates.py`

**Interfaces:**
- The return script produces `<prefix>_rows.csv` and `<prefix>_verdict.csv`.
- The recoil script produces `<prefix>_rows.csv`, `<prefix>_quality.csv`, `<prefix>_summary.csv`, and `<prefix>_verdict.csv`.

- [ ] **Step 1: Write failing tests for deterministic seeds, held-out exclusion, replicate-first aggregation, and quality truth tables**

```python
def test_recoil_transfer_requires_quality_before_curve_decision(self):
    verdict = classify_recoil_transfer(quality_rows, summary_rows)
    self.assertEqual(verdict["quality_pass"], 0.0)
    self.assertEqual(verdict["mechanism_state"], "unresolved_quality")
```

- [ ] **Step 2: Run new script tests and verify import/file failures**

Run: `python3 -m unittest tests.test_ka_replicates.KACageAnchorGateTests -v`

- [ ] **Step 3: Implement return extraction and recoil realization analysis using existing trajectory readers**

```bash
python3 scripts/analyze_ka_cage_anchor_returns.py --help
python3 scripts/analyze_ka_recoil_markov_transfer.py --help
```

- [ ] **Step 4: Run script tests and the complete replicate test module**

Run: `python3 -m unittest tests.test_ka_replicates -q`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_ka_cage_anchor_returns.py scripts/analyze_ka_recoil_markov_transfer.py tests/test_ka_replicates.py
git commit -m "score cage anchor and recoil mechanisms"
```

### Task 4: Two-Temperature Decision and Figure

**Files:**
- Create: `scripts/summarize_ka_cage_anchor_gate.py`
- Modify: `tests/test_ka_replicates.py`
- Modify: `tests/test_arxiv_package.py`

**Interfaces:**
- Produces `classify_cage_anchor_gate(low_returns, high_returns, low_recoil, high_recoil) -> dict[str, object]`, one crossover CSV, and one SVG.

- [ ] **Step 1: Write failing decision-table and claim-boundary tests**

```python
def test_gate_selects_anchor_only_for_return_separation_and_low_only_markov_failure(self):
    result = classify_cage_anchor_gate(low_returns, high_returns, low_recoil, high_recoil)
    self.assertEqual(result["cage_anchor_memory_required"], 1.0)
    self.assertEqual(result["thermodynamic_claim_allowed"], 0.0)
```

- [ ] **Step 2: Verify failures, then implement strict schema checks and SVG rendering**

Run: `python3 -m unittest tests.test_ka_replicates.KACageAnchorGateTests -v`

- [ ] **Step 3: Generate real T=0.45 and T=0.58 artifacts with the frozen parameters**

Run both analyzers with the existing ensemble directories and factorization files, then run `scripts/summarize_ka_cage_anchor_gate.py`.

- [ ] **Step 4: Add package assertions for replicate counts, thresholds, quality, mechanism decision, and claim boundaries**

Run: `python3 -m unittest tests.test_arxiv_package -q`

- [ ] **Step 5: Commit**

```bash
git add scripts/summarize_ka_cage_anchor_gate.py data figures tests
git commit -m "identify cooling induced cage anchor memory"
```

### Task 5: Verification and Scientific Audit

**Files:**
- Modify only files required by failures found in this task.

- [ ] **Step 1: Run full tests**

Run: `python3 -m unittest discover -s tests -q`

- [ ] **Step 2: Build the arXiv source package and check whitespace**

Run: `python3 scripts/build_arxiv_package.py && git diff --check`

- [ ] **Step 3: Inspect the SVG at native size and verify no text overlap**

Open `figures/renewal_cage_ka_cage_anchor_gate.svg` in the browser and inspect the full  view.

- [ ] **Step 4: Audit the final CSV against every frozen decision condition and record any unresolved model boundary**

Expected: the gate may identify cage-anchor memory, but all microdynamic, spatial, and thermodynamic closure flags remain zero.

- [ ] **Step 5: Commit any verification-only corrections, leaving the branch local unless explicitly asked to push**
