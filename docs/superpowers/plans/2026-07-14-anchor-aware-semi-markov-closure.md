# Anchor-Aware Semi-Markov Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and test a directly observed anchor-aware Markov-renewal kernel, then decide from frozen two-temperature held-out MSD, NGP, and multi-k scattering whether semi-Markov timing or exact reversible-cage geometry is required.

**Architecture:** A focused core module extracts per-particle return/escape transition records and simulates either the anchor-aware kernel or a state-schedule control. A reusable analysis script converts synthetic physical event times to block paths, combines them with the existing calibration-only cage residual, and scores held-out curves. A separate selector validates provenance and recomputes the final two-temperature mechanism verdict.

**Tech Stack:** Python 3, NumPy, standard-library `csv/json/argparse/unittest`, existing `ka_replicates` trajectory and observable helpers, deterministic SVG output.

## Global Constraints

- Work in `/Users/luicy/AI/renewal-cage-arxiv` on the existing branch; do not create a worktree.
- Use only `T=0.45` calibration time `5000` with three replicates and `T=0.58` calibration time `750` with five replicates.
- Use fluctuation half-window `5`, return radius `sqrt(debye_waller_factor)`, block size `20`, eight equal-count radial bins, wave numbers `2,4,7.25`, and 16 deterministic realizations per replicate.
- Held-out information is target-only. Keep macro-fit parameter count and held-out calibration-use flags at zero.
- Frozen tolerances are MSD `0.10`, NGP `0.30`, multi-k scattering `0.03`; Monte Carlo tolerances are relative MSD `0.01`, NGP `0.03`, and scattering `0.003`.
- A low-temperature MSD failure rejects the candidate even if NGP or scattering improves.
- Keep `microdynamic_closure_claim_allowed=0`, `spatial_facilitation_claim_allowed=0`, and `thermodynamic_claim_allowed=0` in every output.

---

### Task 1: Extract Directly Observed Anchor Transitions

**Files:**
- Create: `src/ka_anchor_semi_markov.py`
- Create: `tests/test_ka_anchor_semi_markov.py`

**Interfaces:**
- Consumes: event dictionaries from `extract_debye_waller_cage_jumps` with `particle`, `time`, `jump_vector`, `pre_center`, and `post_center` arrays.
- Produces: `extract_anchor_transition_kernel(events, *, debye_waller_factor, radial_bin_count) -> dict[str, object]`.
- Kernel arrays: `particle`, `current_state`, `next_state`, `holding_time`, `normalized_holding_time`, `source_radius_bin`, `target_radius_quantile`, `relative_cosine`, `closure_distance`, and source event indices.
- Particle profiles: mean wait, sorted jump radii, initial state/vector, active particle ids, and non-propagating particle ids.

- [ ] **Step 1: Write failing state-label and boundary tests**

Add tests that construct two particles with exact `A -> B -> A` return paths and `A -> B -> C` escape paths. Assert the return definition is `|post_n - pre_(n-1)| <= sqrt(DW)`, times never join across particles, source bins are in `[0,7]`, and malformed or unsorted arrays fail.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
python3 -m unittest tests.test_ka_anchor_semi_markov.AnchorTransitionExtractionTests -v
```

Expected: failure because `ka_anchor_semi_markov` or `extract_anchor_transition_kernel` does not exist.

- [ ] **Step 3: Implement strict extraction**

Validate aligned finite three-dimensional inputs, grouped particle ids, and increasing within-particle times. Build event states first, then records only from triplets within one particle. Compute each particle's mean inter-event wait and empirical radius quantile mapping. Use deterministic equal-count source-radius bins and reject empty transition support.

- [ ] **Step 4: Run focused extraction tests and the existing anchor tests**

Run:

```bash
python3 -m unittest tests.test_ka_anchor_semi_markov.AnchorTransitionExtractionTests tests.test_ka_replicates.KAReplicatePreparationTests.test_cage_anchor_returns_detect_exact_backtrack_and_analytic_null -v
```

Expected: all pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/ka_anchor_semi_markov.py tests/test_ka_anchor_semi_markov.py
git commit -m "extract observed cage anchor transitions"
```

---

### Task 2: Generate Anchor and State-Schedule Paths

**Files:**
- Modify: `src/ka_anchor_semi_markov.py`
- Modify: `tests/test_ka_anchor_semi_markov.py`

**Interfaces:**
- Consumes: the Task 1 kernel and a NumPy `Generator`.
- Produces: `simulate_anchor_semi_markov(kernel, rng, *, duration, maximum_lag, model) -> dict[str, np.ndarray]` for `model in {"anchor_aware_semi_markov", "state_schedule_without_anchor_geometry"}`.
- Synthetic event fields: `particle`, integer `time`, `jump_vector`, `scheduled_state`, `geometric_return`, `holding_time`, and `unsupported_tuple_count`.

- [ ] **Step 1: Write failing deterministic geometry tests**

Test that fixed seeds reproduce identical paths; return tuples satisfy the sampled closure distance without clipping; escape and control vectors have isotropic, noncoplanar azimuths over many draws; state-schedule and anchor models share state/wait/radius draws for the same seed; and an impossible return tuple raises `ValueError`.

- [ ] **Step 2: Run simulator tests and confirm RED**

```bash
python3 -m unittest tests.test_ka_anchor_semi_markov.AnchorSemiMarkovSimulationTests -v
```

Expected: failure because the simulator is absent.

- [ ] **Step 3: Implement the simulator**

Initialize each active particle from its calibration profile. Sample joint transition records by current state and source-radius bin. Map target radial quantiles through the generated particle's empirical radius distribution. Convert normalized waits to physical frames with `max(1, int(round(value)))`. For returns, compute the exact polar cosine from closure geometry and reject unsupported values rather than clipping. For the control, draw one-step cosine independently from the matching radial bin. Draw uniform 3D azimuths and continue to `duration + maximum_lag`.

- [ ] **Step 4: Implement synthetic quality statistics**

Add `anchor_path_quality(kernel, synthetic, *, model) -> dict[str, float]` reporting scheduled state and transition errors, state-conditioned wait mean/quantile errors, radial and recoil errors, geometric return error, closure-quantile error, support failures, and particle accounting.

- [ ] **Step 5: Run all core tests**

```bash
python3 -m unittest tests.test_ka_anchor_semi_markov -v
```

Expected: all pass with no warnings.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/ka_anchor_semi_markov.py tests/test_ka_anchor_semi_markov.py
git commit -m "simulate anchor aware semi markov paths"
```

---

### Task 3: Score Calibration-Only Held-Out Transfer

**Files:**
- Create: `scripts/analyze_ka_anchor_semi_markov_transfer.py`
- Modify: `tests/test_ka_anchor_semi_markov.py`

**Interfaces:**
- CLI inputs: ensemble directory, Debye-Waller replicate table, calibration and held-out factorization tables, calibration time, block size `20`, radial bins `8`, realizations `16`, base seed, and output prefix.
- Outputs: `<prefix>_transitions.csv`, `_quality.csv`, `_rows.csv`, `_summary.csv`, and `_verdict.csv`.
- Public classifier: `classify_anchor_transfer(quality_rows, summary_rows, replicate_rows, *, model, expected_replicates) -> dict[str, object]`.

- [ ] **Step 1: Write failing classifier and provenance tests**

Construct synthetic rows that independently violate return-state quality, wait quality, recoil quality, closure quality, Monte Carlo precision, MSD, NGP, scattering, exact replicate count, realization completeness, calibration time, block size, radial bins, held-out-use flags, and claim flags. Require failure in each case and require the control not to gate geometric return or closure errors.

- [ ] **Step 2: Run analysis tests and confirm RED**

```bash
python3 -m unittest tests.test_ka_anchor_semi_markov.AnchorTransferAnalysisTests -v
```

Expected: failure because the analysis module is absent.

- [ ] **Step 3: Implement realization analysis**

For each replicate, load the trajectory once, extract calibration events, build the kernel, and generate both models with paired seeds. Bin synthetic events into particle-by-block jump vectors. Use `cumulative_block_observables` at every common lag and combine event observables with the calibration residual via `compound_jump_cage_observables`. Read held-out rows only for observed targets.

- [ ] **Step 4: Implement replicate-first aggregation and precision**

Aggregate realizations within replicate, calculate Monte Carlo standard errors, then aggregate equally across independent replicates. Serialize all calibration/held-out provenance and all three claim flags in every table.

- [ ] **Step 5: Implement strict CLI controls**

Reject any block size other than `20`, radial-bin count other than `8`, realization count other than `16`, incorrect calibration horizon for the manifest temperature, incomplete replicate set, duplicate seed, or missing factorization lag.

- [ ] **Step 6: Run analysis and regression tests**

```bash
python3 -m unittest tests.test_ka_anchor_semi_markov.AnchorTransferAnalysisTests tests.test_ka_replicates.KAReplicatePreparationTests.test_recoil_transfer_requires_quality_before_curve_decision -v
```

Expected: all pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add scripts/analyze_ka_anchor_semi_markov_transfer.py tests/test_ka_anchor_semi_markov.py
git commit -m "score anchor semi markov transfer"
```

---

### Task 4: Select the Two-Temperature Mechanism

**Files:**
- Create: `scripts/summarize_ka_anchor_semi_markov_gate.py`
- Modify: `tests/test_ka_anchor_semi_markov.py`

**Interfaces:**
- Consumes: both temperatures' quality, summary, replicate, and verdict tables; frozen one-step recoil verdict/rows; and contiguous empirical-path verdicts.
- Produces: `classify_anchor_semi_markov_gate(...) -> dict[str, object]`, one combined CSV, and one deterministic SVG.

- [ ] **Step 1: Write failing decision-table tests**

Cover exactly four states: `anchor_geometry_required_within_tested_models`, `semi_markov_state_clock_sufficient_anchor_not_identified`, `anchor_aware_model_rejected`, and `mechanism_unresolved`. Independently toggle low/high anchor closure, low control closure, low replicate improvement, one-step baseline provenance, contiguous upper-bound closure, quality, precision, and all claim flags.

- [ ] **Step 2: Run selector tests and confirm RED**

```bash
python3 -m unittest tests.test_ka_anchor_semi_markov.AnchorGateSelectionTests -v
```

Expected: failure because the selector is absent.

- [ ] **Step 3: Implement provenance-first selection**

Recompute every numerical maximum from input rows, require exact temperature/calibration pairs and exact realization grids, and reject stale stored pass flags. Require every low-temperature replicate's higher-order score to improve strictly over its paired one-step recoil score.

- [ ] **Step 4: Implement deterministic SVG**

Plot normalized MSD, NGP, and maximum scattering errors for the one-step, state-schedule, anchor-aware, and contiguous-path models at both temperatures. Include the selected state and a visible two-line claim-boundary note without clipping.

- [ ] **Step 5: Run selector tests and inspect an adversarial fixture SVG**

```bash
python3 -m unittest tests.test_ka_anchor_semi_markov.AnchorGateSelectionTests -v
```

Expected: all pass and no `nan` or `inf` in SVG text.

- [ ] **Step 6: Commit Task 4**

```bash
git add scripts/summarize_ka_anchor_semi_markov_gate.py tests/test_ka_anchor_semi_markov.py
git commit -m "select anchor semi markov mechanism"
```

---

### Task 5: Run the Frozen Real Protocol

**Files:**
- Create: `data/renewal_cage_ka_replicates_T045_anchor_semi_markov_*.csv`
- Create: `data/renewal_cage_ka_replicates_T058_anchor_semi_markov_*.csv`
- Create: `data/renewal_cage_ka_anchor_semi_markov_gate.csv`
- Create: `figures/renewal_cage_ka_anchor_semi_markov_gate.svg`

**Interfaces:**
- Consumes the committed scripts and existing trajectory/factorization data.
- Produces the exact real-data artifacts required by Task 4.

- [ ] **Step 1: Run `T=0.45` without changing controls**

Use calibration time `5000`, block size `20`, radial bins `8`, 16 realizations, and a recorded base seed. Capture runtime and exit status.

- [ ] **Step 2: Run `T=0.58` with the same structural controls**

Use calibration time `750` and otherwise identical controls. Do not rerun selectively with new seeds after seeing curve errors.

- [ ] **Step 3: Run the combined selector**

Generate the final CSV and SVG from all committed input tables plus the frozen recoil and contiguous-path artifacts.

- [ ] **Step 4: Interpret the result without changing thresholds**

Report which of the four frozen states was selected, all normalized errors, every quality or precision failure, and the unchanged broad claim boundaries. A negative result is retained as the scientific result.

- [ ] **Step 5: Commit Task 5**

```bash
git add data/renewal_cage_ka_replicates_T045_anchor_semi_markov_*.csv data/renewal_cage_ka_replicates_T058_anchor_semi_markov_*.csv data/renewal_cage_ka_anchor_semi_markov_gate.csv figures/renewal_cage_ka_anchor_semi_markov_gate.svg
git commit -m "test anchor semi markov closure"
```

---

### Task 6: Package, Visual, and SOTA Audit

**Files:**
- Modify: `tests/test_arxiv_package.py`
- Modify: `tests/test_ka_anchor_semi_markov.py`
- Modify if required by package inclusion only: `scripts/build_arxiv_package.py`

**Interfaces:**
- Package test recomputes the combined verdict from committed source artifacts.
- Final audit maps the result to the four primary sources named in the design without extending their scope.

- [ ] **Step 1: Write the failing package artifact test**

Require every expected table and SVG, exact row grids and seeds, numerical recomputation of the selector, no held-out calibration use, and zero broad claim flags.

- [ ] **Step 2: Run the package test and confirm RED**

```bash
python3 -m unittest tests.test_arxiv_package.ArxivPackageTests.test_anchor_semi_markov_gate_is_recomputed_and_claim_limited -v
```

Expected: failure until package integration is complete.

- [ ] **Step 3: Add only required package integration**

Update package inputs if the builder uses an explicit allowlist. Do not add manuscript claims or unrelated documentation.

- [ ] **Step 4: Run full verification**

```bash
python3 -m unittest discover -s tests -q
python3 scripts/build_arxiv_package.py
git diff --check
```

Expected: all tests pass, source zip is produced, and no whitespace errors remain.

- [ ] **Step 5: Render and inspect the SVG**

Rasterize the SVG at native aspect ratio and verify title, legends, normalized errors, verdict, and the two-line claim-boundary note are readable and unclipped.

- [ ] **Step 6: Complete the primary-source boundary audit**

Compare the selected result with the correlated-jump, reversible-return, rare-cage-escape, and wave-number-resolved alpha/beta observations listed in the design. State consistency, mismatch, and non-tested scope separately.

- [ ] **Step 7: Commit Task 6**

```bash
git add tests/test_arxiv_package.py tests/test_ka_anchor_semi_markov.py scripts/build_arxiv_package.py
git commit -m "package anchor semi markov gate"
```

## Completion Evidence

The phase is complete only when the real-data selector has one frozen state,
all input provenance can be recomputed from committed artifacts, the full test
suite and arXiv build pass, the SVG is visually checked, and the SOTA comparison
keeps experiment-specific and thermodynamic exclusions explicit. This phase does
not complete the broader glass-transition objective.
