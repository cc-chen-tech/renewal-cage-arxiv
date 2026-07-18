# Segment-Splice Paired-Excess Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompute finite-horizon losses relative to each replicate's full-path baseline and expose only fail-closed exploratory constraints for later microscopic dynamics.

**Architecture:** One focused summarizer validates the committed T=0.45 score, cell, and verdict tables; computes additive paired excess; and writes row, gate, and deterministic SVG artifacts. Unit tests own formulas and failure semantics, while an arXiv test recomputes all committed results from source CSVs.

**Tech Stack:** Python 3.12 standard library, CSV, deterministic SVG, `unittest`.

## Global Constraints

- Treat this as post-run exploratory analysis, not a preregistered mechanism gate.
- Do not alter source trajectories, scores, grids, realizations, precision limits, or curve tolerances.
- Require exact models, replicates `1,2,3`, and lengths `1,2,5,10,25,50,125,250`.
- Require full-path model agreement within `1e-12`.
- Use additive excess and t95 intervals with `df=2`, `t=4.302652729911275`.
- An interval containing zero is unresolved, never equivalent.
- Keep every microscopic, owner-sufficiency, finite-memory, spatial, and thermodynamic claim flag at zero.

---

### Task 1: Exact Paired-Excess Kernel

**Files:**
- Create: `scripts/summarize_ka_segment_splice_paired_excess.py`
- Modify: `tests/test_ka_segment_splice.py`

**Interfaces:**
- Produce `compute_paired_excess_rows(score_rows, *, block_size=20) -> list[dict[str, object]]`.
- Return 14 finite-length rows with three baselines, three excesses, mean, SE, t95 bounds, and degradation flag.

- [ ] **Step 1: Write a failing formula test**

```python
result = analysis.compute_paired_excess_rows(paired_excess_score_fixture(), block_size=20)
row = next(item for item in result if item["model"] == "within_particle_segment_shuffle" and item["segment_length"] == 1.0)
self.assertEqual(row["replicate_1_excess"], 2.0)
self.assertAlmostEqual(row["mean_paired_excess"], 3.0)
self.assertEqual(row["paired_degradation_identified"], 1.0)
```

- [ ] **Step 2: Run RED**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.SegmentPairedExcessTests.test_additive_excess_and_t_interval_are_exact -v
```

Expected: missing module or function.

- [ ] **Step 3: Implement exact indexing and statistics**

Validate finite values, exact support, unique rows, zero source claim flags, and full-path agreement. Compute additive excess, sample SE, and t95 interval without NumPy. Exclude the full length from output.

- [ ] **Step 4: Add malformed-input RED tests and pass them**

Missing rows, duplicate rows, nonfinite scores, nonzero claim flags, and full-path disagreement must each raise `ValueError`.

- [ ] **Step 5: Verify and commit**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.SegmentPairedExcessTests -v
git add scripts/summarize_ka_segment_splice_paired_excess.py tests/test_ka_segment_splice.py
git commit -m "compute replicate centered segment excess"
```

---

### Task 2: Fail-Closed Gate and Owner Contrast

**Files:**
- Modify: `scripts/summarize_ka_segment_splice_paired_excess.py`
- Modify: `tests/test_ka_segment_splice.py`

**Interfaces:**
- Produce `classify_paired_excess_gate(score_rows, cell_rows, source_gate) -> dict[str, object]`.
- Recompute all statistics instead of trusting stored exploratory fields.

- [ ] **Step 1: Write classifier truth-table RED tests**

Require complete paired input and full-path agreement, but keep
`mechanism_state=mechanism_unresolved`,
`paired_excess_equivalence_claim_allowed=0`, and
`finite_memory_state_addition_allowed=0`. A contiguous degradation prefix may
set `short_horizon_information_loss_supported_exploratory=1`.

- [ ] **Step 2: Run RED**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.SegmentPairedExcessGateTests -v
```

- [ ] **Step 3: Implement gate and owner contrast**

Recompute all 21 cross-minus-within differences. Average seven lengths inside
each replicate, then compute three-replicate mean, SE, and t95 interval. Do not
route owner evidence into mechanism selection. Fail closed on malformed source
gate, missing schedule preservation, or non-prefix degradation.

- [ ] **Step 4: Lock all claim boundaries, verify, and commit**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.SegmentPairedExcessTests tests.test_ka_segment_splice.SegmentPairedExcessGateTests -v
git add scripts/summarize_ka_segment_splice_paired_excess.py tests/test_ka_segment_splice.py
git commit -m "classify paired segment excess fail closed"
```

---

### Task 3: Real CSV and SVG Artifacts

**Files:**
- Modify: `scripts/summarize_ka_segment_splice_paired_excess.py`
- Modify: `tests/test_ka_segment_splice.py`
- Create: `data/renewal_cage_ka_segment_splice_paired_excess_rows.csv`
- Create: `data/renewal_cage_ka_segment_splice_paired_excess_gate.csv`
- Create: `figures/renewal_cage_ka_segment_splice_paired_excess.svg`

**Interfaces:**
- CLI: `--scores`, `--cells`, `--source-gate`, `--output-rows`, `--output-gate`, `--output-svg`.

- [ ] **Step 1: Write CLI/SVG RED tests**

Require deterministic reruns, 14 rows, zero-baseline line, t95 interval legend,
finite coordinates, post-run exploratory label, and global-schedule condition.

- [ ] **Step 2: Run RED**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.SegmentPairedExcessArtifactTests -v
```

- [ ] **Step 3: Implement serialization, CLI, and SVG**

Use log `tau_L`, linear paired excess, vertical t95 intervals, a zero line, and
distinct model colors. Label intervals crossing zero as unresolved and draw no
selected memory horizon.

- [ ] **Step 4: Generate real artifacts**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python scripts/summarize_ka_segment_splice_paired_excess.py --scores data/renewal_cage_ka_replicates_T045_segment_splice_replicate_scores.csv --cells data/renewal_cage_ka_replicates_T045_segment_splice_cells.csv --source-gate data/renewal_cage_ka_segment_splice_gate.csv --output-rows data/renewal_cage_ka_segment_splice_paired_excess_rows.csv --output-gate data/renewal_cage_ka_segment_splice_paired_excess_gate.csv --output-svg figures/renewal_cage_ka_segment_splice_paired_excess.svg
```

- [ ] **Step 5: Render, inspect, verify, and commit**

```bash
sips -s format png figures/renewal_cage_ka_segment_splice_paired_excess.svg --out /tmp/paired-excess.png
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.SegmentPairedExcessArtifactTests -v
git add scripts/summarize_ka_segment_splice_paired_excess.py tests/test_ka_segment_splice.py data/renewal_cage_ka_segment_splice_paired_excess_rows.csv data/renewal_cage_ka_segment_splice_paired_excess_gate.csv figures/renewal_cage_ka_segment_splice_paired_excess.svg
git commit -m "measure paired short horizon information loss"
```

---

### Task 4: Scientific Note and Artifact Recompute

**Files:**
- Create: `docs/segment-splice-paired-excess.md`
- Modify: `README.md`
- Modify: `tests/test_arxiv_package.py`

**Interfaces:**
- Artifact test imports both public functions and recomputes all committed fields from source tables.

- [ ] **Step 1: Write arXiv artifact RED test**

Require exact paths, 14-row grid, real degradation prefix, baseline failure,
zero claim flags, SVG labels, and result-note language that unresolved is not
equivalence and no finite-memory Langevin state may be added.

- [ ] **Step 2: Run RED**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_arxiv_package.ArxivPackageTests.test_segment_splice_paired_excess_is_recomputed_and_claim_limited -v
```

- [ ] **Step 3: Write note and README entry**

Report exact means/intervals, identified short-horizon prefix, unresolved longer
lengths, baseline failures, owner contrast, global-schedule condition, and next
action. Do not call the result a memory length or microscopic closure.

- [ ] **Step 4: Run publication verification**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest discover -s tests
/tmp/renewal-cage-py312-segment-splice/bin/python scripts/generate_renewal_cage_results.py
/tmp/renewal-cage-py312-segment-splice/bin/python scripts/build_arxiv_package.py
git diff --check
```

- [ ] **Step 5: Commit, push, and create PR**

```bash
git add README.md docs/segment-splice-paired-excess.md tests/test_arxiv_package.py
git commit -m "document paired segment excess boundary"
git push -u origin codex/paired-excess-baseline-gate
gh pr create --base main --head codex/paired-excess-baseline-gate --title "Measure paired segment excess over replicate baselines" --body "Post-run exploratory paired-excess diagnostic with fail-closed microscopic claims."
```
