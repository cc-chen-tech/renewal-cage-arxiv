# Segment-Splice Memory Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the frozen paired segment-splice nulls and use the real `T=0.45` and `T=0.58` Kob-Andersen trajectories to decide whether finite ordered path memory or persistent particle/environment identity controls the remaining closure gap.

**Architecture:** A focused `ka_segment_splice` module owns segment provenance, constrained permutations, exact information audits, and efficient multi-lag observables. One analysis script loads each real trajectory once, evaluates the fixed two-temperature length grids with nested deterministic realizations, and writes replicate-first results. A separate summarizer recomputes monotone memory lengths and the final mechanism verdict before producing the deterministic figure.

**Tech Stack:** Python 3.12, NumPy, standard-library `argparse/csv/json/unittest`, existing `ka_replicates` trajectory adapters and observable semantics, deterministic SVG.

## Global Constraints

- Work only in `/Users/luicy/AI/renewal-cage-arxiv/.worktrees/segment-splice-gate` on `codex/segment-splice-gate`.
- Use type-A paths with block size `20`, `B=250` at `T=0.45`, `B=37` at `T=0.58`, and `k={2,4,7.25}`.
- Use exactly `L={1,2,5,10,25,50,125,250}` at `T=0.45` and `L={1,2,4,8,16,32,37}` at `T=0.58`.
- Use three independent low-temperature replicates and five independent high-temperature replicates.
- Start with 16 deterministic realizations per randomized cell. If any cell fails precision, rerun every cell with the nested 64-realization grid.
- Frozen curve tolerances are relative MSD `0.10`, absolute NGP `0.30`, and maximum multi-`k` scattering `0.03`.
- Frozen Monte Carlo limits are relative MSD `0.01`, NGP `0.03`, and scattering `0.003`.
- Held-out paths are target-only. Every output keeps `heldout_path_used_in_prediction=0` and `macro_fit_parameter_count=0`.
- Keep `microdynamic_closure_claim_allowed=0`, `spatial_facilitation_claim_allowed=0`, and `thermodynamic_claim_allowed=0` in every output and final verdict.
- The full-path row `L=B` is an implementation upper bound and cannot be selected as a finite memory length.

---

### Task 1: Segment Provenance and Within-Particle Shuffle

**Files:**
- Create: `src/ka_segment_splice.py`
- Create: `tests/test_ka_segment_splice.py`

**Interfaces:**
- Produces `SegmentSurrogate(blocks, source_particle, source_segment, target_segment_lengths)` as an immutable dataclass.
- Produces `segment_slices(block_count: int, segment_length: int) -> tuple[tuple[int, int], ...]`.
- Produces `within_particle_segment_shuffle(blocks: np.ndarray, *, segment_length: int, rng: np.random.Generator) -> SegmentSurrogate`.

- [ ] **Step 1: Write failing segmentation tests**

Use a `3 x 7 x 3` array whose values encode `(particle, block, component)`. Assert that `segment_slices(7,3)` returns `((0,3),(3,6),(6,7))`; invalid dimensions, nonfinite blocks, Boolean lengths, zero lengths, and `L>B` raise `ValueError`.

- [ ] **Step 2: Run RED**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.SegmentRepresentationTests -v
```

Expected: import failure because `ka_segment_splice` does not exist.

- [ ] **Step 3: Implement immutable provenance**

Define:

```python
@dataclass(frozen=True)
class SegmentSurrogate:
    blocks: np.ndarray
    source_particle: np.ndarray
    source_segment: np.ndarray
    target_segment_lengths: np.ndarray
```

Validate finite `particle x block x 3` input. Build deterministic source slices, choose one nonidentity segment-order permutation shared across particles when more than one segment exists, concatenate exact source slices, and record source particle and source segment for every target segment.

- [ ] **Step 4: Write and pass within-particle information tests**

Assert exact output shape, exact per-particle block-vector multiset, exact token-internal order, deterministic fixed seeds, nonidentity order for `L<B`, and identity at `L=B`.

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.SegmentRepresentationTests tests.test_ka_segment_splice.WithinParticleShuffleTests -v
```

- [ ] **Step 5: Commit Task 1**

```bash
git add src/ka_segment_splice.py tests/test_ka_segment_splice.py
git commit -m "implement within particle segment shuffle"
```

---

### Task 2: Cross-Particle Splice and Exact Audit

**Files:**
- Modify: `src/ka_segment_splice.py`
- Modify: `tests/test_ka_segment_splice.py`

**Interfaces:**
- Produces `cross_particle_segment_splice(blocks, *, segment_length, rng, maximum_restarts=100) -> SegmentSurrogate`.
- Produces `audit_segment_surrogate(source_blocks, surrogate, *, segment_length, model) -> dict[str, float]`.

- [ ] **Step 1: Write failing constrained-assignment tests**

For at least five particles, assert every segment-column source-particle array is a permutation, every source differs from its target, adjacent target segments have different source particles, and every `(source_particle, source_segment)` token occurs once. Assert the paired within and cross models have the same target segment-length order for generators initialized with the same segment-order seed.

- [ ] **Step 2: Run RED**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.CrossParticleSpliceTests -v
```

Expected: failure because the cross splice and audit functions are absent.

- [ ] **Step 3: Implement constrained particle permutations**

For each target segment slot, draw a permutation of particle ids and accept it only when:

```python
np.all(source_particle != np.arange(particle_count))
and (previous_source is None or np.all(source_particle != previous_source))
```

Reject after exactly `maximum_restarts` attempts. Use the same source segment ordinal for all particles in one slot, so each source token is consumed exactly once. Reconstruct target paths from exact source slices without padding, wrapping, truncation, reversal, interpolation, or rotation.

- [ ] **Step 4: Implement exact provenance audit**

Reconstruct source block ids from segment provenance and require their flattened multiset to equal `range(particle_count * block_count)`. Report token reuse min/max, per-particle multiset preservation, same-source assignment, adjacent same-source fraction, analytic internal-adjacency fraction, accidental seam adjacency, complete-path equality, and all claim/provenance flags.

- [ ] **Step 5: Pass adversarial tests**

Cover variable terminal lengths, duplicate-valued vectors, an impossible two-particle multi-segment assignment, tampered token reuse, tampered block values, and the `L=B` whole-path permutation control.

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.CrossParticleSpliceTests tests.test_ka_segment_splice.SegmentAuditTests -v
```

- [ ] **Step 6: Commit Task 2**

```bash
git add src/ka_segment_splice.py tests/test_ka_segment_splice.py
git commit -m "implement exact cross particle segment splice"
```

---

### Task 3: Efficient Multi-Lag Observables and Cell Classification

**Files:**
- Modify: `src/ka_segment_splice.py`
- Create: `scripts/analyze_ka_segment_splice_gate.py`
- Modify: `tests/test_ka_segment_splice.py`

**Interfaces:**
- Produces `cumulative_observables_many_lags(blocks, *, block_counts, wave_numbers) -> dict[int, dict[str, float]]`.
- Produces `summarize_segment_rows(rows, *, fs_keys) -> list[dict[str, object]]`.
- Produces `classify_segment_cells(summary_rows, quality_rows, stationarity_rows, *, expected_grids, expected_replicates, expected_realizations) -> list[dict[str, object]]`.

- [ ] **Step 1: Write failing observable parity tests**

For random `4 x 11 x 3` paths and block counts `{1,2,5,11}`, compare every MSD, fourth moment, NGP, particle-window count, and characteristic function with `ka_replicates.cumulative_block_observables` to `1e-12`.

- [ ] **Step 2: Run RED, implement one-prefix multi-lag evaluation, and pass parity**

Build one cumulative prefix array per surrogate, then derive all requested lag windows without recomputing the prefix. Preserve the existing characteristic-function semantics exactly.

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.MultiLagObservableTests -v
```

- [ ] **Step 3: Write failing classifier truth-table tests**

Construct cells that independently fail exact token accounting, same-source exclusion, stationarity, realization completeness, MSD precision, NGP precision, scattering precision, MSD curve, NGP curve, scattering curve, held-out exclusion, macro-fit count, and each claim flag. Require `cell_pass=1` only when all required gates pass.

- [ ] **Step 4: Implement replicate-first summary and cell classifier**

Average realizations within each replicate, compute realization standard errors, then average replicates equally. Recompute every pass flag from raw numeric columns. Require the exact temperature grids, exact replicate ids, nested realization ids `0..15` or `0..63`, and all three committed stationarity comparisons.

- [ ] **Step 5: Run Task 3 tests and regress existing observable semantics**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.MultiLagObservableTests tests.test_ka_segment_splice.SegmentCellClassifierTests tests.test_ka_replicates.KAReplicatePreparationTests.test_cumulative_block_observables_match_direct_windows -v
```

- [ ] **Step 6: Commit Task 3**

```bash
git add src/ka_segment_splice.py scripts/analyze_ka_segment_splice_gate.py tests/test_ka_segment_splice.py
git commit -m "score segment splice memory cells"
```

---

### Task 4: Two-Temperature Real-Protocol Runner

**Files:**
- Modify: `scripts/analyze_ka_segment_splice_gate.py`
- Modify: `tests/test_ka_segment_splice.py`

**Interfaces:**
- CLI consumes both ensemble directories, both held-out factorization tables, both frozen stationarity tables, block size, initial and extended realization counts, base seed, and output directory.
- Writes `renewal_cage_ka_replicates_T{045,058}_segment_splice_{quality,rows,summary,cells,replicate_scores}.csv`.

- [ ] **Step 1: Write failing seed, CLI, and provenance tests**

Require deterministic unique seeds keyed by temperature, replicate, `L`, realization, and model. Reject any nonfrozen block size, length grid, calibration horizon, replicate set, realization counts other than `16/64`, missing held-out lag, missing stationarity comparison, or held-out input used as a source path.

- [ ] **Step 2: Run RED**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.SegmentRunnerTests -v
```

- [ ] **Step 3: Implement bounded-memory trajectory loading**

For each replicate, load only `calibration_time + 1` frames from `trajectory.lammpstrj`, select type-A particles, construct complete block paths, then release the trajectory array. Do not write or modify files below `/Users/luicy/AI/renewal-cage-arxiv/tmp/ka_replicates`.

- [ ] **Step 4: Implement paired realizations and global precision escalation**

Evaluate all cells with realizations `0..15`. If any randomized cell violates a precision limit, rerun every cell at both temperatures with realizations `0..63`; otherwise keep 16. The first 16 seeds and generated surrogates must be identical in both runs. Never extend one cell selectively.

- [ ] **Step 5: Implement deterministic CSV serialization**

Write raw quality rows, replicate/lag prediction rows, replicate-first summaries, cell verdicts, and paired replicate higher-order scores. Record `tau_L=20*L`, full-path-control status, stationarity provenance, exact grids, seed metadata, source paths, and all claim flags.

- [ ] **Step 6: Pass synthetic end-to-end runner tests**

Use a temporary two-temperature fixture with tiny LAMMPS trajectories and held-out tables. Assert exact output schema, 16-to-64 global escalation, nested first-16 equality, deterministic reruns, and no source-directory changes.

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.SegmentRunnerTests -v
```

- [ ] **Step 7: Commit Task 4**

```bash
git add scripts/analyze_ka_segment_splice_gate.py tests/test_ka_segment_splice.py
git commit -m "run frozen two temperature segment gate"
```

---

### Task 5: Monotone Memory Selection and Real Data

**Files:**
- Create: `scripts/summarize_ka_segment_splice_gate.py`
- Modify: `tests/test_ka_segment_splice.py`
- Create: `data/renewal_cage_ka_replicates_T045_segment_splice_*.csv`
- Create: `data/renewal_cage_ka_replicates_T058_segment_splice_*.csv`
- Create: `data/renewal_cage_ka_segment_splice_gate.csv`
- Create: `figures/renewal_cage_ka_segment_splice_gate.svg`

**Interfaces:**
- Produces `select_monotone_memory_length(cell_rows, *, model, temperature, block_count, required_grid) -> int | None`.
- Produces `classify_segment_splice_gate(low_cells, high_cells, low_replicate_scores, high_replicate_scores) -> dict[str, object]`.

- [ ] **Step 1: Write failing monotone-selection tests**

Cover an isolated pass followed by failure, pass only at `L=B`, valid monotone tails, missing grid values, duplicate cells, unresolved precision, and full-path controls incorrectly marked selectable.

- [ ] **Step 2: Write failing decision-table tests**

Cover exactly: `finite_single_particle_path_memory_sufficient_conditional_on_global_schedule`, `persistent_environment_identity_required_beyond_local_path`, `longer_or_richer_path_state_required`, `null_family_pathology_unresolved`, and `mechanism_unresolved`. Require all three low-temperature replicate scores to pass; require strict within-better ordering for the persistent-environment verdict; recompute the cooling crossover label. Preserve a complete `low_temperature_mechanism_state` when only the high-temperature crossover support fails, while keeping the global preregistered mechanism state fail-closed.

- [ ] **Step 3: Implement selector and deterministic SVG**

Plot normalized MSD, NGP, and maximum scattering errors against `tau_L` for both models and temperatures. Mark tolerance one, selected horizons, full-path controls, precision failures, and the final claim boundary without `nan`, `inf`, overlap, or clipped text.

- [ ] **Step 4: Run selector tests**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice.SegmentMemorySelectionTests tests.test_ka_segment_splice.SegmentGateDecisionTests -v
```

- [ ] **Step 5: Run the frozen real protocol**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python scripts/analyze_ka_segment_splice_gate.py \
  --low-ensemble-directory /Users/luicy/AI/renewal-cage-arxiv/tmp/ka_replicates/T045 \
  --high-ensemble-directory /Users/luicy/AI/renewal-cage-arxiv/tmp/ka_replicates/T058 \
  --low-heldout-factorization data/renewal_cage_ka_replicates_T045_event_oracle_factorization_rows.csv \
  --high-heldout-factorization data/renewal_cage_ka_replicates_T058_event_oracle_factorization_rows.csv \
  --low-stationarity data/renewal_cage_ka_replicates_T045_nonlinear_path_stationarity.csv \
  --high-stationarity data/renewal_cage_ka_replicates_T058_block20_nonlinear_path_stationarity.csv \
  --block-size 20 --initial-realizations 16 --extended-realizations 64 \
  --base-seed 20260718 --output-directory data

/tmp/renewal-cage-py312-segment-splice/bin/python scripts/summarize_ka_segment_splice_gate.py \
  --low-cells data/renewal_cage_ka_replicates_T045_segment_splice_cells.csv \
  --high-cells data/renewal_cage_ka_replicates_T058_segment_splice_cells.csv \
  --low-replicate-scores data/renewal_cage_ka_replicates_T045_segment_splice_replicate_scores.csv \
  --high-replicate-scores data/renewal_cage_ka_replicates_T058_segment_splice_replicate_scores.csv \
  --output data/renewal_cage_ka_segment_splice_gate.csv \
  --output-svg figures/renewal_cage_ka_segment_splice_gate.svg
```

- [ ] **Step 6: Inspect real verdict and figure before writing prose**

Verify every exactness, stationarity, precision, and provenance gate from source CSVs. Inspect the SVG for finite coordinates and text containment. State the actual verdict even when it rejects both candidate mechanisms.

- [ ] **Step 7: Commit Task 5**

```bash
git add scripts/summarize_ka_segment_splice_gate.py tests/test_ka_segment_splice.py data/renewal_cage_ka_replicates_T045_segment_splice_*.csv data/renewal_cage_ka_replicates_T058_segment_splice_*.csv data/renewal_cage_ka_segment_splice_gate.csv figures/renewal_cage_ka_segment_splice_gate.svg
git commit -m "test path memory against environment identity"
```

---

### Task 6: Artifact Gate, Scientific Note, and PR Completion

**Files:**
- Modify: `tests/test_arxiv_package.py`
- Create: `docs/segment-splice-memory-gate.md`
- Modify: `README.md`

- [ ] **Step 1: Write failing artifact recomputation test**

Require every planned file, exact two-temperature grid, exact realization count, monotone-tail recomputation, source verdict consistency, all three claim flags at zero, finite SVG coordinates, and explicit language separating persistent environment identity from proven spatial facilitation.

- [ ] **Step 2: Run RED, then write the minimum result note and README index entry**

The note must report protocol, exact real verdict, selected or unresolved horizons, normalized observable errors, null failures, and the next microscopic implication. It must not promote a path-information result to an autonomous single-particle Langevin closure.

- [ ] **Step 3: Pass focused artifact tests**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest tests.test_ka_segment_splice tests.test_arxiv_package.ArxivPackageTests.test_segment_splice_memory_gate_is_recomputed_and_claim_limited -v
```

- [ ] **Step 4: Run complete verification**

```bash
/tmp/renewal-cage-py312-segment-splice/bin/python -m unittest discover -s tests
/tmp/renewal-cage-py312-segment-splice/bin/python scripts/generate_renewal_cage_results.py
/tmp/renewal-cage-py312-segment-splice/bin/python scripts/build_arxiv_package.py
git diff --check
git status --short
```

Expected: all tests pass, generators and package builder exit zero, no whitespace errors, and only intended files differ.

- [ ] **Step 5: Commit and update PR #9**

```bash
git add tests/test_arxiv_package.py docs/segment-splice-memory-gate.md README.md
git commit -m "document segment splice memory verdict"
git push origin codex/segment-splice-gate
gh pr ready 9
```

Do not merge automatically. Report the exact real verdict, tests, PR URL, and remaining microscopic gap.
