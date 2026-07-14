# Same-Label Soft-Mode Precursor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test a fixed six-dimensional instantaneous local Hessian state against the current five-parent smooth-center escape labels.

**Architecture:** Add one shared descriptor that wraps the exact local KA Hessian, expose the already validated parent microstate from the radial cache loader, and run the existing survival/propensity diagnostics in a compact same-label analysis script. Preserve the geometry baseline and all stronger claim boundaries.

**Tech Stack:** Python 3, NumPy eigendecomposition, existing KA Hessian and grouped diagnostics, CSV/NPZ, `unittest`.

## Global Constraints

- Keep the current 5 parents, 8 clones, 64 targets, 1731 events, 829 censors, `T=0.58`, `p_hop=0.08`, half-window 8, horizon `20 tau`, and `L2=1` unchanged.
- Use cluster cutoff `1.5`, ranks `(0,3)`, and eigenvalue floor `1e-6` without tuning.
- Require clone invariance and geometry reproduction within `1e-12`.
- Require combined Brier improvement `>=0.01`, absolute Brier reference `>0.026964`, nonnegative held-parent likelihood gains, survival error `<=0.10`, and positive improved binomial skill.
- Keep `event_clock_claim_allowed`, `autonomous_single_particle_gle_claim_allowed`, `kramers_escape_claim_allowed`, and `thermodynamic_claim_allowed` false.
- Do not stage `tmp/` or unrelated untracked artifacts.

---

### Task 1: Shared Instantaneous Local Soft-Mode State

**Files:**
- Modify: `src/ka_structural_precursor.py`
- Modify: `tests/test_ka_structural_precursor.py`

**Interfaces:**
- Consumes: `ka_local_cluster_hessian(...)` from `ka_local_cage`.
- Produces: `instantaneous_local_soft_mode_features(positions, particle_types, box_lengths, target_indices, *, cluster_cutoff=1.5, ranks=(0,3), eigenvalue_floor=1e-6) -> tuple[np.ndarray, tuple[str,...]]`.

- [ ] **Step 1: Write a failing descriptor equivalence test**

For one target, independently diagonalize `ka_local_cluster_hessian`, calculate
the two log inverse eigenvalues, two log target-weighted softnesses, log cluster
count, and nonpositive-mode count, then compare to the new descriptor at
`1e-12`.

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_ka_structural_precursor.StructuralPrecursorTests.test_local_soft_mode_features_match_exact_cluster_hessian -v
```

Expected: import failure for `instantaneous_local_soft_mode_features`.

- [ ] **Step 3: Implement and validate the descriptor**

Use the sorted positive eigenmodes above `eigenvalue_floor`, require enough
positive modes for every requested rank, locate the target inside the cluster,
and return the six fixed finite features in rank-major order followed by
cluster size and nonpositive-mode count.

- [ ] **Step 4: Run focused tests and commit**

```bash
python -m unittest tests.test_ka_structural_precursor tests.test_ka_replicates.KAReplicatePreparationTests.test_local_cluster_soft_mode_features_are_target_symmetric_for_a_pair -v
python -m py_compile src/ka_structural_precursor.py
git add src/ka_structural_precursor.py tests/test_ka_structural_precursor.py
git commit -m "add instantaneous local softmode precursor state"
```

### Task 2: Compact Same-Label Held-Parent Analysis

**Files:**
- Modify: `scripts/analyze_ka_radial_precursor_residual.py`
- Create: `scripts/analyze_ka_softmode_precursor_residual.py`
- Modify: `tests/test_ka_structural_precursor.py`

**Interfaces:**
- Extends: `load_parent_state_and_labels(...)` to return the validated common `positions`, `particle_types`, and `box_lengths` without changing radial outputs.
- Produces: soft-mode `_models.csv`, `_committor.csv`, and `_summary.csv`.

- [ ] **Step 1: Write failing protocol/gate tests**

Require exact soft-mode controls, exact counts, models `geometry`, `softmode`,
and `geometry_softmode`, both grouped diagnostics, geometry reproduction, and
false event-clock/GLE/Kramers/thermodynamic claims. Test
`evaluate_softmode_precursor_gates` with passing synthetic metrics and a
negative held-parent likelihood counterexample.

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_ka_structural_precursor.StructuralPrecursorTests.test_softmode_residual_analysis_freezes_same_label_protocol -v
```

Expected: failure because the script is absent.

- [ ] **Step 3: Expose the validated parent microstate**

Add `positions`, `particle_types`, and `box_lengths` to the return dictionary of
`load_parent_state_and_labels`. Rerun the radial script into a temporary prefix
and compare its summary byte-for-byte to the committed summary.

- [ ] **Step 4: Implement the compact analysis**

Import the radial loader and fixed geometry constants. Compute one six-feature
soft-mode state per parent-target, create the three model tensors, expand each
over clones, and call `grouped_exponential_escape_diagnostic` and
`grouped_binomial_logistic_committor_diagnostic`. Write model summaries and all
15 held-parent rows, 320 committor rows, and one gate summary with LF endings.

- [ ] **Step 5: Run focused tests and commit**

```bash
python -m unittest tests.test_ka_structural_precursor -v
python -m py_compile scripts/analyze_ka_radial_precursor_residual.py scripts/analyze_ka_softmode_precursor_residual.py
git add scripts/analyze_ka_radial_precursor_residual.py scripts/analyze_ka_softmode_precursor_residual.py tests/test_ka_structural_precursor.py
git commit -m "add same-label softmode precursor analysis"
```

### Task 3: Execute and Freeze the Five-Parent Test

**Files:**
- Create: `data/renewal_cage_ka_softmode_precursor_T058_models.csv`
- Create: `data/renewal_cage_ka_softmode_precursor_T058_committor.csv`
- Create: `data/renewal_cage_ka_softmode_precursor_T058_summary.csv`

- [ ] **Step 1: Run the fixed analysis**

```bash
python scripts/analyze_ka_softmode_precursor_residual.py \
  --initial-state-directories \
    tmp/smooth_cage_event_clock_initial_parent00 \
    tmp/smooth_cage_event_clock_initial_parent01 \
    tmp/smooth_cage_event_clock_initial_parent02 \
    tmp/smooth_cage_event_clock_initial_parent03 \
    tmp/smooth_cage_event_clock_initial_parent04 \
  --cache-directory tmp/smooth_cage_event_clock_labels_T058 \
  --output-prefix data/renewal_cage_ka_softmode_precursor_T058
```

Expected: 2560 survival rows, 320 propensity rows, 1731/829 events/censors,
and a printed fixed verdict.

- [ ] **Step 2: Audit without retuning**

Inspect all held-parent likelihood and Brier rows. Confirm the geometry metrics
exactly match the prior checkpoint. Do not change cutoff, ranks, floor,
regularization, labels, or targets.

- [ ] **Step 3: Commit artifacts**

```bash
git add data/renewal_cage_ka_softmode_precursor_T058_models.csv \
  data/renewal_cage_ka_softmode_precursor_T058_committor.csv \
  data/renewal_cage_ka_softmode_precursor_T058_summary.csv
git commit -m "run same-label softmode precursor test"
```

### Task 4: Verdict and Dynamic-Memory Routing

**Files:**
- Create: `docs/microscopic-softmode-precursor-residual.md`
- Modify: `tests/test_arxiv_package.py`

- [ ] **Step 1: Write a failing package gate from the actual result**

Require exact integrity fields, model comparisons, actual verdict, all false
stronger claims, and a statement that failure ends scalar instantaneous local
descriptor searches.

- [ ] **Step 2: Run RED**

```bash
python -m unittest tests.test_arxiv_package.ArxivPackageTests.test_softmode_precursor_residual_gate_is_complete -v
```

Expected: failure because the scientific note is absent.

- [ ] **Step 3: Write the calibrated scientific note**

Explain the Hessian construction, distinguish this label set from earlier
soft-mode tests, report every model and held-parent result, and route a failure
to a nonlinear/state-dependent Mori-Zwanzig memory experiment.

- [ ] **Step 4: Run full verification and commit**

```bash
python -m unittest discover -s tests
python -m py_compile src/ka_structural_precursor.py \
  scripts/analyze_ka_radial_precursor_residual.py \
  scripts/analyze_ka_softmode_precursor_residual.py
git diff --check
git add docs/microscopic-softmode-precursor-residual.md tests/test_arxiv_package.py
git commit -m "document same-label softmode precursor verdict"
```
