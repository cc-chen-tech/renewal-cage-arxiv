# Radial Microscopic Precursor Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a fixed species-resolved microscopic packing state explains held-parent smooth-cage escape information that the exact projected geometry `(u,G)` misses.

**Architecture:** Add one shared structural-descriptor module and one geometry-only helper, then build a focused analysis script that consumes the already frozen frame-zero states and first-passage caches. Score the same data both as clone-level censored survival and as parent-target binomial propensity, write machine-readable gates, and preserve all stronger Langevin, Kramers, event-clock, and thermodynamic claims as false.

**Tech Stack:** Python 3 standard library, NumPy, existing `ka_smooth_cage` and `ka_local_cage` diagnostics, `unittest`, CSV/NPZ artifacts.

## Global Constraints

- Use exactly five independent `T=0.58` parents, the first eight clones, the existing 64 A targets selected with seed `20260714`, and the cached 1731 event / 829 censor labels.
- Keep `p_hop=0.08`, half-window `8`, horizon `20 tau`, `L2=1`, and leave-one-complete-parent-out validation unchanged.
- Use radial centers `[0.8, 1.05, 1.3, 1.55, 1.8, 2.05, 2.3]`, width `0.12`, and cutoff `2.5`; do not tune them after seeing results.
- Keep the structural difficulty reference fixed at held-parent Brier skill `0.026964`.
- Require clone-invariant structural features within `1e-12` and exact reproduction of the frozen geometry metrics within `1e-12`.
- Keep `event_clock_claim_allowed`, `autonomous_single_particle_gle_claim_allowed`, `kramers_escape_claim_allowed`, and `thermodynamic_claim_allowed` false regardless of this experiment's result.
- Stage only named source, test, data, and documentation files. Do not add `tmp/` caches or other untracked research artifacts.

---

### Task 1: Shared Microscopic Structural Coordinates

**Files:**
- Create: `src/ka_structural_precursor.py`
- Modify: `src/ka_smooth_cage.py`
- Modify: `scripts/analyze_ka_committor_radial_structure.py`
- Create: `tests/test_ka_structural_precursor.py`
- Modify: `tests/test_ka_smooth_cage.py`

**Interfaces:**
- Produces: `species_resolved_radial_features(positions, particle_types, box_lengths, target_indices, *, radii, width, cutoff, block_size=128) -> np.ndarray` with shape `(targets, 2*len(radii))`.
- Produces: `expand_isoconfigurational_structural_rows(features, first_passage, escaped) -> dict[str, np.ndarray]`, where features have shape `(parents, targets, dimensions)` and labels have shape `(parents, clones, targets)`.
- Produces: `smooth_cage_geometry_features(positions, *, particle_types, box_lengths, target_indices) -> np.ndarray` with columns `log|u|^2` and sorted `log eig(JJ^T)`.
- Consumes: existing `smooth_force_support_cage` and existing grouped diagnostics.

- [ ] **Step 1: Write failing tests for the radial descriptor**

Add tests that manually evaluate a small periodic A/B configuration, verify
the `(targets, 2*radii)` shape, verify A and B channels separately, and verify
that translating every particle by the same vector leaves the result fixed.

```python
feature = species_resolved_radial_features(
    positions,
    particle_types,
    np.array([10.0, 10.0, 10.0]),
    np.array([0]),
    radii=np.array([1.0]),
    width=0.2,
    cutoff=2.5,
)
self.assertEqual(feature.shape, (1, 2))
self.assertTrue(np.allclose(feature, translated_feature, atol=1e-14))
self.assertGreater(feature[0, 0], 0.0)
self.assertGreater(feature[0, 1], 0.0)
```

- [ ] **Step 2: Run the radial tests and verify RED**

Run:

```bash
python -m unittest tests.test_ka_structural_precursor -v
```

Expected: import failure because `ka_structural_precursor` does not exist.

- [ ] **Step 3: Implement the radial descriptor with explicit validation**

Implement minimum-image distances, cosine cutoff, self exclusion, two species
channels, and finite/shape/radius validation in `src/ka_structural_precursor.py`.
Move the existing implementation out of
`scripts/analyze_ka_committor_radial_structure.py` and import the shared
function there without changing that script's numerical defaults.

- [ ] **Step 4: Run the radial tests and the prior radial-script tests**

Run:

```bash
python -m unittest tests.test_ka_structural_precursor tests.test_ka_replicates -v
```

Expected: all tests pass and the prior CLI keeps the same feature grid.

- [ ] **Step 5: Write failing tests for geometry-only extraction**

Construct one finite KA state and compare the new geometry-only helper to the
first four entries returned by `smooth_cage_invariant_features` after the full
projected observable calculation.

```python
geometry = smooth_cage_geometry_features(
    positions,
    particle_types=particle_types,
    box_lengths=box_lengths,
    target_indices=np.array([0]),
)
full = smooth_cage_invariant_features(observable)["geometry"]
np.testing.assert_allclose(geometry[0], full, rtol=0.0, atol=1e-12)
```

- [ ] **Step 6: Run the geometry test and verify RED**

Run:

```bash
python -m unittest tests.test_ka_smooth_cage.SmoothCageTests.test_geometry_only_features_match_full_projected_geometry -v
```

Expected: import failure for `smooth_cage_geometry_features`.

- [ ] **Step 7: Implement the geometry-only helper**

For each target, call `smooth_force_support_cage`, set `u` to
`relative_position`, set `G=J J^T`, symmetrize `G`, clip eigenvalues at
`1e-14`, and return

```python
np.concatenate([[np.log(np.dot(u, u) + 1e-14)], np.log(np.linalg.eigvalsh(G).clip(min=1e-14))])
```

- [ ] **Step 8: Run focused tests and commit**

Run:

```bash
python -m unittest tests.test_ka_structural_precursor tests.test_ka_smooth_cage -v
python -m py_compile src/ka_structural_precursor.py src/ka_smooth_cage.py scripts/analyze_ka_committor_radial_structure.py
```

Expected: all focused tests pass and compilation exits zero.

Commit:

```bash
git add src/ka_structural_precursor.py src/ka_smooth_cage.py scripts/analyze_ka_committor_radial_structure.py tests/test_ka_structural_precursor.py tests/test_ka_smooth_cage.py
git commit -m "add microscopic radial precursor coordinates"
```

### Task 2: Isoconfigurational Row and Propensity Assembly

**Files:**
- Modify: `src/ka_structural_precursor.py`
- Modify: `tests/test_ka_structural_precursor.py`

**Interfaces:**
- Consumes: parent-target structural features and clone-resolved first-passage labels.
- Produces: clone-level `features`, `first_passage`, `escaped`, and `groups`, plus aggregated `configuration_features`, `successes`, `trials`, and `configuration_groups`.

- [ ] **Step 1: Write a failing ordering and aggregation test**

Use two parents, three clones, two targets, and distinguishable feature values.
Assert that clone-level rows follow parent, clone, target order and that the
aggregated successes are sums over the clone axis.

```python
result = expand_isoconfigurational_structural_rows(features, first_passage, escaped)
np.testing.assert_array_equal(result["groups"], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
np.testing.assert_array_equal(result["successes"], escaped.sum(axis=1).reshape(-1))
np.testing.assert_array_equal(result["trials"], np.full(4, 3))
```

- [ ] **Step 2: Run the assembly test and verify RED**

Run:

```bash
python -m unittest tests.test_ka_structural_precursor.StructuralPrecursorTests.test_isoconfigurational_expansion_preserves_parent_clone_target_order -v
```

Expected: import failure for `expand_isoconfigurational_structural_rows`.

- [ ] **Step 3: Implement strict shape and censor validation**

Reject nonfinite features, nonpositive or out-of-horizon first-passage times,
misaligned label tensors, fewer than two parents/clones, and escaped labels
whose times are not strictly below or equal to the common finite censor time.
Return only NumPy arrays with deterministic row ordering.

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
python -m unittest tests.test_ka_structural_precursor -v
```

Expected: all tests pass.

Commit:

```bash
git add src/ka_structural_precursor.py tests/test_ka_structural_precursor.py
git commit -m "add isoconfigurational precursor row assembly"
```

### Task 3: Frozen Five-Parent Residual Analysis

**Files:**
- Create: `scripts/analyze_ka_radial_precursor_residual.py`
- Modify: `tests/test_ka_structural_precursor.py`

**Interfaces:**
- Consumes: five `smooth_cage_event_clock_initial_parent*` directories and `smooth_cage_event_clock_labels_T058`.
- Produces: `<prefix>_details.csv`, `_models.csv`, `_survival.csv`, `_committor.csv`, and `_summary.csv`.
- Uses: `grouped_exponential_escape_diagnostic` and `grouped_binomial_logistic_committor_diagnostic` with `L2=1`.

- [ ] **Step 1: Write failing source-contract and synthetic gate tests**

The source-contract test must require all fixed constants, all three primary
models, both diagnostics, exact event/censor checks, geometry reproduction,
clone-invariance, and all false stronger claims. A pure helper
`evaluate_radial_precursor_gates(summary_metrics) -> dict[str,bool]` must be
tested once with passing synthetic metrics and once with a negative held-parent
likelihood gain.

- [ ] **Step 2: Run the analysis tests and verify RED**

Run:

```bash
python -m unittest tests.test_ka_structural_precursor.StructuralPrecursorTests.test_radial_residual_analysis_freezes_protocol_and_claim_boundaries -v
```

Expected: failure because the analysis script does not exist.

- [ ] **Step 3: Implement frozen-cache loading and integrity checks**

Read target indices from the first label cache, require exact equality in all
40 caches, require `threshold=0.08`, `half_window=8`, `horizon=20`, and
`thermodynamic_claim_allowed=0`. Read every initial state, verify matching
types/boxes/positions within each parent, compute geometry and radial features,
and require maximum clone variation no larger than `1e-12`.

- [ ] **Step 4: Implement the two held-parent diagnostics**

For clone-level models, construct `geometry`, `radial`, and
`geometry_radial`, then call the censored exponential diagnostic at fixed
survival times `[1,2,4,8,12,16,20]`. For aggregated models, call the grouped
binomial diagnostic on the 320 parent-target rows. Never select features or
regularization from held-parent scores.

- [ ] **Step 5: Implement exact gates and deterministic CSV output**

Require the prior geometry metrics:

```text
mean held-parent Brier skill                 0.008357453921058
mean log-likelihood gain per observation    0.0064617896947432914
minimum parent likelihood gain              1.2946834014508113
maximum survival error                      0.06742767815935347
```

Set `static_radial_precursor_allowed` true only when all of the following are
true: `geometry_radial` Brier skill is at least `0.01` above both `geometry`
and `radial`; it exceeds `0.026964`; its mean likelihood gain per observation
is positive; its minimum held-parent total likelihood gain is nonnegative;
its maximum survival error is at most `0.10`; and the aggregated
`geometry_radial` binomial Brier skill is positive and exceeds aggregated
`geometry`. Write CSV with
`lineterminator="\n"`, stable row ordering, and `thermodynamic_claim_allowed`
on every output row.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
python -m unittest tests.test_ka_structural_precursor -v
python -m py_compile scripts/analyze_ka_radial_precursor_residual.py
```

Expected: all tests pass and compilation exits zero.

Commit:

```bash
git add scripts/analyze_ka_radial_precursor_residual.py tests/test_ka_structural_precursor.py
git commit -m "add radial precursor residual analysis"
```

### Task 4: Execute the Frozen Experiment

**Files:**
- Create: `data/renewal_cage_ka_radial_precursor_T058_details.csv`
- Create: `data/renewal_cage_ka_radial_precursor_T058_models.csv`
- Create: `data/renewal_cage_ka_radial_precursor_T058_survival.csv`
- Create: `data/renewal_cage_ka_radial_precursor_T058_committor.csv`
- Create: `data/renewal_cage_ka_radial_precursor_T058_summary.csv`

**Interfaces:**
- Consumes: the unchanged 40 reduced initial states and 40 cached label files.
- Produces: the preregistered numerical verdict and sufficient detail to reproduce each gate.

- [ ] **Step 1: Run the five-parent analysis once**

Run:

```bash
python scripts/analyze_ka_radial_precursor_residual.py \
  --initial-state-directories \
    tmp/smooth_cage_event_clock_initial_parent00 \
    tmp/smooth_cage_event_clock_initial_parent01 \
    tmp/smooth_cage_event_clock_initial_parent02 \
    tmp/smooth_cage_event_clock_initial_parent03 \
    tmp/smooth_cage_event_clock_initial_parent04 \
  --cache-directory tmp/smooth_cage_event_clock_labels_T058 \
  --output-prefix data/renewal_cage_ka_radial_precursor_T058
```

Expected: exactly 2560 observations, 1731 events, 829 censors, 320
configuration rows, and a printed JSON verdict.

- [ ] **Step 2: Audit outputs without changing protocol**

Check that geometry reproduces the frozen metrics, inspect every held-parent
row, compare clone-level and aggregated scores, and run:

```bash
git diff --check -- data/renewal_cage_ka_radial_precursor_T058_*.csv
```

Expected: no whitespace errors. Do not modify radii, width, cutoff, targets,
regularization, or event labels in response to the result.

- [ ] **Step 3: Commit the result artifacts**

```bash
git add data/renewal_cage_ka_radial_precursor_T058_details.csv \
  data/renewal_cage_ka_radial_precursor_T058_models.csv \
  data/renewal_cage_ka_radial_precursor_T058_survival.csv \
  data/renewal_cage_ka_radial_precursor_T058_committor.csv \
  data/renewal_cage_ka_radial_precursor_T058_summary.csv
git commit -m "run radial microscopic precursor residual test"
```

### Task 5: Scientific Interpretation and Package Gate

**Files:**
- Create: `docs/microscopic-radial-precursor-residual.md`
- Modify: `tests/test_arxiv_package.py`

**Interfaces:**
- Consumes: the frozen result CSVs.
- Produces: an evidence-calibrated route to either time-evolving softness or nonlocal/history closure.

- [ ] **Step 1: Write a failing package test from the actual summary**

Require exact counts, all integrity fields, the actual gate verdict, and false
stronger claims. The test must also require that the scientific note states
why a positive structural reaction coordinate is not yet a readiness clock or
why a negative result routes to Hessian/history variables.

- [ ] **Step 2: Run the package test and verify RED**

Run:

```bash
python -m unittest tests.test_arxiv_package.ArXivPackageTests.test_radial_precursor_residual_gate -v
```

Expected: failure because the note/package gate is absent.

- [ ] **Step 3: Write the scientific note and complete the package test**

Document the formula, fixed data, all held-parent scores, the aggregated
propensity check, the preregistered decision, and the next microscopic branch.
Do not call a learned radial coordinate a Kramers barrier at one temperature.

- [ ] **Step 4: Run full verification**

Run:

```bash
python -m unittest discover -s tests
python -m py_compile src/ka_structural_precursor.py src/ka_smooth_cage.py \
  scripts/analyze_ka_committor_radial_structure.py \
  scripts/analyze_ka_radial_precursor_residual.py
git diff --check
git status --short --untracked-files=no
```

Expected: all tests pass, compilation exits zero, no whitespace errors, and
only intended tracked files are modified.

- [ ] **Step 5: Commit documentation and package gate**

```bash
git add docs/microscopic-radial-precursor-residual.md tests/test_arxiv_package.py
git commit -m "document radial microscopic precursor verdict"
```
