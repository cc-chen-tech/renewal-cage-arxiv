# Activated Cage Geometry Quotient Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether calibration-only cage-jump geometry closes the T=0.45 multi-k held-out scattering shape after conditioning on held-out MSD and NGP.

**Architecture:** A calibration extractor converts registered trajectory prefixes into a compact jump-geometry table. A deterministic summarizer combines that table with the existing gamma diagnostic rows, evaluates fixed-length and empirical compound-Poisson closures, writes frozen verdict artifacts, and renders one SVG. CI recomputes summary artifacts from the committed compact table because raw KA trajectories are not packaged.

**Tech Stack:** Python 3.12, NumPy, standard-library CSV/JSON, `unittest`, existing `ka_replicates` trajectory and event helpers.

## Global Constraints

- Work only in `/Users/luicy/AI/renewal-cage-arxiv/.worktrees/activated-cage-geometry` on `codex/activated-cage-geometry`.
- Use T=0.45 as the primary stationary diagnostic and T=0.58 as a canary only.
- Keep the existing lags, `k={2,4,7.25}`, Debye-Waller event definition, `half_window=5`, and absolute `F_s` tolerance 0.03 unchanged.
- No held-out event, jump-vector, or cage-residual information enters calibration geometry.
- All strong scientific claim flags remain zero.
- Use `/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3` for Python 3.12 verification.

---

### Task 1: Calibration Jump-Geometry Extractor

**Files:**
- Create: `scripts/extract_ka_calibration_jump_geometry.py`
- Create: `tests/test_ka_activated_cage_geometry.py`

**Interfaces:**
- Consumes: registered `ensemble_manifest.json`, replicate trajectories, calibration time, and committed Debye-Waller threshold rows.
- Produces: `jump_geometry_statistics(jump_vectors, wave_numbers) -> dict[str, float]` and a CLI table with one row per temperature and replicate.

- [ ] **Step 1: Write the failing formula/statistics tests**

```python
def test_jump_geometry_statistics_uses_component_characteristic(self):
    jumps = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    row = self.analysis.jump_geometry_statistics(jumps, np.array([2.0]))
    self.assertAlmostEqual(row["jump_msd"], 1.0)
    self.assertAlmostEqual(row["jump_component_fourth_moment"], 1.0 / 3.0)
    self.assertAlmostEqual(row["jump_characteristic_k2"], (math.cos(2.0) + 2.0) / 3.0)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_ka_activated_cage_geometry.JumpGeometryExtractionTests -v`

Expected: import failure because `scripts/extract_ka_calibration_jump_geometry.py` does not exist.

- [ ] **Step 3: Implement the statistic and strict prefix extraction**

```python
def jump_geometry_statistics(jump_vectors, wave_numbers):
    jumps = np.asarray(jump_vectors, dtype=float)
    squared = np.sum(jumps**2, axis=1)
    flat = jumps.reshape(-1)
    result = {
        "event_count": float(len(jumps)),
        "jump_msd": float(np.mean(squared)),
        "jump_radial_fourth_moment": float(np.mean(squared**2)),
        "jump_component_fourth_moment": float(np.mean(flat**4)),
    }
    for k in wave_numbers:
        result[wave_key(k)] = float(np.mean(np.cos(float(k) * flat)))
    return result
```

The CLI must slice `positions[:calibration_time + 1]` before calling
`position_fluctuation_values` and `extract_debye_waller_cage_jumps`, and write
`calibration_events_only=1`, `heldout_events_used=0`, source manifest path,
source trajectory path, threshold source, event definition, and all strong
claim flags as zero.

- [ ] **Step 4: Verify GREEN**

Run: `/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_ka_activated_cage_geometry.JumpGeometryExtractionTests -v`

Expected: all extraction tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_ka_calibration_jump_geometry.py tests/test_ka_activated_cage_geometry.py
git commit -m "extract calibration cage jump geometry"
```

---

### Task 2: Compound-Poisson Geometry Quotient And Frozen Gate

**Files:**
- Create: `scripts/summarize_ka_activated_cage_geometry.py`
- Modify: `tests/test_ka_activated_cage_geometry.py`

**Interfaces:**
- Consumes: compact geometry rows, gamma variance-mixture rows, and existing stationarity/provenance controls.
- Produces: `empirical_geometry_quotient(...)`, `fixed_length_geometry_quotient(...)`, `classify_geometry_gate(...)`, rows CSV, gate CSV, and SVG.

- [ ] **Step 1: Write failing quotient tests**

```python
def test_empirical_geometry_quotient_matches_compound_poisson_formula(self):
    result = self.summary.empirical_geometry_quotient(
        msd=0.6,
        ngp=0.5,
        jump_msd=0.2,
        jump_component_fourth_moment=0.04,
        jump_characteristic={2.0: 0.8},
    )
    self.assertAlmostEqual(result["mean_event_count"], 1.5)
    self.assertAlmostEqual(result["cage_variance"], (0.6 - 0.3) / 6.0)
    self.assertAlmostEqual(result["predicted_fs"][2.0], math.exp(-0.2 - 0.3))
```

Add separate tests that reject negative inferred cage variance, recover the
fixed-length 3/21 support canary, require at least 80% primary support, require
all three primary replicates, and lock all strong claim flags to zero.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_ka_activated_cage_geometry.GeometryQuotientTests -v`

Expected: import failure because the summarizer does not exist.

- [ ] **Step 3: Implement the minimal quotient and classifier**

```python
def empirical_geometry_quotient(*, msd, ngp, jump_msd, jump_component_fourth_moment, jump_characteristic):
    count = ngp * msd**2 / (3.0 * jump_component_fourth_moment)
    cage = (msd - count * jump_msd) / 6.0
    if count < 0.0 or cage < 0.0:
        return {"supported": 0.0}
    return {
        "supported": 1.0,
        "mean_event_count": count,
        "cage_variance": cage,
        "predicted_fs": {
            k: math.exp(-k * k * cage + count * (phi - 1.0))
            for k, phi in jump_characteristic.items()
        },
    }
```

The classifier must use absolute `F_s` errors, not fitted normalized scores,
and keep T=0.58 canary-only whenever source stationarity is false.

- [ ] **Step 4: Verify GREEN**

Run: `/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_ka_activated_cage_geometry -v`

Expected: all quotient and extraction tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/summarize_ka_activated_cage_geometry.py tests/test_ka_activated_cage_geometry.py
git commit -m "test activated cage geometry quotient"
```

---

### Task 3: Real Two-Temperature Artifacts And Interpretation

**Files:**
- Create: `data/renewal_cage_ka_replicates_T045_calibration_jump_geometry.csv`
- Create: `data/renewal_cage_ka_replicates_T058_calibration_jump_geometry.csv`
- Create: `data/renewal_cage_ka_activated_cage_geometry_rows.csv`
- Create: `data/renewal_cage_ka_activated_cage_geometry_gate.csv`
- Create: `figures/renewal_cage_ka_activated_cage_geometry.svg`
- Create: `docs/microscopic-activated-cage-geometry.md`
- Modify: `README.md`
- Modify: `tests/test_arxiv_package.py`

**Interfaces:**
- Consumes: shared read-only raw KA trajectory ensembles for the extraction run, then committed compact geometry tables for deterministic recomputation.
- Produces: reviewable evidence artifacts and a claim-bounded result document.

- [ ] **Step 1: Add failing artifact tests**

Tests must assert exact grids and row counts, source/calibration flags, the
fixed-length support count, empirical support coverage, every temperature gate
field, all strong zero flags, README/doc wording, and byte-identical CSV/SVG
recomputation in a temporary directory.

- [ ] **Step 2: Run artifact tests and verify RED**

Run: `/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_arxiv_package.ActivatedCageGeometryArtifactTests -v`

Expected: missing artifact failure.

- [ ] **Step 3: Extract compact calibration geometry**

Run the extractor separately for T=0.45 and T=0.58 against the registered raw
ensembles.  Never write into those source ensemble directories.

- [ ] **Step 4: Generate rows, verdict, and SVG**

Run the summarizer using only committed compact geometry tables and existing
committed diagnostic/control tables.  Record the result as pass, fail, or
unsupported without changing any tolerance.

- [ ] **Step 5: Write result interpretation**

The document must derive the formula, report per-k errors and support, compare
against the gamma and fixed-length nulls, and state whether a distributed-basin
continuous potential is justified as the next phase.  It must not claim a
unique potential or microscopic closure.

- [ ] **Step 6: Verify GREEN and visually inspect the SVG**

Run the focused artifact tests, render the SVG to PNG, inspect axis semantics,
clipping labels, support annotations, and all text bounds.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/microscopic-activated-cage-geometry.md data/renewal_cage_ka_*activated_cage_geometry*.csv data/renewal_cage_ka_replicates_T0*_calibration_jump_geometry.csv figures/renewal_cage_ka_activated_cage_geometry.svg tests/test_arxiv_package.py
git commit -m "report activated cage geometry gate"
```

---

### Task 4: Full Verification And Publication

**Files:**
- Modify only files required by deterministic recomputation or packaging failures discovered during verification.

**Interfaces:**
- Consumes: the complete branch.
- Produces: a pushed branch and ready pull request with local and remote validation evidence.

- [ ] **Step 1: Run focused and full tests**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_ka_activated_cage_geometry -v
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s tests -p 'test_*.py' -q
```

- [ ] **Step 2: Run repository generators**

```bash
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/generate_renewal_cage_results.py
/Users/luicy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/build_arxiv_package.py
git diff --check
```

- [ ] **Step 3: Re-run tests after generators**

Confirm the working tree contains only intentional generated changes and that
the full suite remains green.

- [ ] **Step 4: Commit any deterministic packaging fixes**

Use a narrow commit message that names the reproducibility fix; do not change
physical tolerances or claim flags.

- [ ] **Step 5: Push and create the PR**

Push `codex/activated-cage-geometry`, create a ready PR with the actual verdict
and validation bundle, wait for CI, and fix any reproducibility issue without
requesting repeated approval.
