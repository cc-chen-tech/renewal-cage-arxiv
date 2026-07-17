# Nonlinear Cage-Path Cumulant Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a calibration-only constrained-surrogate gate that isolates low-temperature connected path memory beyond one-block radial disorder and the two-point displacement spectrum, then quantify the low-k scattering range controlled by the fourth path cumulant.

**Architecture:** Pure NumPy kernels in `src/ka_replicates.py` implement shared-phase spectral randomization, radial multivariate iteration, spectral diagnostics, and the fourth-cumulant scattering formula. One temperature-level CLI performs deterministic realizations, held-out scoring, Monte Carlo precision, paired replicate scoring, and calibration-quarter stationarity controls. Two small summarizers produce the cumulant-validity and two-temperature mechanism artifacts without adding a microscopic fit.

**Tech Stack:** Python 3, NumPy, standard-library CSV/JSON/argparse/unittest, deterministic SVG, existing LAMMPS trajectory loader.

## Global Constraints

- Work in `/Users/luicy/AI/renewal-cage-arxiv`; do not create another repository or worktree.
- Primary decision: `T=0.45`, calibration time `5000`, block size `20`, at least eight deterministic surrogate realizations.
- Sensitivity only: `T=0.58` at block sizes `20` and `10`; do not claim a binary crossover unless both resolutions agree.
- No held-out path, event, or observable may enter a surrogate.
- Frozen curve tolerances: relative MSD `0.10`, NGP `0.30`, multi-k `F_s` `0.03`.
- Frozen quality tolerances: radial error `1e-12`, spectral-matrix NRMSE `0.015`, one-block relative MSD `0.01`, one-block NGP `0.03`, one-block `F_s` `0.003`.
- Frozen Monte Carlo SE tolerances: relative MSD `0.01`, NGP `0.03`, `F_s` `0.003`.
- Keep `microdynamic_closure_claim_allowed=0`, `spatial_facilitation_claim_allowed=0`, and `thermodynamic_claim_allowed=0` in every artifact.
- The observed-MSD/NGP cumulant calculation is a diagnostic, never a held-out prediction.

---

### Task 1: Pure Spectral, Radial, and Cumulant Kernels

**Files:**
- Modify: `src/ka_replicates.py` near `cumulative_block_observables`
- Test: `tests/test_ka_replicates.py`

**Interfaces:**
- Produces: `cross_spectral_matrix_nrmse(reference: np.ndarray, candidate: np.ndarray) -> float`
- Produces: `shared_phase_spectral_projection(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray`
- Produces: `phase_randomized_cross_spectrum(reference: np.ndarray, rng: np.random.Generator) -> np.ndarray`
- Produces: `radial_rank_projection(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray`
- Produces: `radial_multivariate_surrogate(reference: np.ndarray, rng: np.random.Generator, *, iteration_count: int) -> dict[str, object]`
- Produces: `fourth_cumulant_scattering(msd: np.ndarray, ngp: np.ndarray, wave_number: float) -> np.ndarray`

- [ ] **Step 1: Write failing spectral and radial property tests**

Add tests with a fixed `(2, 8, 3)` block array. Require a shared-phase projection to preserve the reference spectral matrix, radial projection to preserve each particle's sorted radii, and the iterative surrogate to be deterministic for equal seeds but different for unequal seeds.

```python
def test_shared_phase_projection_preserves_cross_spectral_matrix(self):
    reference = np.arange(48, dtype=float).reshape(2, 8, 3) / 17.0
    candidate = reference[:, ::-1].copy()
    projected = ka_replicates.shared_phase_spectral_projection(reference, candidate)
    self.assertLess(
        ka_replicates.cross_spectral_matrix_nrmse(reference, projected),
        1e-12,
    )

def test_radial_projection_preserves_particle_radius_multisets(self):
    reference = np.arange(48, dtype=float).reshape(2, 8, 3) / 17.0
    candidate = np.roll(reference, 2, axis=1) + 0.1
    projected = ka_replicates.radial_rank_projection(reference, candidate)
    np.testing.assert_allclose(
        np.sort(np.linalg.norm(projected, axis=2), axis=1),
        np.sort(np.linalg.norm(reference, axis=2), axis=1),
    )
```

Also require `phase_randomized_cross_spectrum` to preserve the spectral matrix
to `1e-12`, preserve the DC component, and produce different paths for different
seeds.

- [ ] **Step 2: Run the new property tests and verify missing-function failures**

Run:

```bash
python -m unittest \
  tests.test_ka_replicates.KAReplicatePreparationTests.test_shared_phase_projection_preserves_cross_spectral_matrix \
  tests.test_ka_replicates.KAReplicatePreparationTests.test_radial_projection_preserves_particle_radius_multisets -v
```

Expected: `AttributeError` for the missing functions.

- [ ] **Step 3: Implement strict pure projections**

Validate finite `(particles, blocks, 3)` arrays with at least two blocks. In the spectral projection, retain the reference Fourier amplitudes and relative component phases while choosing one least-squares common phase per particle and frequency from the candidate. In the radial projection, preserve candidate directions and assign reference radii by rank; reject zero candidate radii paired with positive assigned radii.

```python
reference_fft = np.fft.rfft(reference, axis=1)
candidate_fft = np.fft.rfft(candidate, axis=1)
reference_amplitude = np.abs(reference_fft)
reference_unit = np.divide(
    reference_fft,
    reference_amplitude,
    out=np.ones_like(reference_fft),
    where=reference_amplitude > 1e-15,
)
candidate_unit = np.divide(
    candidate_fft,
    np.abs(candidate_fft),
    out=np.ones_like(candidate_fft),
    where=np.abs(candidate_fft) > 1e-15,
)
common = np.sum(
    reference_amplitude**2 * candidate_unit * np.conjugate(reference_unit),
    axis=2,
)
common = np.divide(common, np.abs(common), out=np.ones_like(common), where=np.abs(common) > 1e-15)
return np.fft.irfft(
    reference_amplitude * reference_unit * common[:, :, np.newaxis],
    n=reference.shape[1],
    axis=1,
)
```

- [ ] **Step 4: Write failing iterative-surrogate validation and determinism tests**

Require positive integer iterations, a NumPy generator, finite input, deterministic outputs and diagnostics, exact radial preservation, and decreasing spectral NRMSE relative to the initial within-particle shuffle.

- [ ] **Step 5: Implement the radial iteration and diagnostics**

Initialize with `within_particle_time_shuffle`, alternate spectral and radial projections exactly `iteration_count` times, and return:

```python
{
    "displacements": surrogate,
    "iteration_count": float(iteration_count),
    "cross_spectral_matrix_nrmse": cross_spectral_matrix_nrmse(reference, surrogate),
    "radial_distribution_maximum_absolute_error": radial_error,
}
```

Do not accept a tolerance or silently stop early; convergence is classified by the CLI so every realization has identical computational treatment.

- [ ] **Step 6: Write and run a failing cumulant identity test**

```python
def test_fourth_cumulant_scattering_matches_isotropic_expansion(self):
    msd = np.array([0.3, 0.6])
    ngp = np.array([0.2, 0.4])
    expected = np.exp(-4.0 * msd / 6.0 + 16.0 * ngp * msd**2 / 72.0)
    np.testing.assert_allclose(
        ka_replicates.fourth_cumulant_scattering(msd, ngp, 2.0), expected
    )
```

Expected before implementation: missing-function failure.

- [ ] **Step 7: Implement the cumulant kernel with validity checks**

Require aligned finite nonnegative MSD, finite NGP greater than `-1`, and positive finite wave number. Return the exponential without clipping; downstream code must mark values above one invalid rather than hide truncation failure.

- [ ] **Step 8: Run focused and module tests**

Run:

```bash
python -m unittest tests.test_ka_replicates.KAReplicatePreparationTests -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit the pure kernels**

```bash
git add src/ka_replicates.py tests/test_ka_replicates.py
git commit -m "add nonlinear path surrogate kernels"
```

---

### Task 2: Temperature-Level Surrogate and Stationarity Analyzer

**Files:**
- Create: `scripts/analyze_ka_nonlinear_path_surrogate.py`
- Modify: `tests/test_ka_replicates.py`

**Interfaces:**
- Consumes: Task 1 kernels, `load_lammps_custom_trajectory`, existing held-out factorization rows
- Produces: `<prefix>_{rows,summary,replicate_scores,stationarity,verdict}.csv`
- Produces: `surrogate_seed(base_seed: int, replicate: int, realization: int) -> int`
- Produces: `classify_nonlinear_path_surrogate(...) -> dict[str, object]`

- [ ] **Step 1: Write failing aggregation and decision-table tests**

Use synthetic rows to require all of the following: quality pass, precision pass, three stationarity comparisons pass, ensemble contiguous closure, surrogate MSD closure, surrogate higher-order failure, every replicate surrogate score above one, and every paired contiguous score smaller. Flip each condition separately and assert `nonlinear_single_particle_path_memory_required == 0.0`.

- [ ] **Step 2: Run focused tests and verify missing-script failures**

Run:

```bash
python -m unittest \
  tests.test_ka_replicates.KAReplicatePreparationTests.test_nonlinear_path_surrogate_requires_quality_stationarity_and_paired_consensus \
  tests.test_ka_replicates.KAReplicatePreparationTests.test_nonlinear_path_surrogate_seed_is_deterministic -v
```

Expected: import or missing-function failure.

- [ ] **Step 3: Implement deterministic loading and realization rows**

CLI arguments:

```text
ensemble_directory
--calibration-time
--heldout-factorization
--block-size
--surrogate-realizations
--iteration-count
--seed
--output-prefix
```

Load only `calibration_time + 1` frames. Construct type-A block vectors as in
`analyze_ka_empirical_path_transfer.py`. Recompute the contiguous baseline from
those same blocks at every local lag so block-size sensitivity does not depend on
an older table with a different lag set. For every realization, score both
`phase_randomized_cross_spectrum` and `radial_multivariate_surrogate`; record
quality diagnostics, one-block errors, predicted held-out metrics, and provenance
flags. The exact-spectrum phase null is diagnostic and cannot satisfy the radial
quality gate. Never omit failed-quality rows. Artifact tests must confirm that
the regenerated `T=0.45`, block-20 contiguous summary matches the existing
empirical-path table.

- [ ] **Step 4: Implement ensemble summaries and Monte Carlo precision**

Average realizations within replicate and then independent replicates. Report maximum curve errors and maximum Monte Carlo SE using the frozen thresholds. Keep quality failure separate from physical curve failure.

- [ ] **Step 5: Implement calibration-quarter stationarity controls**

Split the calibration block axis in half. Use `cumulative_block_observables` on early and late quarters at common lags no longer than a quarter. Emit early-late, early-heldout, and late-heldout summary rows and their frozen pass/fail fields.

- [ ] **Step 6: Implement paired replicate scoring and the final classifier**

Use:

```python
score = max(max_ngp_absolute_error / 0.30, max_fs_absolute_error / 0.03)
```

Record individual contiguous closure separately from paired superiority. Set `nonlinear_single_particle_path_memory_required=1` only under the complete design truth table. Set `next_minimal_model` to `finite_lifetime_reversible_cage_state` but keep all closure claim flags zero.

- [ ] **Step 7: Run focused and module tests**

```bash
python -m unittest tests.test_ka_replicates -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit the analyzer**

```bash
git add scripts/analyze_ka_nonlinear_path_surrogate.py tests/test_ka_replicates.py
git commit -m "score nonlinear path surrogate null"
```

---

### Task 3: Connected-Cumulant Scattering Validity

**Files:**
- Create: `scripts/analyze_ka_path_cumulant_scattering.py`
- Modify: `tests/test_ka_replicates.py`

**Interfaces:**
- Consumes: held-out factorization rows containing observed MSD, NGP, and multi-k `F_s`
- Produces: `<prefix>_{rows,validity}.csv`
- Produces: `cumulant_scattering_validity(rows, *, error_tolerance: float = 0.03) -> list[dict[str, object]]`

- [ ] **Step 1: Write failing tests for formula use and diagnostic provenance**

Synthetic rows must prove correct replicate aggregation, maximum absolute error, longest contiguous valid lag, invalidation when `F_s^(4) > 1`, and fields `observed_msd_used=1`, `observed_ngp_used=1`, `heldout_prediction_claim_allowed=0`.

- [ ] **Step 2: Run tests and verify missing-script failure**

Run:

```bash
python -m unittest \
  tests.test_ka_replicates.KAReplicatePreparationTests.test_path_cumulant_scattering_reports_contiguous_validity_horizon -v
```

Expected: import failure.

- [ ] **Step 3: Implement row generation and validity horizons**

Parse wave numbers from `observed_fs_k*`, aggregate observed MSD/NGP over independent replicates at each lag, call `fourth_cumulant_scattering`, and emit one row per `(temperature, lag, k)`. A validity horizon ends at the first tested lag whose error exceeds `0.03` or whose approximation leaves `[0, 1]`; later accidental re-entry does not extend it.

- [ ] **Step 4: Run focused and module tests**

```bash
python -m unittest tests.test_ka_replicates -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the diagnostic**

```bash
git add scripts/analyze_ka_path_cumulant_scattering.py tests/test_ka_replicates.py
git commit -m "add path cumulant scattering diagnostic"
```

---

### Task 4: Two-Temperature Mechanism Summary and Figure

**Files:**
- Create: `scripts/summarize_ka_nonlinear_path_gate.py`
- Modify: `tests/test_ka_replicates.py`
- Modify: `tests/test_arxiv_package.py`

**Interfaces:**
- Consumes: primary `T=0.45` verdict, both `T=0.58` resolution verdicts, cumulant validity tables, existing empirical-path crossover
- Produces: `data/renewal_cage_ka_nonlinear_path_gate.csv`
- Produces: `figures/renewal_cage_ka_nonlinear_path_gate.svg`
- Produces: `classify_nonlinear_path_gate(...) -> dict[str, object]`

- [ ] **Step 1: Write a failing mechanism truth-table test**

Require the low-temperature nonlinear-memory decision, preserve the high-temperature block-resolution disagreement, set `binary_temperature_crossover_claim_allowed=0`, and keep all closure boundaries zero.

- [ ] **Step 2: Run the decision test and verify missing-script failure**

Expected: import failure.

- [ ] **Step 3: Implement the combined classifier and deterministic SVG**

The SVG has two panels: normalized held-out errors for contiguous versus radial surrogate, and fourth-cumulant `F_s` error versus lag for each k. Use existing restrained figure conventions, fixed dimensions, no raster dependency, and no text overlap.

- [ ] **Step 4: Add artifact contract tests**

Require all expected CSV columns, finite numeric values, `3/3` low-temperature paired consensus, quality/precision/stationarity pass, nonlinear-memory selection, high-temperature resolution sensitivity, and all claim boundaries zero. Require the SVG to contain its title and no `nan` or `inf`.

- [ ] **Step 5: Run focused tests**

```bash
python -m unittest \
  tests.test_ka_replicates.KAReplicatePreparationTests.test_nonlinear_path_gate_requires_quality_stationarity_and_paired_consensus \
  tests.test_arxiv_package.ArxivPackageTests.test_nonlinear_path_gate_artifacts_preserve_claim_boundaries -v
```

Expected: pass after implementation.

- [ ] **Step 6: Commit the summarizer and contracts**

```bash
git add scripts/summarize_ka_nonlinear_path_gate.py tests/test_ka_replicates.py tests/test_arxiv_package.py
git commit -m "classify nonlinear cage path memory"
```

---

### Task 5: Real Data, Visual Inspection, and Repository Verification

**Files:**
- Create: `data/renewal_cage_ka_replicates_T045_nonlinear_path_*.csv`
- Create: `data/renewal_cage_ka_replicates_T058_block20_nonlinear_path_*.csv`
- Create: `data/renewal_cage_ka_replicates_T058_block10_nonlinear_path_*.csv`
- Create: `data/renewal_cage_ka_replicates_{T045,T058}_path_cumulant_{rows,validity}.csv`
- Create: `data/renewal_cage_ka_nonlinear_path_gate.csv`
- Create: `figures/renewal_cage_ka_nonlinear_path_gate.svg`

**Interfaces:**
- Consumes: Tasks 1-4 CLIs and existing trajectory ensembles
- Produces: frozen real-data mechanism result and package artifacts

- [ ] **Step 1: Run the primary low-temperature analyzer**

```bash
python scripts/analyze_ka_nonlinear_path_surrogate.py \
  tmp/ka_replicates/T045 \
  --calibration-time 5000 \
  --heldout-factorization data/renewal_cage_ka_replicates_T045_event_oracle_factorization_rows.csv \
  --block-size 20 --surrogate-realizations 8 --iteration-count 110 --seed 211003 \
  --output-prefix data/renewal_cage_ka_replicates_T045_nonlinear_path
```

Expected: quality, precision, stationarity, ensemble MSD, `3/3` surrogate rejection, and `3/3` paired superiority pass; NGP and `F_s` null curves fail.

- [ ] **Step 2: Run both high-temperature sensitivity analyzers**

```bash
python scripts/analyze_ka_nonlinear_path_surrogate.py \
  tmp/ka_replicates/T058 \
  --calibration-time 750 \
  --heldout-factorization data/renewal_cage_ka_replicates_T058_event_oracle_factorization_rows.csv \
  --block-size 20 --surrogate-realizations 8 --iteration-count 240 --seed 211003 \
  --output-prefix data/renewal_cage_ka_replicates_T058_block20_nonlinear_path

python scripts/analyze_ka_nonlinear_path_surrogate.py \
  tmp/ka_replicates/T058 \
  --calibration-time 750 \
  --heldout-factorization data/renewal_cage_ka_replicates_T058_event_oracle_factorization_rows.csv \
  --block-size 10 --surrogate-realizations 8 --iteration-count 240 --seed 211003 \
  --output-prefix data/renewal_cage_ka_replicates_T058_block10_nonlinear_path
```

Expected: apply the same `0.015` spectral-quality threshold and record any
failure; the combined classifier must not permit a binary crossover claim when
the two block resolutions disagree or either quality gate fails.

- [ ] **Step 3: Generate both cumulant diagnostics and the combined artifact**

```bash
python scripts/analyze_ka_path_cumulant_scattering.py \
  data/renewal_cage_ka_replicates_T045_event_oracle_factorization_rows.csv \
  --output-prefix data/renewal_cage_ka_replicates_T045_path_cumulant

python scripts/analyze_ka_path_cumulant_scattering.py \
  data/renewal_cage_ka_replicates_T058_event_oracle_factorization_rows.csv \
  --output-prefix data/renewal_cage_ka_replicates_T058_path_cumulant

python scripts/summarize_ka_nonlinear_path_gate.py \
  --low-verdict data/renewal_cage_ka_replicates_T045_nonlinear_path_verdict.csv \
  --high-block20-verdict data/renewal_cage_ka_replicates_T058_block20_nonlinear_path_verdict.csv \
  --high-block10-verdict data/renewal_cage_ka_replicates_T058_block10_nonlinear_path_verdict.csv \
  --low-cumulant-validity data/renewal_cage_ka_replicates_T045_path_cumulant_validity.csv \
  --high-cumulant-validity data/renewal_cage_ka_replicates_T058_path_cumulant_validity.csv \
  --empirical-path-crossover data/renewal_cage_ka_empirical_path_crossover.csv \
  --output-csv data/renewal_cage_ka_nonlinear_path_gate.csv \
  --output-svg figures/renewal_cage_ka_nonlinear_path_gate.svg
```

Expected: low-k validity is longer than high-k validity and all outputs are
deterministic.

- [ ] **Step 4: Visually inspect the SVG**

Render or open `figures/renewal_cage_ka_nonlinear_path_gate.svg`. Check labels, normalized-tolerance line, legend, panel titles, and smallest text at desktop scale. Fix only deterministic figure code and regenerate if needed.

- [ ] **Step 5: Run full verification**

```bash
python -m unittest discover -s tests -v
python scripts/generate_renewal_cage_results.py
python scripts/build_arxiv_package.py
git diff --check
```

Expected: zero test failures, deterministic result generation, `dist/renewal-cage-arxiv-source.zip`, and no whitespace errors.

- [ ] **Step 6: Audit the final claim fields and status**

Confirm the combined CSV selects nonlinear low-temperature path memory, does not claim a robust binary temperature crossover, does not claim a unique reversible-cage model, and leaves microdynamic closure, spatial facilitation, and thermodynamic claims at zero.

- [ ] **Step 7: Commit the real-data result**

```bash
git add data/renewal_cage_ka_*nonlinear_path* data/renewal_cage_ka_*path_cumulant* \
  data/renewal_cage_ka_nonlinear_path_gate.csv \
  figures/renewal_cage_ka_nonlinear_path_gate.svg
git commit -m "identify nonlinear cage path cumulants"
```

- [ ] **Step 8: Verify the branch is clean and do not push**

```bash
git status --short --branch
```

Expected: clean `codex/nonlinear-path-cumulant-gate`; pushing and local merge remain separate user decisions.
