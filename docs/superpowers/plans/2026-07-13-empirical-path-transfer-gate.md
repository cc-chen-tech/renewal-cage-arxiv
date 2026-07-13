# Empirical Path Transfer Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether ordered single-particle block paths, rather than another mobility clock, are necessary and sufficient for held-out low-temperature MSD, NGP, and multi-k self-intermediate scattering.

**Architecture:** Add pure block-path observable and null-model functions to `src/ka_replicates.py`. A new temperature-level CLI reads calibration trajectories only, scores contiguous, within-particle-shuffled, and analytically direction-randomized paths against existing held-out factorization rows, and writes model-comparable CSVs. A separate classifier combines both temperatures with the unchanged Markov verdict and produces the mechanism-selection artifact.

**Tech Stack:** Python 3, NumPy, standard-library CSV/JSON/argparse/unittest, deterministic SVG.

## Global Constraints

- Use existing `T=0.45` and `T=0.58` calibration/held-out halves and block size 20.
- No held-out displacement, event, or observable enters a calibration kernel.
- Score only common held-out lags divisible by 20.
- Tolerances remain MSD `0.10`, NGP `0.30`, and multi-k `F_s` `0.03`.
- Keep `microdynamic_closure_claim_allowed=0`, `spatial_facilitation_claim_allowed=0`, and `thermodynamic_claim_allowed=0`.
- Do not modify or refit the existing two-state Markov artifacts.

---

### Task 1: Pure Path Kernels

**Files:**
- Modify: `src/ka_replicates.py`
- Test: `tests/test_ka_replicates.py`

**Interfaces:**
- Produces: `cumulative_block_observables(block_displacements, block_count, wave_numbers) -> dict[str, float]`
- Produces: `within_particle_time_shuffle(block_displacements, rng) -> np.ndarray`
- Produces: `direction_randomized_block_observables(block_displacements, block_count, wave_numbers) -> dict[str, float]`

- [ ] Write failing tests proving cumulative vectors reproduce direct sums, time shuffling preserves each particle's vector multiset while changing order, and direction randomization preserves ordered lengths while removing vector recoil analytically.
- [ ] Run `python -m unittest tests.test_ka_replicates.KAReplicatePreparationTests.test_cumulative_block_observables_match_direct_windows tests.test_ka_replicates.KAReplicatePreparationTests.test_within_particle_shuffle_preserves_particle_vector_multisets tests.test_ka_replicates.KAReplicatePreparationTests.test_direction_randomized_path_preserves_lengths_not_recoil -v`; expect missing-function failures.
- [ ] Implement strict finite-array validation. Use prefix sums for contiguous cumulative vectors. For the direction null in three dimensions, average `sum(l_i^2)`, `sum(l_i^4) + 5/3 * ((sum(l_i^2))^2 - sum(l_i^4))`, and `product(sinc(k*l_i/pi))` over particle windows.
- [ ] Re-run the three tests and the full `tests.test_ka_replicates` module; expect PASS.
- [ ] Commit `add empirical block path kernels`.

### Task 2: Temperature-Level Held-Out Transfer

**Files:**
- Create: `scripts/analyze_ka_empirical_path_transfer.py`
- Modify: `tests/test_ka_replicates.py`

**Interfaces:**
- Consumes the Task 1 functions and existing LAMMPS trajectory loader.
- Produces `<prefix>_rows.csv`, `<prefix>_summary.csv`, and `<prefix>_verdict.csv` with rows for `contiguous_empirical_path`, `within_particle_time_shuffle`, and `direction_randomized_path`.
- Produces: `classify_path_model_transfer(summary_rows, *, replicate_rows) -> list[dict[str, object]]`.

- [ ] Write failing tests for model-wise ensemble aggregation, max-error tolerances, paired replicate consensus, deterministic shuffle seeds, Monte Carlo precision fields, and held-out-exclusion provenance.
- [ ] Run the focused new tests; expect missing-script or missing-function failures.
- [ ] Implement the CLI with `--calibration-time`, `--heldout-factorization`, `--block-size 20`, `--shuffle-realizations`, `--seed`, and `--output-prefix`. Generate shuffles independently per replicate and realization, average predictions, record their Monte Carlo standard errors, and set `shuffle_precision_pass=1` only when relative MSD SE is at most `0.01`, NGP SE at most `0.03`, and each `F_s` SE at most `0.003`.
- [ ] Ensure every output row records `heldout_path_used_in_prediction=0`, `macro_fit_parameter_count=0`, `calibration_path_distribution_used=1`, and all three claim boundaries as zero.
- [ ] Run focused and module tests; expect PASS.
- [ ] Commit `score empirical path transfer nulls`.

### Task 3: Real Two-Temperature Mechanism Selection

**Files:**
- Create: `scripts/summarize_ka_empirical_path_transfer.py`
- Modify: `tests/test_ka_replicates.py`
- Modify: `tests/test_arxiv_package.py`
- Create: `data/renewal_cage_ka_replicates_T045_empirical_path_{rows,summary,verdict}.csv`
- Create: `data/renewal_cage_ka_replicates_T058_empirical_path_{rows,summary,verdict}.csv`
- Create: `data/renewal_cage_ka_empirical_path_crossover.csv`
- Create: `figures/renewal_cage_ka_empirical_path_crossover.svg`

**Interfaces:**
- Produces: `classify_empirical_path_crossover(low_verdicts, high_verdicts, markov_low) -> dict[str, object]`.

- [ ] Write a failing decision-table test requiring low-temperature contiguous closure, low-temperature Markov failure, low-temperature shuffle failure on the same higher-order channel, at least two paired replicates favoring contiguous paths, and high-temperature contiguous closure.
- [ ] Run the decision test; expect missing-function failure.
- [ ] Implement the classifier and deterministic SVG showing each model's error divided by the frozen tolerance at both temperatures.
- [ ] Run the temperature CLI at `T=0.45` with calibration time 5000 and at `T=0.58` with calibration time 750, using 64 fixed-seed shuffle realizations; increase realizations only if the preregistered precision gate fails.
- [ ] Generate the crossover CSV/SVG and add artifact tests for numeric outcomes, provenance, and all claim boundaries.
- [ ] Run focused tests and visually inspect the SVG; expect PASS and a legible non-overlapping figure.
- [ ] Commit `select single particle cage path memory`.

### Task 4: Repository Verification

**Files:**
- Modify only files required by failed deterministic generation or package checks.

**Interfaces:**
- Consumes all previous artifacts; produces a clean verified commit state.

- [ ] Run `python -m unittest discover -s tests -v`; expect zero failures.
- [ ] Run `python scripts/generate_renewal_cage_results.py`; expect exit zero without unrelated drift.
- [ ] Run `python scripts/build_arxiv_package.py`; expect `dist/renewal-cage-arxiv-source.zip`.
- [ ] Run `git diff --check` and inspect `git status --short`; expect only intended files before the final commit and a clean worktree afterward.
- [ ] Commit any verification-only deterministic artifact update with `verify empirical path transfer gate`; do not push without explicit user instruction.
