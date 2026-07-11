# Stationary Finite-Flight Renewal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an equilibrium stationary renewal clock, finite-duration cage-flight observables, and event-space Green-Kubo transport correction, then test them against fixed derived Kob-Andersen trajectory canaries.

**Architecture:** Keep the legacy delayed-first persistence/exchange API unchanged. Add independent stationary-clock, flight-kernel, and correlated-transport functions in the existing analytic module, then expose their fixed real-data audit through the established CSV/SVG/PDF generator and manuscript package gates.

**Tech Stack:** Python 3, NumPy, unittest, CSV, SVG, ReportLab, LaTeX.

## Global Constraints

- Preserve all existing public signatures and outputs.
- Commit no raw Zenodo trajectories; commit only derived statistics and source metadata.
- Keep `thermodynamic_claim_allowed=0` in every new audit row.
- Treat overlap susceptibility as a proxy, not a derived spatial four-point correlation function.
- Use fixed `T=0.58` and `T=0.45` numerical canaries; do not tune against held-out observables during generation.

---

### Task 1: Stationary renewal clock

**Files:**
- Modify: `src/renewal_cage.py`
- Modify: `tests/test_renewal_cage.py`

**Interfaces:**
- Produces: `StationaryRenewalParams(exchange_mean: float, exchange_cv2: float)`, `stationary_gamma_count_moments(times: np.ndarray, params: StationaryRenewalParams) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write failing tests** for parameter validation, `persistence_mean = exchange_mean*(1+exchange_cv2)/2`, and `mean_count = times/exchange_mean` at several nonzero times.
- [ ] **Step 2: Verify RED** with `python -m unittest tests.test_renewal_cage.RenewalCageTests.test_stationary_renewal_residual_life -v`; expect an import error for `StationaryRenewalParams`.
- [ ] **Step 3: Implement the stationary gamma renewal calculation** using gamma survival quadrature for the equilibrium first wait and renewal recursion for the second moment; overwrite the numerical first moment with the exact stationary identity to eliminate quadrature drift.
- [ ] **Step 4: Verify GREEN** with the focused test and `python -m unittest tests.test_renewal_cage -v`; expect all renewal-cage tests to pass.
- [ ] **Step 5: Commit** `src/renewal_cage.py tests/test_renewal_cage.py` with message `add stationary renewal count closure`.

### Task 2: Finite-flight observables

**Files:**
- Modify: `src/renewal_cage.py`
- Modify: `tests/test_renewal_cage.py`

**Interfaces:**
- Consumes: stationary count means and variances from Task 1.
- Produces: `finite_flight_weight_integral`, `finite_flight_moments_1d`, `finite_flight_ngp_1d`, `finite_flight_self_intermediate_scattering`.

- [ ] **Step 1: Write failing tests** checking the two analytic branches at `time=duration/2` and `time=2*duration`, the zero-duration limit, long-time NGP recovery, and convergence to instantaneous-jump moments as duration approaches zero.
- [ ] **Step 2: Verify RED** with `python -m unittest tests.test_renewal_cage.RenewalCageTests.test_finite_flight_weight_integral -v`; expect an import error for the new function.
- [ ] **Step 3: Implement the kernels** with vectorized NumPy inputs, nonnegative time/duration checks, Gaussian cage cumulants, measured second/fourth mark moments, and a normalized discrete mark characteristic function for scattering.
- [ ] **Step 4: Verify GREEN** with focused finite-flight tests and the full `tests.test_renewal_cage` module.
- [ ] **Step 5: Commit** with message `add finite-duration cage flight observables`.

### Task 3: Event-space Green-Kubo audit

**Files:**
- Modify: `src/renewal_cage.py`
- Modify: `tests/test_renewal_cage.py`

**Interfaces:**
- Produces: `event_space_correlated_diffusion(event_rate: float, jump_squared_mean: float, jump_dot_correlations: Sequence[float], dimension: int = 3) -> float`.

- [ ] **Step 1: Write failing tests** for the independent-jump limit and the fixed `T=0.45` values `C1/q=-0.05485`, `C2/q=-0.05191`, requiring the corrected diffusion `2.81491e-5` to lie within 2% of held-out `2.785581176e-5`.
- [ ] **Step 2: Verify RED** with the focused Green-Kubo test; expect an import error.
- [ ] **Step 3: Implement the formula** with finite-value, positivity, and dimensional validation, rejecting a nonpositive corrected jump bracket.
- [ ] **Step 4: Verify GREEN** with focused and full unit-test modules.
- [ ] **Step 5: Commit** with message `add event-space correlated diffusion diagnostic`.

### Task 4: Fixed real-data closure artifact and manuscript claim

**Files:**
- Modify: `scripts/generate_renewal_cage_results.py`
- Create: `data/renewal_cage_stationary_finite_flight.csv`
- Create: `figures/renewal_cage_stationary_finite_flight.svg`
- Create: `figures/renewal_cage_stationary_finite_flight.pdf`
- Modify: `tests/test_arxiv_package.py`
- Modify: `scripts/build_arxiv_package.py`
- Modify: `paper/main.tex`
- Modify: `paper/references.bib`

**Interfaces:**
- Consumes: all Task 1-3 functions and fixed derived trajectory values.
- Produces: a two-temperature calibration/held-out audit with source DOI, extraction rule, uncertainty/threshold fields, diffusion, alpha, NGP, overlap-proxy, finite-flight, and Green-Kubo verdicts.

- [ ] **Step 1: Write failing package tests** requiring both temperatures, `thermodynamic_claim_allowed=0`, corrected `T=0.45` diffusion relative error below 0.02, finite-flight `T=0.58` NGP error below the instantaneous error, and all three artifacts in the package manifest.
- [ ] **Step 2: Verify RED** with the focused package test; expect the CSV to be absent.
- [ ] **Step 3: Extend the generator** with fixed, provenance-labelled derived inputs and deterministic CSV/SVG/PDF writers; include explicit failure columns for the uncorrected cold diffusion and stationary-gamma count-variance mismatch.
- [ ] **Step 4: Regenerate artifacts** using `python scripts/generate_renewal_cage_results.py`; expect all three files to be created deterministically.
- [ ] **Step 5: Update manuscript and bibliography** to state the stationary residual-life identity, finite-flight correction, cold event anti-correlation result, held-out metrics, and limitations. Correct the Hedges et al. DOI to `10.1063/1.2803062`.
- [ ] **Step 6: Verify package tests** with `python -m unittest tests.test_arxiv_package -v`; expect all tests to pass.
- [ ] **Step 7: Commit** artifact, generator, test, package, and paper changes with message `validate stationary finite-flight closure on KA trajectories`.

### Task 5: Full verification

**Files:**
- Verify all modified and generated files.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: reproducible test and artifact evidence.

- [ ] **Step 1: Run the complete suite** with `python -m unittest discover -s tests -v`; expect zero failures.
- [ ] **Step 2: Re-run generation and check cleanliness** with `python scripts/generate_renewal_cage_results.py && git status --short`; expect no unexpected nondeterministic changes.
- [ ] **Step 3: Build the arXiv package** using the repository's package command and verify the manifest contains the new CSV, SVG, and PDF.
- [ ] **Step 4: Render and inspect the generated PDF** with Poppler, checking nonblank axes, legible labels, and no clipping.
- [ ] **Step 5: Review claims** with `rg -n "explain(s|ed)? all|full glass transition theory|thermodynamic_claim_allowed" paper data/renewal_cage_stationary_finite_flight.csv`; expect no overclaim and zeros in all new thermodynamic flags.
- [ ] **Step 6: Commit any verification-only corrections** with message `tighten stationary finite-flight verification`.
