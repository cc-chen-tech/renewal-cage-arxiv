# Relative Second Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the instantaneous smooth-cage `L2p` coordinate from the full KA Langevin generator and test whether it closes the held non-Gaussian Mori residual with Gaussian white driving.

**Architecture:** A sparse periodic KA Hessian-action supplies `LF=-H_U V`; a smooth-cage generator routine combines phase-space directional derivatives and an Ito trace estimator; a cache/analyzer freezes discovery controls before independent-clone validation.

**Tech Stack:** Python, NumPy, existing KA trajectory readers, `unittest`, discrete Mori and VAR bath modules.

## Global Constraints

- Use the C3-switched KA pair protocol and existing minimum-image convention.
- Fit no diffusion, scattering, event, or thermodynamic observable.
- Keep `thermodynamic_claim_allowed = 0` in every output.
- Use tests before production implementation and stage only explicit files.
- Keep discovery and independent validation clones separate.

---

### Task 1: Sparse KA Hessian Action

**Files:**
- Modify: `src/ka_local_cage.py`
- Modify: `tests/test_ka_replicates.py`

**Interfaces:**
- Consumes: positions, velocities, KA types, box lengths, selected particles.
- Produces: `ka_lj_sparse_force_generator_observables(...)` with `force` and `force_generator` arrays aligned to selected particles.

- [x] Write tests comparing sparse and dense force/`LF` on both protocols and species.
- [x] Run the focused tests and verify they fail because the sparse API is absent.
- [x] Implement periodic linked cells with a cell width no smaller than the maximum KA cutoff.
- [x] Run the focused tests and benchmark one 4096-particle frame.
- [x] Commit the tested sparse Hessian action.

### Task 2: Smooth-Cage `L2p`

**Files:**
- Modify: `src/ka_smooth_cage.py`
- Modify: `tests/test_ka_smooth_cage.py`

**Interfaces:**
- Consumes: full `(R,V,F,LF)`, fixed targets, finite-difference controls, and fixed trace probes.
- Produces: `smooth_cage_second_generator_batch(...)` with `relative_drift`, `second_relative_generator`, directional terms, and Ito trace diagnostics.

- [x] Write a failing harmonic-coordinate test with an analytic `L2p` value.
- [x] Write a failing small-KA-system test against the conditional short-time generator definition.
- [x] Implement centered position and velocity derivatives plus the antithetic Ito trace.
- [x] Run focused tests and a three-step convergence check.
- [x] Commit the tested second-generator observable.

### Task 3: Discovery Cache And Numerical Convergence

**Files:**
- Create: `scripts/cache_ka_relative_second_generator.py`
- Modify: `tests/test_ka_relative_mori.py`

**Interfaces:**
- Consumes: full-state clone trajectories and existing decomposed-drift caches.
- Produces: per-clone `relative_second_generator` caches with source hashes, step sensitivity, probe sensitivity, and exact target alignment.

- [ ] Write a failing CLI contract test for separate source, cache, and numerical controls.
- [ ] Implement resumable cache generation without retaining additional full trajectories.
- [ ] Run discovery subsets for probe counts `4,8,16,32` and three step sizes.
- [ ] Freeze the shortest converged controls before reading validation scores.
- [ ] Commit the cache protocol and reduced discovery diagnostics.

### Task 4: Held Gaussian-Closure Test

**Files:**
- Create: `scripts/analyze_ka_relative_second_generator_closure.py`
- Modify: `tests/test_ka_markov_bath.py`
- Modify: `tests/test_arxiv_package.py`

**Interfaces:**
- Consumes: fixed discovery and validation `L2p` caches.
- Produces: detail, summary, and correlation tables comparing `(u,p,Lp)` with `(u,p,Lp,L2p)` under Gaussian VAR driving.

- [ ] Write failing tests for squared-residual correlation and frozen validation controls.
- [ ] Implement the matched baseline/extension analysis with memory order `40` and VAR order `16`.
- [ ] Run discovery and two independent validation clones with disjoint Monte Carlo seed sets.
- [ ] Apply every preregistered gate without relaxing thresholds.
- [ ] Commit only the analyzer, tests, reduced tables, and claim-limited report.

### Task 5: Verification And Scientific Verdict

**Files:**
- Create: `docs/microscopic-relative-second-generator.md`

**Interfaces:**
- Consumes: committed discovery and validation tables.
- Produces: equations, convergence evidence, pass/fail verdict, and unchanged broad claim boundaries.

- [ ] Run focused tests, full `unittest` discovery, `py_compile`, and `git diff --check`.
- [ ] Verify the report numbers directly against the reduced CSV outputs.
- [ ] Record either the passing Gaussian closure or the precise failed gate.
- [ ] Commit the checkpoint without adding unrelated untracked artifacts.
