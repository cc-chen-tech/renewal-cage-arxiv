# Variance-Mixture Shape Quotient Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parameter-free, fixed-MSD gate testing whether the NGP residual closes multi-k self-scattering through a family-robust positive variance mixture.

**Architecture:** A standalone summarizer consumes the committed transport-clock row and gate tables, evaluates one fourth-order and two analytic all-order quotient corrections, then writes deterministic rows, a claim-limited gate, and one SVG. Existing transport-clock code and artifacts remain unchanged.

**Tech Stack:** Python 3.12 standard library, CSV, SVG, `unittest`.

## Global Constraints

- Held-out MSD and NGP are diagnostic inputs, so blind prediction remains zero.
- No interpolation outside committed calibration-MSD support.
- The 0.03 self-scattering tolerance is frozen.
- Gamma and inverse-Gaussian resummations must both pass before family-robust closure is allowed.
- Static-environment, finite-exchange, microscopic, spatial, and thermodynamic claims remain zero.
- `T=0.58` remains a stationarity-unresolved canary.

---

### Task 1: Analytic quotient kernel

**Files:**
- Create: `scripts/summarize_ka_variance_mixture_shape_quotient.py`
- Create: `tests/test_ka_variance_mixture_shape_quotient.py`

**Interfaces:**
- Consumes: committed transport-clock quotient rows.
- Produces: `mixture_log_scattering(alpha, x, family)` and `compute_shape_quotient_rows(rows)`.

- [x] Write failing tests for Gaussian limits, common fourth-order expansion, invalid domains, exact support filtering, and zero claim flags.
- [x] Run the focused test and confirm the missing module failure.
- [x] Implement the gamma, inverse-Gaussian, and fourth-order quotient kernels.
- [x] Implement strict row-grid validation and row scoring.
- [x] Run the focused tests to green.

### Task 2: Gate and deterministic artifacts

**Files:**
- Modify: `scripts/summarize_ka_variance_mixture_shape_quotient.py`
- Modify: `tests/test_ka_variance_mixture_shape_quotient.py`
- Create: `data/renewal_cage_ka_variance_mixture_shape_quotient_rows.csv`
- Create: `data/renewal_cage_ka_variance_mixture_shape_quotient_gate.csv`
- Create: `figures/renewal_cage_ka_variance_mixture_shape_quotient.svg`

**Interfaces:**
- Consumes: scored rows and the source transport-clock gate.
- Produces: `classify_shape_quotient_gate(rows, source_gate)` and deterministic CLI outputs.

- [x] Write failing tests for exact source-gate validation, family-robust pass logic, canary handling, and byte-deterministic CLI output.
- [x] Run the focused test and confirm the new gate tests fail.
- [x] Implement classification, canonical CSV serialization, SVG rendering, and CLI wiring.
- [x] Generate the three committed artifacts from source tables.
- [x] Recompute artifacts twice and verify byte identity.
- [x] Run focused tests to green.

### Task 3: Package integration and verification

**Files:**
- Modify: `tests/test_arxiv_package.py`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-18-variance-mixture-shape-quotient.md`

**Interfaces:**
- Consumes: committed rows, gate, and SVG.
- Produces: package-level recomputation and claim-boundary coverage.

- [x] Add a failing package test that recomputes all artifacts and checks the low-temperature result, high-temperature canary, and closed claims.
- [x] Add the three artifacts to the reproducibility inventory and a concise README result row.
- [x] Run focused tests, result generation, arXiv package build, `git diff --check`, and the complete Python 3.12 suite.
- [x] Render the SVG and inspect labels, clipping, and claim-boundary text.
- [x] Mark plan steps complete only after the corresponding verification passes.
- [ ] Publish an isolated remote branch and PR, wait for every CI check, inspect review threads, and merge only if clean.
