# Gamma Variance-Mixture Langevin Closure Plan

**Goal:** Test whether one positive scalar mobility Langevin field explains the
fixed-MSD displacement-shape residual, and identify the first observable where
it fails.

## Constraints

- Do not change source tolerances or segment-splice verdicts.
- Treat heldout MSD, NGP, and optional `Fs(k=2)` as diagnostic inputs.
- Use no macro fit and make no blind, spatial, microscopic-closure, or
  thermodynamic claim.
- Validate the squared-OU Langevin formula by simulation independently of KA.

## Tasks

- [x] Add RED unit tests for gamma/shifted-gamma formulas, Gaussian limit,
  bracketed cage inversion, malformed inputs, and exact provenance.
- [x] Implement validated real-data rows and fail-closed two-temperature gate.
- [x] Add RED simulation tests, then implement exact stationary OU updates and
  deterministic finite-`tau_D` validation.
- [x] Generate deterministic CSV/SVG artifacts and derive the microscopic
  formula in a concise research note.
- [x] Add arXiv recomputation/claim-boundary coverage and README result text.
- [x] Run focused tests, full suite, result generator, arXiv builder, visual
  inspection, and `git diff --check`.
- [ ] Commit, push, create a ready PR, and inspect CI.
