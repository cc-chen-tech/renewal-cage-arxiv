# Transport-Clock / Shape Quotient Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate calibration-to-heldout transport-clock drift from fixed-MSD shape residuals without adding a latent mechanism.

**Architecture:** A standard-library summarizer validates committed full-path, stationarity, and provenance tables, reconstructs the minimal manifest contract, computes replicate-resolved no-extrapolation MSD quotients, classifies a fail-closed T045 result and T058 canary, and writes deterministic CSV/SVG artifacts. Focused tests own interpolation, exact support, malformed-input rejection, classification, and artifact recomputation.

**Tech Stack:** Python 3.12 standard library, CSV, JSON, deterministic SVG, `unittest`.

## Global Constraints

- Use only committed segment-splice, stationarity, and provenance tables; run no simulation.
- Heldout MSD is a diagnostic input, so `blind_prediction_claim_allowed=0`.
- Use piecewise-linear interpolation in calibration MSD and prohibit extrapolation.
- Require at least 80% supported lags and one anchor-alpha point per replicate.
- Keep original tolerances: NGP `0.30`; every `Fs` `0.03`.
- T058 remains a canary; no cooling claim is allowed.
- Keep all microscopic, finite/static-exchange, spatial, and thermodynamic claims closed.

---

### Task 1: Quotient Kernel and Exact Input Contract

**Files:**
- Create: `tests/test_ka_transport_clock_shape_quotient.py`
- Create: `scripts/summarize_ka_transport_clock_shape_quotient.py`

**Interfaces:**
- `interpolate_no_extrapolation(xs, ys, target) -> float | None`
- `compute_transport_clock_shape_rows(source_rows, manifest, *, temperature) -> list[dict[str, object]]`

- [x] Write RED tests for interpolation, exact grids, support, full-path model agreement at `1e-12`, and malformed inputs.
- [x] Run `python -m unittest tests.test_ka_transport_clock_shape_quotient -v` and confirm missing-module failure.
- [x] Implement only the validated row computation and canonical CSV serializer.
- [x] Re-run focused tests to GREEN and `git diff --check`.

### Task 2: Fail-Closed Two-Temperature Classifier

**Files:**
- Modify: `tests/test_ka_transport_clock_shape_quotient.py`
- Modify: `scripts/summarize_ka_transport_clock_shape_quotient.py`

**Interfaces:**
- `classify_transport_clock_shape_gate(rows, stationarity_rows, manifests) -> list[dict[str, object]]`

- [x] Add RED truth-table tests for T045 separation, T058 canary, support failure, source stationarity, independent-parent status, and every closed claim flag.
- [x] Implement the classifier with same-time errors restricted to matched support and a separate full-grid diagnostic.
- [x] Verify T045 matched maxima `NGP=1.068606`, `Fs2=0.123587`, `Fs4=0.508749`, `Fs7.25=1.133592` within `5e-6`.
- [x] Verify T058 matched maxima `NGP=0.227337`, `Fs2=0.566487`, `Fs4=0.824309`, `Fs7.25=0.598179` within `5e-6`.

### Task 3: Deterministic Artifacts

**Files:**
- Modify: `tests/test_ka_transport_clock_shape_quotient.py`
- Modify: `scripts/summarize_ka_transport_clock_shape_quotient.py`
- Create: `data/renewal_cage_ka_transport_clock_shape_quotient_rows.csv`
- Create: `data/renewal_cage_ka_transport_clock_shape_quotient_gate.csv`
- Create: `figures/renewal_cage_ka_transport_clock_shape_quotient.svg`

**Interfaces:**
- CLI arguments for two row tables, two stationarity tables, two manifests, and three outputs.

- [x] Add RED tests for byte-deterministic CSV/SVG generation, finite SVG coordinates, and explicit diagnostic/canary labels.
- [x] Implement the CLI and a two-panel SVG showing same-time versus matched normalized error and replicate clock-drift signs.
- [x] Generate artifacts from committed inputs twice and byte-compare both runs.
- [x] Render the SVG and inspect clipping, labels, and claim boundaries.

### Task 4: Publication Gate and PR

**Files:**
- Modify: `README.md`
- Modify: `tests/test_arxiv_package.py`

**Interfaces:**
- The arXiv test reruns the summarizer and byte-compares all three committed artifacts.

- [x] Add a RED artifact test requiring exact values, manifests marked non-independent, T058 canary status, and all closed claims.
- [x] Add one concise README result paragraph; do not add manuscript prose or claim a cooling law.
- [ ] Run focused tests, full Python 3.12 suite, result generator, arXiv package builder, and `git diff --check`.
- [ ] Commit intentionally, push `codex/transport-clock-shape-quotient`, create a ready PR with `gh pr create`, and inspect remote CI.
