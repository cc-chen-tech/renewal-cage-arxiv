# L2p Conditional-Diffusion Gate Implementation Plan

> Execute this plan in the isolated `codex/projected-noise-qz-gate` worktree.
> Do not modify or switch the shared checkout.

**Goal:** Derive a probe-converged microscopic estimate of
`Q_c=2 gamma T grad_V(L2p) grad_V(L2p)^T`, then test on held KA clones whether
that covariance explains the volatility memory and non-Gaussianity of the
existing `L2p` Markov-bath innovation.

**Architecture:** Extend the exact smooth-cage generator layer with a
temperature-independent velocity-directional derivative of `L2p`.  A cache CLI
evaluates nested Rademacher prefixes and step sensitivities on the existing
second-generator trajectories.  A separate analysis module reconstructs the
frozen Mori-40/VAR-16 white innovations, fits covariance models on training
clones only, and scores all held clones.  Reproducible CSV/SVG/docs are wired
into the standard generator and arXiv package tests with all broad claim flags
closed.

**Tech stack:** Python 3, NumPy, standard-library `unittest`, existing KA
linked-cell force generator and smooth-cage projection, LAMMPS
`stable_22Jul2025_update4` only if removed raw caches must be regenerated.

## Frozen Constraints

- Use the exact design in
  `docs/superpowers/specs/2026-07-19-l2p-conditional-diffusion-gate-design.md`.
- No event labels or macro observables enter any fit.
- Keep Mori memory 40, VAR order 16, four discovery clones, 64 fixed A targets,
  200 frames, and the existing target/probe seeds.
- Reuse identical inner Ito trace probes in every matched velocity perturbation.
- Exploit that the Ito trace term in `L2p` is independent of `V`: outer
  velocity derivatives evaluate `L2p` with `temperature=0` and no inner trace
  probes, avoiding a nested probe estimator without changing `grad_V(L2p)`.
- Raw simulations, restarts, builds, and NPZ caches remain under `tmp/` and are
  never packaged.
- Every output keeps autonomous, event-clock, Kramers, and thermodynamic claims
  at zero.

## Task 1: Velocity-Directional L2p Derivative

**Files**

- Modify: `src/ka_smooth_cage.py`
- Modify: `tests/test_ka_smooth_cage.py`

1. Add a failing synthetic test for a small smooth periodic configuration.
   Compare the new directional derivative against a direct centered difference
   of `smooth_cage_second_generator_batch` at nonzero temperature while holding
   fixed the same inner trace probes.
2. Add a failing test proving the derivative is unchanged when the inner trace
   probe count or temperature changes, because `grad_V(gamma T Delta_V b)=0`.
3. Implement `smooth_cage_l2p_velocity_directional_derivative_batch(...)`.
   Validate full-state shapes, finite positive step, target alignment, and
   direction shape.  Evaluate matched `V +/- h eta` with fixed `R,F,LF`,
   `temperature=0`, and `trace_probes=None`.
4. Add central-step convergence and zero-direction tests.
5. Run:

```bash
python -m unittest tests.test_ka_smooth_cage -v
```

## Task 2: Conditional Diffusion Estimator

**Files**

- Create: `src/ka_l2p_conditional_diffusion.py`
- Create: `tests/test_ka_l2p_conditional_diffusion.py`

1. Write failing tests for `nested_rademacher_diffusion_estimates` using a
   synthetic linear `c(V)=A V`.  Require nested prefixes to reproduce the
   direct probe sums exactly and converge toward `2 gamma T A A^T`.
2. Test deterministic seed behavior, prefix nesting, PSD symmetry, zero and
   malformed inputs, and preservation of `thermodynamic_claim_allowed=0`.
3. Implement fixed-seed Rademacher generation and prefix covariance assembly.
   Store directional responses so all prefix estimates are exact nested
   reductions rather than reruns.
4. Implement relative Frobenius convergence summaries with an eigenvalue floor
   used only for numerical diagnostics, never to alter the stored covariance.
5. Run:

```bash
python -m unittest tests.test_ka_l2p_conditional_diffusion -v
```

## Task 3: Microscopic Qc Cache CLI

**Files**

- Create: `scripts/cache_ka_l2p_conditional_diffusion.py`
- Modify: `tests/test_ka_l2p_conditional_diffusion.py`

1. Add a failing CLI help test for clone, drift, second-generator, output,
   prefix, velocity-step, sensitivity-step, seed, frame, and checkpoint
   controls.
2. Add a tiny fixture test that rejects trajectory/hash/target/protocol
   mismatches before expensive evaluation.
3. Implement resumable per-clone NPZ caching.  Load each full trajectory and
   paired drift/second-generator cache by SHA256.  Recompute sparse KA `F` and
   `LF` using the declared potential protocol.  Evaluate `Qc` prefixes
   `{4,8,16,32}` at `h=1e-5`, plus `h={3e-6,3e-5}` on the frozen sensitivity
   frame subset.
4. Store full per-frame/target `Qc` only for the primary 32-probe estimator;
   store prefix and step convergence errors as reduced arrays to control disk
   use.  Include completed-frame count, hashes, target indices, seeds,
   numerical controls, and zero claim flags.
5. Verify interrupted resume is byte-stable after completion.
6. Run:

```bash
python -m unittest tests.test_ka_l2p_conditional_diffusion -v
```

## Task 4: Frozen Mori/VAR Innovation Extraction

**Files**

- Modify: `scripts/analyze_ka_relative_second_generator_closure.py`
- Modify: `tests/test_ka_markov_bath.py`
- Modify: `tests/test_ka_relative_mori.py`

1. Add a failing test for a public helper that returns training and held
   normalized states, Mori operators, VAR coefficients, and aligned white
   innovations without running autonomous simulations.
2. Assert the helper reproduces the existing fold-level `L2p` residual
   correlation, squared-correlation, and kurtosis metrics exactly on a
   synthetic fixture.
3. Refactor `score_fold_model` to use the helper without changing existing CSV
   bytes or scientific verdicts.
4. Return alignment indices mapping each white innovation back to its source
   frame and target so `Qc(X_n)` cannot be shifted or leaked.
5. Run:

```bash
python -m unittest tests.test_ka_markov_bath tests.test_ka_relative_mori -v
python scripts/generate_renewal_cage_results.py
git diff --exit-code -- data/renewal_cage_ka_relative_second_generator_discovery200_T058_details.csv data/renewal_cage_ka_relative_second_generator_discovery200_T058_summary.csv
```

## Task 5: Conditional Covariance Models And Diagnostics

**Files**

- Modify: `src/ka_l2p_conditional_diffusion.py`
- Modify: `tests/test_ka_l2p_conditional_diffusion.py`

1. Write failing synthetic heteroscedastic tests for nonnegative
   likelihood fitting of `a Q_n + delta I`.  Recover known `a,delta` without
   using held samples.
2. Add constant-full, constant-isotropic, trace-only, tensor, and fixed-seed
   within-path permuted-tensor models.
3. Add whitening diagnostics: NLL, Mahalanobis/3, mean, covariance error,
   component kurtosis, correlations and squared correlations through lag 40,
   energy/trace correlation, floor fraction, and energy distance to a fixed
   Gaussian reference.
4. Test that the tensor model passes synthetic aligned heteroscedastic data,
   the constant and permuted nulls fail, and tensor orientation is not claimed
   for an isotropic scale-only fixture.
5. Implement replicate-first Student-t confidence intervals for four fold
   differences using the fixed `df=3` critical value; reject pooled-sample
   pseudo-replication.
6. Run:

```bash
python -m unittest tests.test_ka_l2p_conditional_diffusion -v
```

## Task 6: Real Held-Clone Gate CLI

**Files**

- Create: `scripts/analyze_ka_l2p_conditional_diffusion.py`
- Modify: `tests/test_ka_l2p_conditional_diffusion.py`

1. Add a failing end-to-end fixture test requiring complete four-fold output,
   model completeness, training-only fits, exact frame/target alignment,
   per-fold decision gates, probe convergence fields, and every broad claim
   flag equal to zero.
2. Implement the CLI by loading matched drift, `L2p`, and `Qc` caches; call the
   Task 4 extractor with frozen Mori/VAR controls; score all five covariance
   models; and classify the frozen decision tree.
3. Emit:
   - `<prefix>_details.csv` for held fold/model metrics;
   - `<prefix>_convergence.csv` for prefix and step sensitivity;
   - `<prefix>_summary.csv` for replicate-first comparisons and verdict.
4. Make CSV serialization deterministic across supported Python versions by
   canonicalizing reported floats at the output boundary.
5. Run focused tests twice and compare output bytes.

## Task 7: Restore Or Regenerate Microscopic Inputs

**Files**

- Raw outputs only under `tmp/`
- Modify scripts only if a recorded regeneration interface is missing, with
  new tests before implementation.

1. Search for the original four clone trajectories, decomposed drift caches,
   and second-generator caches.  Reuse only when manifests and SHA256 fields
   match the tracked checkpoint.
2. If absent, locate or rebuild the pinned LAMMPS source and binary.  Validate
   `lmp -h`, source revision, and binary hash.  Do not substitute an unrecorded
   engine.
3. Regenerate a separately named four-clone `T=0.58` protocol from the recorded
   restart/seeds with `x,v,f`, `dt=0.001`, saved interval `0.01 tau`, duration
   `10 tau`, and `thermodynamic_claim_allowed=false`.
4. Regenerate drift and `L2p` caches with the existing frozen commands.  If
   trajectory hashes differ from the historical checkpoint, label all outputs
   `rerun_validation` and do not compare byte-for-byte with historical raw
   caches.
5. Run the Qc cache first on two frames and one clone as a runtime/memory
   canary.  Estimate total cost before launching 4 x 200 frames.  Keep nested
   prefix arrays and checkpoints so interruption loses at most 25 frames.

## Task 8: Run Real Gate And Interpret Without Overclaim

**Files**

- Create: `data/renewal_cage_ka_l2p_conditional_diffusion_T058_details.csv`
- Create: `data/renewal_cage_ka_l2p_conditional_diffusion_T058_convergence.csv`
- Create: `data/renewal_cage_ka_l2p_conditional_diffusion_T058_summary.csv`
- Create: `figures/renewal_cage_ka_l2p_conditional_diffusion_T058.svg`
- Create: `docs/microscopic-l2p-conditional-diffusion.md`

1. Run all four held folds with frozen controls.
2. Independently recompute the verdict from CSV rows in a test; never trust a
   prewritten flag.
3. Generate an SVG showing per-fold NLL improvement and squared-residual memory
   for constant, trace-only, tensor, and permuted models.  Label normalization,
   tolerance lines, and any clipping explicitly.
4. Write the result document with numerical convergence, every fold, null
   failures, and the exact claim boundary.  A partial improvement is described
   as informative but insufficient, not as a Langevin closure.
5. Decide the next branch mechanically:
   - absolute tensor pass: autonomous `(c,Qc)` evolution;
   - informative but insufficient: derive `L3p`;
   - uninformative: nonlinear position-dependent memory/transient potential.

## Task 9: Package Integration And Verification

**Files**

- Modify: `scripts/generate_renewal_cage_results.py`
- Modify: `scripts/build_arxiv_package.py`
- Modify: `tests/test_arxiv_package.py`
- Modify: `README.md`
- Modify paper files only if the result changes the manuscript's existing
  microscopic claim boundary.

1. Add failing package tests requiring all three CSVs, deterministic SVG,
   result documentation, recomputed verdict, and closed broad claim flags.
2. Wire deterministic regeneration and arXiv packaging.
3. Run focused tests:

```bash
python -m unittest tests.test_ka_l2p_conditional_diffusion tests.test_ka_markov_bath tests.test_ka_relative_mori -v
python -m unittest tests.test_arxiv_package.ArxivPackageTests.test_l2p_conditional_diffusion_is_recomputed_and_claim_limited -v
```

4. Run full verification:

```bash
python -m unittest discover -s tests -v
python scripts/generate_renewal_cage_results.py
python scripts/build_arxiv_package.py
git diff --check
git status --short
```

5. Commit implementation and artifacts in reviewable stages, push
   `codex/projected-noise-qz-gate`, open a ready PR against the current PR-stack
   base if it remains open, and report local test count separately from remote
   CI state.
