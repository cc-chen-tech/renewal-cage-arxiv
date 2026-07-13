# C3-Switched KA Tangent Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Follow test-first development and do not start the full ensemble before the force-parity gate passes.

**Goal:** Replace the non-differentiable KA cutoff by a C3 septic switch in both LAMMPS and the exact Python generator, then test the microscopic common-noise tangent covariance over a complete 8-member horizon.

**Architecture:** A radial-potential layer supplies consistent energy derivatives to the existing many-particle force-generator code. The response protocol selects either the legacy hard cutoff or the new C3 switch explicitly. A pinned Lepton-enabled LAMMPS binary realizes the same energy expression, a dedicated runner prepares a switched-potential parent restart, and the existing matched-response/covariance pipeline scores the full uncensored horizon.

**Tech Stack:** Python 3, NumPy, `unittest`, CMake, serial LAMMPS `stable_22Jul2025_update4` with LEPTON, existing KA restart and generator-response diagnostics.

## Global Constraints

- Preserve all legacy hard-cutoff behavior and tests.
- Preserve `thermodynamic_claim_allowed = 0` in every new manifest and table.
- Use explicit protocol names: `ka_lj_cut` and `ka_lj_c3_switch`.
- Use `r_on=2.0 sigma`, `r_c=2.5 sigma` for AA, AB, and BB.
- Never compare LAMMPS switched trajectories with hard-cutoff Python generators.
- Require force parity before equilibration and equilibration before response runs.
- Keep generated LAMMPS builds, raw trajectories, and restarts under `tmp/`.
- Stage only intentional source, tests, plans, documentation, and compact result tables.

---

### Task 1: Analytic C3 Radial Potential

**Files:**
- Modify: `src/ka_local_cage.py`
- Modify: `tests/test_ka_replicates.py`

- [ ] Add a failing test for a public `ka_lj_radial_derivatives(r, epsilon, sigma, protocol)` API returning `U,U',U'',U'''`.
- [ ] Require exact LJ derivatives below `2.0 sigma`, zero values at and above `2.5 sigma`, and continuity of values through third derivative on both switch endpoints.
- [ ] Run the focused test and verify RED because the API is absent.
- [ ] Implement the septic switch and analytic product-rule derivatives with explicit input validation.
- [ ] Add a failing finite-difference test at interior LJ and switching radii for derivative orders one through three.
- [ ] Run RED, adjust only production derivatives, then verify both endpoint and finite-difference tests GREEN.
- [ ] Run all existing KA force/Hessian/generator unit tests to prove the hard-cutoff default did not change.
- [ ] Commit: `add C3-switched KA radial derivatives`.

### Task 2: Switched Many-Particle Force Generator

**Files:**
- Modify: `src/ka_local_cage.py`
- Modify: `src/ka_generator_response.py`
- Modify: `tests/test_ka_replicates.py`

- [ ] Add a failing two-particle test requiring switched pair force and Hessian to match radial derivatives in the LJ, switching, and near-cutoff regions.
- [ ] Run RED because microscopic geometry does not accept a protocol.
- [ ] Parameterize the internal target-pair geometry by explicit protocol while preserving `ka_lj_cut` defaults.
- [ ] Add failing tests that `LF` equals the centered trajectory derivative of switched force and `L2F` equals the drift derivative away from switch endpoints.
- [ ] Run RED, then thread the protocol through force, Hessian, `LF`, `L2F`, and extraction APIs.
- [ ] Require extraction output to store `potential_protocol` and switch parameters.
- [ ] Run focused and neighboring generator tests GREEN.
- [ ] Commit: `extend exact generator to C3-switched KA`.

### Task 3: Lepton Expression and Tiny-System Parity

**Files:**
- Modify: `src/ka_generator_response.py`
- Create: `scripts/build_lammps_lepton.py`
- Create: `scripts/check_ka_c3_lepton_parity.py`
- Modify: `tests/test_ka_replicates.py`

- [ ] Add a failing test for `ka_c3_lepton_expression(epsilon, sigma)` and switched LAMMPS pair commands for all three KA pair types.
- [ ] Require pair-specific cutoffs `2.5`, `2.0`, and `2.2` and no table interpolation.
- [ ] Run RED, then implement the expression builder and protocol command block.
- [ ] Add a failing `--help` test for a reproducible Lepton build helper accepting source and build directories.
- [ ] Implement CMake configure/build with `PKG_LEPTON=ON`, serial MPI stubs, and an atomic build manifest containing source and binary hashes.
- [ ] Build the pinned executable and verify `lmp -h` lists `lepton`.
- [ ] Add a parity CLI that runs isolated AA, AB, and BB dimers at radii below, inside, and near the switching boundary and compares LAMMPS forces with Python forces.
- [ ] Require maximum relative force error below `1e-10` and absolute near-zero error below `1e-11`.
- [ ] Run parity and retain a compact CSV summary.
- [ ] Commit: `verify Lepton parity for C3-switched KA`.

### Task 4: Switched-Parent Equilibration

**Files:**
- Create: `scripts/prepare_ka_c3_switched_parent.py`
- Modify: `tests/test_ka_replicates.py`

- [ ] Add a failing `--help` test requiring source restart, Lepton binary, output directory, temperature, friction, seed, timestep, and duration controls.
- [ ] Implement an input that reads the hard-cutoff restart, replaces pair style/coefficients, performs `run 0`, then runs NVE plus Langevin and writes a switched restart.
- [ ] Write an atomic manifest with source/binary/output hashes, exact pair expressions, thermodynamic trace path, and claim boundary.
- [ ] Run a `0.1 tau` canary; require finite temperature, energy, pressure, and output restart.
- [ ] Run the preregistered `10 tau` equilibration at `T=0.58`, `gamma=1`, `dt=0.001`.
- [ ] Check temperature and energy stationarity over the final half without promoting this to an equilibrium glass claim.
- [ ] Commit source and compact equilibration summary only.

### Task 5: Protocol-Aware Matched Response Runner

**Files:**
- Modify: `scripts/run_ka_generator_response.py`
- Modify: `src/ka_generator_response.py`
- Modify: `tests/test_ka_replicates.py`

- [ ] Add a failing CLI test for `--potential-protocol` and a failing input test that switched runs emit Lepton pair commands after `read_restart`.
- [ ] Run RED, then implement protocol selection and reject incompatible restart/protocol combinations unless explicitly switching during parent preparation.
- [ ] Store protocol, switch window, Lepton binary hash, and parent hash in every NPZ and manifest row.
- [ ] Extend extraction verification to require protocol metadata and full pair Hessians.
- [ ] Run a one-member/two-epsilon canary for `0.2 tau`, saved every `0.001 tau`.
- [ ] Require 201 frames per path, 8 paths, finite generator arrays, and complete raw-to-NPZ verification.
- [ ] Commit: `run matched responses for C3-switched KA`.

### Task 6: Full-Horizon Covariance Analysis

**Files:**
- Modify: `scripts/analyze_ka_tangent_noise_covariance.py`
- Create: `scripts/analyze_ka_c3_tangent_control.py`
- Modify: `tests/test_ka_replicates.py`

- [ ] Add a failing test for a `--require-full-horizon` covariance option that rejects any right-censored interval.
- [ ] Implement protocol-aware validation: hard cutoff may right-censor, C3 switch must retain every interval.
- [ ] Add a control-summary CLI that combines resolution, cross-epsilon, tangent identities, covariance calibration, and gate verdicts.
- [ ] Run the 8-member, two-epsilon ensemble with the same seeds as the hard-cutoff experiment.
- [ ] Produce 32 compressed paths and verify `320/320` intervals per epsilon are eligible.
- [ ] Run covariance and resolution analyses at strides `1,2,5`.
- [ ] Evaluate all preregistered gates without parameter fitting or selective member removal.
- [ ] Commit scripts and compact summary tables.

### Task 7: Physical Fidelity and Scientific Boundary

**Files:**
- Create: `scripts/compare_ka_c3_physical_fidelity.py`
- Create: `docs/microscopic-c3-switched-tangent-control.md`
- Modify: `tests/test_arxiv_package.py`

- [ ] Add a failing document gate requiring protocol, derivative order, parity result, equilibration limitation, covariance verdict, and `thermodynamic_claim_allowed = 0`.
- [ ] Implement a compact comparison of hard-cutoff and switched-parent pair-distance distribution, force norm, cage curvature, and short-time tagged MSD on matched analysis windows.
- [ ] Report differences with uncertainty where multiple members exist; do not call a `10 tau` parent equilibrium.
- [ ] State whether the full-horizon tangent covariance passed, failed, or remained underpowered and identify the next microscopic projection step.
- [ ] Run full unit tests, `py_compile`, all data-generating CLIs used by the document, and `git diff --check`.
- [ ] Commit: `document C3-switched microscopic tangent control`.

## Completion Evidence

This plan is complete only when the repository contains:

- analytic and finite-difference evidence for C3 radial derivatives;
- direct LAMMPS/Python force parity for every KA pair type;
- a hashed switched-parent preparation manifest;
- a complete 8-member, two-epsilon, full-horizon response manifest;
- parameter-free covariance and resolution tables with explicit gate verdicts;
- a physical-fidelity comparison and a claim-limited scientific conclusion;
- a fresh green full test suite.
