# Second-Generator Krylov Response Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the exact microscopic `L2 F` observable extends a C3 KA tagged-response closure beyond the existing `(x,v,F,L F)` model.

**Architecture:** A focused `ka_generator_krylov` module owns the 15-state assembly, weak-form last-row fit, constrained propagation, and residual diagnostics. A dedicated analysis script validates a one-tau C3 response manifest and compares the 12-state baseline, 15-state constrained model, and free-transition null under identical leave-one-member-out and cross-epsilon gates.

**Tech Stack:** Python 3, NumPy, `unittest`, existing KA C3/response utilities, pinned Lepton-enabled LAMMPS.

## Global Constraints

- Use `T=0.58`, `gamma=1`, target id `821`, integrator step `0.001 tau`, duration `1.0 tau`, and saved interval `0.005 tau`.
- Use `epsilon=0.001,0.002` and the eight existing C3 seed pairs.
- Fit only microscopic response coordinates; never fit diffusion, NGP, scattering, or event statistics.
- Require `potential_protocol=ka_lj_c3_switch` and verified manifest hashes.
- Keep `autonomous_stochastic_single_particle_gle_allowed`, `event_clock_claim_allowed`, `kramers_escape_claim_allowed`, and `thermodynamic_claim_allowed` false.
- Stage only named files because the research worktree contains unrelated untracked experiment artifacts.

---

### Task 1: Weak-Form Second-Generator Core

**Files:**
- Create: `src/ka_generator_krylov.py`
- Create: `tests/test_ka_generator_krylov.py`

**Interfaces:**
- Consumes: 12-state arrays in `(x,v,F,L F)` order and aligned `L2 F` arrays.
- Produces: `assemble_second_generator_state`, `fit_second_generator_constrained_response`, `fit_free_second_generator_transition`, `propagate_linear_response`, and `second_generator_residual_diagnostic`.

- [ ] **Step 1: Write failing state-assembly and validation tests**

```python
def test_assemble_second_generator_state_preserves_microscopic_order():
    state = np.arange(2 * 7 * 12, dtype=float).reshape(2, 7, 12)
    second = np.arange(2 * 7 * 3, dtype=float).reshape(2, 7, 3)
    result = assemble_second_generator_state(state, second)
    np.testing.assert_array_equal(result[..., :12], state)
    np.testing.assert_array_equal(result[..., 12:15], second)

def test_assemble_second_generator_state_rejects_misaligned_second_mode():
    with self.assertRaisesRegex(ValueError, "second_force_response"):
        assemble_second_generator_state(np.zeros((7, 12)), np.zeros((6, 3)))
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m unittest tests.test_ka_generator_krylov -v`

Expected: import failure for `ka_generator_krylov`.

- [ ] **Step 3: Implement state assembly and shared validation**

```python
def assemble_second_generator_state(state_response, second_force_response):
    state = np.asarray(state_response, dtype=float)
    second = np.asarray(second_force_response, dtype=float)
    if state.ndim not in (2, 3) or state.shape[-1] != 12:
        raise ValueError("state_response must end in the 12 microscopic response coordinates")
    if second.shape != state.shape[:-1] + (3,):
        raise ValueError("second_force_response must align with state_response")
    if np.any(~np.isfinite(state)) or np.any(~np.isfinite(second)):
        raise ValueError("response coordinates must be finite")
    return np.concatenate((state, second), axis=-1)
```

- [ ] **Step 4: Add a synthetic exact 15-state weak-form recovery test**

Construct a stable block generator with the first four microscopic rows fixed,
generate eight independent initial responses, and assert:

```python
fit = fit_second_generator_constrained_response(
    states,
    frame_time=0.005,
    friction=1.0,
    fit_frames=41,
)
np.testing.assert_allclose(fit["fitted_third_generator_block"], exact_block, rtol=2e-3, atol=2e-3)
self.assertLess(fit["heldout_position_relative_l2_error"], 2e-3)
self.assertLessEqual(fit["spectral_radius"], 1.0 + 1e-6)
```

- [ ] **Step 5: Implement the weak-form constrained fit**

Use midpoint rows and integrated `L2 F` increments:

```python
source = 0.5 * (training[:, 1:] + training[:, :-1])
target = (training[:, 1:, 12:15] - training[:, :-1, 12:15]) / frame_time
scale = np.sqrt(np.mean(source.reshape(-1, 15) ** 2, axis=0))
coefficient, _, rank, _ = np.linalg.lstsq(source.reshape(-1, 15) / scale, target.reshape(-1, 3), rcond=None)
third_block = coefficient.T / scale[None, :]
```

Build the continuous `15 x 15` generator with exact rows
`xdot=v`, `vdot=F-gamma*v`, `Fdot=LF`, `LFdot=L2F`, and the fitted final block.
Use a fourth-order matrix polynomial to construct the saved-step transition.

- [ ] **Step 6: Add free-transition, propagation, and residual tests**

Verify that the free transition has shape `(15,15)`, propagation preserves the
initial state, and the residual diagnostic reports near-zero residual and
state correlation for the exact synthetic system.

- [ ] **Step 7: Implement the control and diagnostic functions**

`fit_free_second_generator_transition` fits a scaled discrete map on the same
training frames. `second_generator_residual_diagnostic` evaluates

```text
rho_n = H2_(n+1) - H2_n - dt B (Z2_(n+1)+Z2_n)/2
```

and returns relative residual L2 plus the maximum absolute finite
residual-state correlation.

- [ ] **Step 8: Run focused tests and commit**

Run: `python -m unittest tests.test_ka_generator_krylov -v`

Expected: all focused tests pass.

Commit:

```bash
git add src/ka_generator_krylov.py tests/test_ka_generator_krylov.py
git commit -m "add second-generator Krylov response core"
```

---

### Task 2: Held-Out Response Analysis

**Files:**
- Create: `scripts/analyze_ka_second_generator_response.py`
- Modify: `tests/test_ka_generator_krylov.py`

**Interfaces:**
- Consumes: a `run_ka_generator_response.py` manifest and reduced NPZ paths.
- Produces: `<prefix>_summary.csv` and `<prefix>_curve.csv` with model, fold, epsilon, fit horizon, response horizon, stability, linearity, residual, and claim-gate fields.

- [ ] **Step 1: Write a failing CLI contract test**

```python
completed = subprocess.run(
    [sys.executable, str(ROOT / "scripts/analyze_ka_second_generator_response.py"), "--help"],
    check=True,
    capture_output=True,
    text=True,
)
self.assertIn("--fit-times", completed.stdout)
self.assertIn("--horizons", completed.stdout)
self.assertIn("--linearity-tolerance", completed.stdout)
```

- [ ] **Step 2: Run the CLI test and confirm RED**

Run: `python -m unittest tests.test_ka_generator_krylov.SecondGeneratorKrylovTests.test_second_generator_cli_exposes_preregistered_gates -v`

Expected: failure because the script does not exist.

- [ ] **Step 3: Implement manifest and response loading**

Require:

```python
manifest["potential_protocol"] == "ka_lj_c3_switch"
manifest["thermodynamic_claim_allowed"] is False
manifest["member_count"] == expected_member_count
manifest["epsilons"] == [0.001, 0.002]
manifest["saved_frame_interval_tau"] == 0.005
manifest["duration_tau"] == 1.0
```

Verify every path SHA256, common seeds, path protocol, and 201-frame grid.
Use `matched_generator_response` to create each central tangent response and
`assemble_second_generator_state` to append `delta L2 F`.

- [ ] **Step 4: Implement identical leave-one-member-out comparisons**

For fit times `0.05,0.10,0.20 tau`, fit the other seven `epsilon=0.001`
members with the existing first-generator fit and both new 15-state fits.
Propagate from each held initial state and score position response at
`0.20,0.50,1.00 tau` for both epsilon values.

Use only the `0.20 tau` fit for boolean claim gates. Treat the shorter fits as
rank and convergence diagnostics, never as selectable alternatives.

- [ ] **Step 5: Implement linearity and claim gates**

Mark a member/horizon identified only when the cross-epsilon position mismatch
is at most `0.02`. Emit per-fold and aggregate rows with:

```text
position_relative_l2_error
paired_improvement_fraction
transition_stable
maximum_held_residual_state_correlation
identified_fold_count
second_generator_response_allowed
one_tau_generator_response_allowed
autonomous_stochastic_single_particle_gle_allowed=0
event_clock_claim_allowed=0
kramers_escape_claim_allowed=0
thermodynamic_claim_allowed=0
```

- [ ] **Step 6: Add a synthetic end-to-end fixture test**

Create a temporary two-member, two-epsilon manifest from exact synthetic
15-state NPZ paths, run the CLI with `--expected-member-count 2`, and assert
that hashes are checked, all three models appear, and all forbidden claim
flags remain zero.

- [ ] **Step 7: Run focused tests and commit**

Run: `python -m unittest tests.test_ka_generator_krylov -v`

Expected: all tests pass.

Commit:

```bash
git add scripts/analyze_ka_second_generator_response.py tests/test_ka_generator_krylov.py
git commit -m "add second-generator heldout response analysis"
```

---

### Task 3: Generate The One-Tau C3 Response Panel

**Files:**
- Create under untracked scratch: `tmp/generator_response_c3_krylov8_T058/`

**Interfaces:**
- Consumes: verified C3 parent restart and pinned Lepton-enabled LAMMPS binary.
- Produces: 32 reduced NPZ records plus a hashed manifest; raw trajectories are deleted after each verified extraction.

- [ ] **Step 1: Record the pre-run disk and binary gates**

Run `df -h .`, verify at least `0.8 GiB` available, and verify both the pinned
LAMMPS executable and C3 parent restart exist.

- [ ] **Step 2: Run the preregistered panel**

```bash
python scripts/run_ka_generator_response.py \
  --lammps-binary tmp/toolchains/lammps-22Jul2025_update4/build-lepton/lmp \
  --parent-restart tmp/ka_c3_parent_T058_tau10/equilibrated_c3.restart \
  --output-directory tmp/generator_response_c3_krylov8_T058 \
  --velocity-seeds 82101 82139 82157 82181 82203 82239 82277 82301 \
  --langevin-seeds 83101 83139 83157 83181 83203 83239 83277 83301 \
  --epsilons 0.001 0.002 \
  --potential-protocol ka_lj_c3_switch \
  --target-id 821 --temperature 0.58 --friction 1.0 \
  --integration-time-step 0.001 --duration 1.0 --dump-interval 0.005 \
  --directional-step 1e-5
```

Expected: exit zero, 32 manifest records, and no retained raw trajectory.

- [ ] **Step 3: Verify the generated manifest independently**

Check every SHA256 and require `(201,3)` arrays for `position`, `velocity`,
`force`, `force_generator`, and `second_force_generator`, with
`potential_protocol=ka_lj_c3_switch` in every NPZ.

---

### Task 4: Run The Experiment And Freeze The Verdict

**Files:**
- Create: `data/renewal_cage_ka_second_generator_response_T058_summary.csv`
- Create: `data/renewal_cage_ka_second_generator_response_T058_curve.csv`
- Create: `docs/microscopic-second-generator-krylov-response.md`
- Modify: `tests/test_arxiv_package.py`

**Interfaces:**
- Consumes: Task 3 manifest and Task 2 analysis CLI.
- Produces: tracked numerical evidence, formula-level interpretation, and package claim guards.

- [ ] **Step 1: Run the analysis**

```bash
python scripts/analyze_ka_second_generator_response.py \
  tmp/generator_response_c3_krylov8_T058/manifest.json \
  --output-prefix data/renewal_cage_ka_second_generator_response_T058 \
  --fit-times 0.05 0.10 0.20 --horizons 0.20 0.50 1.00 \
  --linearity-tolerance 0.02 --expected-member-count 8
```

Expected: exit zero and both CSV files are nonempty.

- [ ] **Step 2: Write a failing package gate**

Add `test_second_generator_krylov_response_is_complete_and_claim_limited`.
Require the design, script, report, summary, and curve artifacts; verify at
least one row for each model and assert every forbidden claim flag is zero.

- [ ] **Step 3: Run the package test and confirm RED**

Run: `python -m unittest tests.test_arxiv_package.ArxivPackageTests.test_second_generator_krylov_response_is_complete_and_claim_limited -v`

Expected: failure because the report is absent.

- [ ] **Step 4: Write the scientific report from the authoritative CSV**

Document the exact hierarchy, weak-form fit, C3 protocol, fold counts,
baseline/second/free errors, stability, cross-epsilon identification, residual
correlations, boolean verdicts, and the evidence-based next branch.

- [ ] **Step 5: Run focused verification and commit data/report**

```bash
python -m unittest tests.test_ka_generator_krylov tests.test_arxiv_package.ArxivPackageTests.test_second_generator_krylov_response_is_complete_and_claim_limited -v
python -m py_compile src/ka_generator_krylov.py scripts/analyze_ka_second_generator_response.py
git diff --check
```

Commit only named files:

```bash
git add data/renewal_cage_ka_second_generator_response_T058_summary.csv \
  data/renewal_cage_ka_second_generator_response_T058_curve.csv \
  docs/microscopic-second-generator-krylov-response.md \
  tests/test_arxiv_package.py
git commit -m "test second-generator microscopic response closure"
```

---

### Task 5: Full Verification And Research Checkpoint

**Files:**
- No new files.

**Interfaces:**
- Consumes: all preceding commits.
- Produces: a clean tracked research checkpoint and the evidence-based next branch.

- [ ] **Step 1: Run the full test suite**

Run: `python -m unittest discover -s tests -v`

Expected: zero failures and zero errors.

- [ ] **Step 2: Verify tracked scope and reproducibility commands**

```bash
git status -sb --untracked-files=no
git diff --check HEAD~4..HEAD
git log --oneline -5
```

Expected: no tracked modifications and only the planned commits at the branch tip.

- [ ] **Step 3: Update the active research plan**

If the second mode passes, schedule stochastic long-trajectory propagation
with the measured tangent covariance. If it fails or only partially improves,
schedule time-lagged microscopic force modes followed by a position-dependent
kernel test. Do not mark the overall single-particle microscopic goal complete.
