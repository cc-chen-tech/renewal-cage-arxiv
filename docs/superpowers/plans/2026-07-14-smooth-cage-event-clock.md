# Smooth Cage Microscopic Event Clock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the exact smooth-cage projected state `(u,p,G,b)` predicts nonrecrossing cage-center first escape across withheld many-particle KA parent configurations.

**Architecture:** Extend the smooth-cage module with force-injected projected states, invariant feature maps, and a grouped censored-exponential diagnostic. A low-disk reconstruction script reproduces only frame-zero velocities and forces from recorded seeds. A separate analysis script extracts fixed smooth-center labels from existing long trajectories, caches reduced arrays, runs parent holdouts and ablations, and writes claim-limited CSV evidence.

**Tech Stack:** Python 3, NumPy, existing LAMMPS binary and KA restart/trajectory artifacts, `unittest`, CSV/NPZ/JSON outputs.

## Global Constraints

- Use exactly five existing parents and the first eight completed clones per parent.
- Use exactly 64 A-particle targets selected once with seed `20260714`.
- Use smooth-center `p_hop=0.08`, half-window 8, and recrossing radius `sqrt(0.08)`.
- Use fixed censored-exponential `L2=1`; do not select regularization on held-out data.
- Never fit diffusion, NGP, scattering, event threshold, or thermodynamic observables.
- Keep `event_clock_claim_allowed`, `autonomous_single_particle_gle_claim_allowed`, `kramers_escape_claim_allowed`, and `thermodynamic_claim_allowed` false.
- Do not stage or delete unrelated untracked experiment artifacts.

---

### Task 1: Force-injected projected state and invariant features

**Files:**
- Modify: `src/ka_smooth_cage.py`
- Modify: `tests/test_ka_smooth_cage.py`

**Interfaces:**
- Consumes: `smooth_force_support_cage` and `smooth_cage_projected_observables`.
- Produces: optional `forces: np.ndarray | None` on `smooth_cage_projected_observables` and `smooth_cage_invariant_features(observable) -> dict[str, np.ndarray]`.

- [ ] **Step 1: Write failing force-injection and invariance tests**

```python
def test_projected_observables_accept_exact_microscopic_forces(self):
    inputs = self.microscopic_configuration()
    velocities = np.arange(12, dtype=float).reshape(4, 3) / 20.0
    known_forces = np.arange(12, dtype=float).reshape(4, 3) / 10.0
    direct = smooth_cage_projected_observables(
        **inputs, velocities=velocities, forces=known_forces,
        friction=1.0, temperature=0.58, directional_step=1e-5,
        potential_protocol="ka_lj_cut",
    )
    expected = np.einsum("nab,nb->a", direct["jacobian"], known_forces)
    np.testing.assert_allclose(direct["force_drift"], expected)

def test_smooth_cage_features_are_rotation_invariant(self):
    features = smooth_cage_invariant_features(observable)
    rotated = smooth_cage_invariant_features(rotate_observable(observable, rotation))
    np.testing.assert_allclose(features["full"], rotated["full"], atol=1e-12)
    self.assertEqual(features["geometry"].shape, (4,))
    self.assertEqual(features["kinematic"].shape, (6,))
    self.assertEqual(features["full"].shape, (9,))
```

- [ ] **Step 2: Run tests and verify they fail because the API is absent**

Run: `python3 -m unittest tests.test_ka_smooth_cage.SmoothCageTests.test_projected_observables_accept_exact_microscopic_forces tests.test_ka_smooth_cage.SmoothCageTests.test_smooth_cage_features_are_rotation_invariant -v`

Expected: import/signature failure for the new feature interface.

- [ ] **Step 3: Add optional force contraction and fixed invariant maps**

```python
def smooth_cage_invariant_features(observable):
    u = np.asarray(observable["relative_position"], dtype=float)
    p = np.asarray(observable["relative_velocity"], dtype=float)
    b = np.asarray(observable["projected_drift"], dtype=float)
    eigenvalue = np.linalg.eigvalsh(np.asarray(observable["jacobian_gram"], dtype=float))
    floor = np.finfo(float).tiny
    log_u2 = math.log(max(float(u @ u), floor))
    log_p2 = math.log(max(float(p @ p), floor))
    log_b2 = math.log(max(float(b @ b), floor))
    cos_up = float(u @ p) / math.sqrt(max(float(u @ u) * float(p @ p), floor))
    cos_ub = float(u @ b) / math.sqrt(max(float(u @ u) * float(b @ b), floor))
    cos_pb = float(p @ b) / math.sqrt(max(float(p @ p) * float(b @ b), floor))
    geometry = np.array([log_u2, *np.log(np.maximum(eigenvalue, floor))])
    kinematic = np.array([log_u2, log_p2, cos_up, *np.log(np.maximum(eigenvalue, floor))])
    full = np.array([log_u2, log_p2, cos_up, *np.log(np.maximum(eigenvalue, floor)), log_b2, cos_ub, cos_pb])
    return {"geometry": geometry, "kinematic": kinematic, "full": full}
```

Validate injected `forces` as finite and aligned; otherwise retain the existing analytic force calculation.

- [ ] **Step 4: Run focused tests**

Run: `python3 -m unittest tests.test_ka_smooth_cage -v`

Expected: all smooth-cage tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ka_smooth_cage.py tests/test_ka_smooth_cage.py
git commit -m "add microscopic smooth cage state features"
```

### Task 2: Grouped censored-exponential escape diagnostic

**Files:**
- Modify: `src/ka_smooth_cage.py`
- Modify: `tests/test_ka_smooth_cage.py`

**Interfaces:**
- Consumes: finite feature rows, first-passage/censor times, event flags, parent groups, common horizon.
- Produces: `grouped_exponential_escape_diagnostic(features, first_passage, escaped, groups, *, horizon, survival_times, l2_regularization=1.0) -> dict`.

- [ ] **Step 1: Write a failing synthetic leave-parent-out test**

```python
def test_grouped_exponential_escape_recovers_transferable_microscopic_rate(self):
    rng = np.random.default_rng(20260714)
    groups = np.repeat(np.arange(5), 800)
    feature = rng.normal(size=(len(groups), 2))
    rate = np.exp(-3.0 + 0.8 * feature[:, 0] - 0.5 * feature[:, 1])
    raw = rng.exponential(1.0 / rate)
    escaped = raw <= 20.0
    first = np.minimum(raw, 20.0)
    result = grouped_exponential_escape_diagnostic(
        feature, first, escaped, groups,
        horizon=20.0,
        survival_times=np.array([1, 2, 4, 8, 12, 16, 20]),
        l2_regularization=1.0,
    )
    self.assertGreater(result["mean_heldout_brier_skill"], 0.05)
    self.assertGreater(result["mean_heldout_log_likelihood_gain_per_observation"], 0.01)
    self.assertLess(result["maximum_heldout_survival_calibration_error"], 0.05)
```

- [ ] **Step 2: Run the test and verify missing-function failure**

Run: `python3 -m unittest tests.test_ka_smooth_cage.SmoothCageTests.test_grouped_exponential_escape_recovers_transferable_microscopic_rate -v`

Expected: import failure for `grouped_exponential_escape_diagnostic`.

- [ ] **Step 3: Implement analytic Newton fitting per held parent**

For every held group, standardize on training rows, prepend an intercept, initialize the intercept with `log(events/exposure)`, and iterate

```python
rate = np.exp(np.clip(x_train @ coefficient, -30.0, 30.0))
gradient = x_train.T @ (rate * train_time - train_event) + penalty @ coefficient
hessian = x_train.T @ ((rate * train_time)[:, None] * x_train) + penalty
step = np.linalg.solve(hessian, gradient)
coefficient -= step
```

until `max(abs(step)) < 1e-9` or 100 iterations. Return out-of-group rates, event probabilities, baselines, group scores, pooled means, and survival calibration. Reject singular/invalid input rather than silently dropping rows.

- [ ] **Step 4: Add validation tests for censoring and group boundaries**

```python
with self.assertRaisesRegex(ValueError, "at least two parent groups"):
    grouped_exponential_escape_diagnostic(
        np.ones((4, 1)), np.ones(4), np.ones(4, dtype=bool), np.zeros(4),
        horizon=2.0, survival_times=np.array([1.0, 2.0]),
    )
with self.assertRaisesRegex(ValueError, "first_passage"):
    grouped_exponential_escape_diagnostic(
        np.ones((4, 1)), np.full(4, 21.0), np.ones(4, dtype=bool), np.arange(4) % 2,
        horizon=20.0, survival_times=np.array([1.0, 20.0]),
    )
```

- [ ] **Step 5: Run focused tests and commit**

Run: `python3 -m unittest tests.test_ka_smooth_cage -v`

```bash
git add src/ka_smooth_cage.py tests/test_ka_smooth_cage.py
git commit -m "add grouped microscopic escape diagnostic"
```

### Task 3: Low-disk initial-state reconstruction

**Files:**
- Create: `scripts/reconstruct_ka_clone_initial_state.py`
- Modify: `tests/test_ka_smooth_cage.py`

**Interfaces:**
- Consumes: clone ensemble manifest, existing clone trajectories, parent restarts, LAMMPS binary.
- Produces: one reduced NPZ per clone with frame-zero positions, velocities, forces, types, box lengths, seeds, hashes, and reconstruction error.

- [ ] **Step 1: Write failing runner-contract tests**

```python
def test_initial_state_reconstruction_uses_recorded_seeds_and_run_zero(self):
    source = (ROOT / "scripts" / "reconstruct_ka_clone_initial_state.py").read_text()
    self.assertIn("velocity all create", source)
    self.assertIn("run 0", source)
    self.assertIn("vx vy vz fx fy fz", source)
    self.assertIn("maximum_position_reconstruction_error", source)
    self.assertIn("unlink", source)
```

- [ ] **Step 2: Run and verify missing-script failure**

Run: `python3 -m unittest tests.test_ka_smooth_cage.SmoothCageTests.test_initial_state_reconstruction_uses_recorded_seeds_and_run_zero -v`

- [ ] **Step 3: Implement reconstruction and atomic reduction**

The script must validate manifest hashes and common protocol, generate a `run 0` LAMMPS input for each requested clone, run it with `subprocess.run(command, cwd=clone_output, check=True, capture_output=True, text=True)`, parse the single full-state frame, compare it with the original clone frame zero, write a temporary NPZ and atomically replace the final NPZ, then delete the dump and transient log files.

- [ ] **Step 4: Run one-clone canary and verify reconstruction**

Run:

```bash
python3 scripts/reconstruct_ka_clone_initial_state.py \
  tmp/isoconfigurational_replicate_01_T058 \
  --lammps tmp/toolchains/lammps-22Jul2025_update4/build-lepton/lmp \
  --output-directory tmp/smooth_cage_event_clock_initial_canary \
  --clone-count 1
```

Expected: one NPZ, position error no larger than `2e-5`, and no retained raw dump.

- [ ] **Step 5: Commit**

```bash
git add scripts/reconstruct_ka_clone_initial_state.py tests/test_ka_smooth_cage.py
git commit -m "add clone initial state reconstruction"
```

### Task 4: Fixed-label smooth event-clock analysis

**Files:**
- Create: `scripts/analyze_ka_smooth_cage_event_clock.py`
- Modify: `tests/test_ka_smooth_cage.py`

**Interfaces:**
- Consumes: five clone directories, reconstructed initial-state NPZs, fixed target/event/model controls.
- Produces: reduced per-parent cache NPZs and detail, ablation, survival, and summary CSVs.

- [ ] **Step 1: Write failing analysis-contract tests**

```python
def test_smooth_event_clock_analysis_preregisters_claim_limited_gates(self):
    source = (ROOT / "scripts" / "analyze_ka_smooth_cage_event_clock.py").read_text()
    for token in ("20260714", "0.08", "half_window=8", "target_count=64", "0.026964", "event_clock_claim_allowed"):
        self.assertIn(token, source)
```

- [ ] **Step 2: Implement fixed target and event extraction**

For every path, loop over frames and fixed target indices, evaluate `smooth_force_support_cage`, store `C_i(t)`, call `extract_nonrecrossing_phop_events`, and record the earliest event or censoring for every clone-target row. Cache only centers, first-passage arrays, target indices, and protocol metadata.

- [ ] **Step 3: Implement projected features and ablations**

Load each reconstructed full-state NPZ, call `smooth_cage_projected_observables` with `forces=forces` and `potential_protocol="ka_lj_cut"`, map to geometry/kinematic/full features, and score all three with `grouped_exponential_escape_diagnostic` using the parent id as group.

- [ ] **Step 4: Implement exact summary gates**

```python
gate = (
    full["mean_heldout_brier_skill"] > 0.026964
    and full["mean_heldout_log_likelihood_gain_per_observation"] > 0.0
    and full["mean_heldout_brier_skill"] >= geometry["mean_heldout_brier_skill"] + 0.01
    and full["mean_heldout_brier_skill"] >= kinematic["mean_heldout_brier_skill"]
    and full["maximum_heldout_survival_calibration_error"] <= 0.10
    and full["minimum_group_log_likelihood_gain"] >= 0.0
)
```

Set `microscopic_initial_escape_state_allowed=gate`; set all stronger claim flags false regardless of outcome.

- [ ] **Step 5: Run synthetic/contract tests and commit**

Run: `python3 -m unittest tests.test_ka_smooth_cage -v`

```bash
git add scripts/analyze_ka_smooth_cage_event_clock.py tests/test_ka_smooth_cage.py
git commit -m "add smooth cage event clock analysis"
```

### Task 5: Five-parent experiment and scientific record

**Files:**
- Create: `data/renewal_cage_ka_smooth_cage_event_clock_T058_details.csv`
- Create: `data/renewal_cage_ka_smooth_cage_event_clock_T058_models.csv`
- Create: `data/renewal_cage_ka_smooth_cage_event_clock_T058_survival.csv`
- Create: `data/renewal_cage_ka_smooth_cage_event_clock_T058_summary.csv`
- Create: `docs/microscopic-smooth-cage-event-clock.md`
- Modify: `tests/test_arxiv_package.py`

**Interfaces:**
- Consumes: Tasks 1-4 and existing five-parent trajectories.
- Produces: machine-readable verdict and claim-limited derivation/evidence document.

- [ ] **Step 1: Reconstruct all 40 initial states**

Run the reconstruction script once per parent with `--clone-count 8`. Verify five distinct restart hashes, 40 NPZs, zero retained raw dumps, and maximum position reconstruction error within the canary tolerance.

- [ ] **Step 2: Run the fixed full analysis**

Run `scripts/analyze_ka_smooth_cage_event_clock.py` with the five aligned trajectory/reconstruction directories and output prefix `data/renewal_cage_ka_smooth_cage_event_clock_T058`.

- [ ] **Step 3: Inspect all ablations before writing conclusions**

Report event/censor counts per parent, parent-specific scores, geometry/kinematic/full means, survival errors, feature conditioning, and exact gate booleans. Do not retune threshold, target count, regularization, or feature list after seeing results.

- [ ] **Step 4: Write a failing package evidence test**

The test must require all four CSVs and the document; assert five parents, forty clones, sixty-four targets, fixed protocol strings, exact summary flags, and all stronger claims false.

- [ ] **Step 5: Write the derivation and result document**

Include the censored likelihood derivation, full protocol, reconstruction validation, ablations, parent-level holdouts, negative results, and next-decision boundary. State explicitly whether the initial state gate passed without implying a time-varying renewal clock.

- [ ] **Step 6: Run verification**

```bash
python3 -m unittest tests.test_ka_smooth_cage -v
python3 -m unittest tests.test_arxiv_package.ArxivPackageTests.test_smooth_cage_event_clock_is_complete_and_claim_limited -v
python3 -m unittest discover -s tests -v
python3 -m py_compile src/ka_smooth_cage.py scripts/reconstruct_ka_clone_initial_state.py scripts/analyze_ka_smooth_cage_event_clock.py
git diff --check
```

Expected: all tests pass; runtime warnings from pre-existing synthetic ill-conditioned null tests may remain but no failures are permitted.

- [ ] **Step 7: Commit only exact experiment artifacts**

```bash
git add src/ka_smooth_cage.py tests/test_ka_smooth_cage.py \
  scripts/reconstruct_ka_clone_initial_state.py \
  scripts/analyze_ka_smooth_cage_event_clock.py \
  data/renewal_cage_ka_smooth_cage_event_clock_T058_details.csv \
  data/renewal_cage_ka_smooth_cage_event_clock_T058_models.csv \
  data/renewal_cage_ka_smooth_cage_event_clock_T058_survival.csv \
  data/renewal_cage_ka_smooth_cage_event_clock_T058_summary.csv \
  docs/microscopic-smooth-cage-event-clock.md tests/test_arxiv_package.py
git commit -m "test microscopic smooth cage event clock"
```
