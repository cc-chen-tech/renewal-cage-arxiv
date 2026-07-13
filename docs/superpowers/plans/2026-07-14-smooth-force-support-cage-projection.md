# Smooth Force-Support Cage Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive and validate a differentiable single-particle cage-coordinate SDE and its parameter-free tangent-noise covariance from full 4096-particle C3 KA Langevin trajectories.

**Architecture:** A focused `ka_smooth_cage` module owns the compact cage map, analytic Jacobian, exact projected drift, and matched-path covariance. Unit tests establish each identity before a low-disk runner regenerates common-noise C3 pairs, immediately reduces full dumps to small tangent records, and a separate analysis script applies preregistered covariance gates.

**Tech Stack:** Python 3, NumPy, `unittest`, existing KA force helpers, existing LAMMPS Lepton build, CSV/NPZ artifacts.

## Global Constraints

- Use `w(s)=(1-s)^6(35s^2+18s+3)` on `0 <= s < 1`, zero otherwise.
- Use support radius `2.5 sigma_ij`; do not tune it against any observable.
- Use only `ka_lj_c3_switch` for the differentiable tangent experiment.
- Fit no diffusion, NGP, scattering, event-clock, or thermodynamic observable.
- Preserve `thermodynamic_claim_allowed = 0` in every returned or saved result.
- Do not stage unrelated untracked experiment artifacts.

---

### Task 1: Compact cage map and analytic Jacobian

**Files:**
- Create: `src/ka_smooth_cage.py`
- Create: `tests/test_ka_smooth_cage.py`

**Interfaces:**
- Produces: `wendland_c4_weight(scaled_distance)` returning `(weight, derivative)`.
- Produces: `smooth_force_support_cage(positions, particle_types, box_lengths, target_index)` returning coordinate, center, weights, support, and particle-aligned Jacobian blocks.

- [ ] **Step 1: Write failing weight and boundary tests**

```python
def test_wendland_weight_and_derivative_are_compact_and_smooth(self):
    s = np.array([0.0, 0.5, 1.0, 1.1])
    weight, derivative = wendland_c4_weight(s)
    self.assertAlmostEqual(weight[0], 3.0)
    self.assertGreater(weight[1], 0.0)
    np.testing.assert_array_equal(weight[2:], 0.0)
    np.testing.assert_array_equal(derivative[2:], 0.0)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python3 -m unittest tests.test_ka_smooth_cage.SmoothCageTests.test_wendland_weight_and_derivative_are_compact_and_smooth -v`

Expected: import or symbol failure because `ka_smooth_cage.py` does not exist.

- [ ] **Step 3: Implement the weight and derivative**

```python
def wendland_c4_weight(scaled_distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scaled_distance, dtype=float)
    active = (s >= 0.0) & (s < 1.0)
    x = np.where(active, s, 1.0)
    weight = np.where(active, (1.0 - x) ** 6 * (35.0 * x**2 + 18.0 * x + 3.0), 0.0)
    derivative = np.where(
        active,
        -56.0 * x * (5.0 * x + 1.0) * (1.0 - x) ** 5,
        0.0,
    )
    return weight, derivative
```

- [ ] **Step 4: Add failing translation, decomposition, and Jacobian tests**

Use a four-particle periodic configuration with mixed KA types. Check:

```python
np.testing.assert_allclose(result["relative_position"], tagged - result["cage_position"], atol=1e-12)
np.testing.assert_allclose(np.sum(result["jacobian"], axis=0), np.zeros((3, 3)), atol=1e-12)
```

For every particle and Cartesian component, compare the analytic Jacobian with
a centered coordinate difference at step `1e-6`; require relative L2 error
below `1e-6`.

- [ ] **Step 5: Implement `smooth_force_support_cage`**

Implement the design equations exactly:

```python
gradient = derivative[:, None] * unit / support_radius[:, None]
mu = np.sum(weight[:, None] * displacement, axis=0) / total_weight
block = (
    weight[:, None, None] * np.eye(3)
    + (displacement - mu)[..., :, None] * gradient[..., None, :]
) / total_weight
jacobian[target_index] = np.sum(block, axis=0)
jacobian[other_indices] = -block
```

Reject nonfinite inputs, invalid types/box/target, self-overlap, and zero total
weight. Return `support_radius`, `support`, and `weight` for auditability.

- [ ] **Step 6: Run focused tests**

Run: `python3 -m unittest tests.test_ka_smooth_cage -v`

Expected: all Task 1 tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/ka_smooth_cage.py tests/test_ka_smooth_cage.py
git commit -m "add differentiable microscopic cage coordinate"
```

---

### Task 2: Exact projected drift and FDT geometry

**Files:**
- Modify: `src/ka_smooth_cage.py`
- Modify: `tests/test_ka_smooth_cage.py`

**Interfaces:**
- Consumes: `smooth_force_support_cage(...)`.
- Produces: `smooth_cage_projected_observables(..., velocities, friction, temperature, directional_step, potential_protocol)`.

- [ ] **Step 1: Write failing kinematic and covariance tests**

Check

```python
relative_velocity = np.einsum("nab,nb->a", result["jacobian"], velocities)
np.testing.assert_allclose(result["relative_velocity"], relative_velocity)
np.testing.assert_allclose(
    result["noise_covariance_rate"],
    2.0 * friction * temperature * result["jacobian_gram"],
)
np.testing.assert_allclose(
    result["effective_mass"] @ result["jacobian_gram"], np.eye(3), atol=1e-10
)
```

- [ ] **Step 2: Run tests and verify symbol failure**

Run: `python3 -m unittest tests.test_ka_smooth_cage.SmoothCageTests.test_projected_observables_obey_kinematics_and_fdt -v`

Expected: `smooth_cage_projected_observables` is missing.

- [ ] **Step 3: Implement local force contraction and directional curvature**

Use `ka_lj_force_and_isotropic_curvature` with the nonzero Jacobian particle
indices and `potential_protocol="ka_lj_c3_switch"`. Compute

```python
jacobian_force = np.einsum("nab,nb->a", jacobian[active], force[active])
plus_velocity = J(R+hV) @ V
minus_velocity = J(R-hV) @ V
geometric_drift = (plus_velocity - minus_velocity) / (2*h)
drift = jacobian_force + geometric_drift - friction * relative_velocity
```

Return eigenvalues and condition number of `G=JJ^T`; reject a non-positive
eigenvalue instead of silently regularizing it.

- [ ] **Step 4: Add a deterministic finite-difference drift test**

For zero friction and temperature, compare the implemented drift with

```text
[p(R+dt V, V+dt F)-p(R,V)]/dt
```

over decreasing `dt`. Require second-order directional-derivative convergence
and final relative error below `2e-4`.

- [ ] **Step 5: Run focused tests and compile**

Run:

```bash
python3 -m unittest tests.test_ka_smooth_cage -v
python3 -m py_compile src/ka_smooth_cage.py
```

Expected: all tests and compilation pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/ka_smooth_cage.py tests/test_ka_smooth_cage.py
git commit -m "derive exact smooth cage projected SDE"
```

---

### Task 3: Matched tangent path and covariance reduction

**Files:**
- Modify: `src/ka_smooth_cage.py`
- Modify: `tests/test_ka_smooth_cage.py`

**Interfaces:**
- Produces: `extract_smooth_cage_path(trajectory_path, ...)`.
- Produces: `matched_smooth_cage_tangent(positive, negative, epsilon)`.
- Produces: `integrated_smooth_cage_tangent_covariance(path, stride)`.

- [ ] **Step 1: Write a failing LAMMPS-path extraction test**

Create a two-frame four-particle custom dump with positions and velocities.
Assert the returned arrays have shapes:

```python
relative_position: (frames, 3)
relative_velocity: (frames, 3)
projected_drift: (frames, 3)
jacobian: (frames, particles, 3, 3)
noise_covariance_rate: (frames, 3, 3)
```

- [ ] **Step 2: Implement path extraction**

Reuse `load_lammps_custom_trajectory`. Evaluate Task 2 observables frame by
frame and return physical time from dump timesteps. Do not save macro
observables.

- [ ] **Step 3: Write failing central-response tests**

Synthetic plus/minus dictionaries must yield

```python
delta_p = (p_plus-p_minus)/(2*epsilon)
delta_b = (b_plus-b_minus)/(2*epsilon)
delta_J = (J_plus-J_minus)/(2*epsilon)
covariance_rate = 2*gamma*T*np.einsum("tnab,tncb->tac", delta_J, delta_J)
```

- [ ] **Step 4: Implement matched reduction and stride integration**

For interval stride `s`, use trapezoidal drift subtraction and covariance
integration:

```python
residual = delta_p[s:] - delta_p[:-s] - 0.5*dt*s*(delta_b[s:] + delta_b[:-s])
integrated_covariance = 0.5*dt*s*(covariance_rate[s:] + covariance_rate[:-s])
```

Return the full arrays required by `tangent_noise_covariance_diagnostic`.

- [ ] **Step 5: Run focused tests**

Run: `python3 -m unittest tests.test_ka_smooth_cage -v`

Expected: all Task 1--3 tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/ka_smooth_cage.py tests/test_ka_smooth_cage.py
git commit -m "add smooth cage tangent covariance reduction"
```

---

### Task 4: Low-disk C3 matched-response runner

**Files:**
- Create: `scripts/run_ka_smooth_cage_response.py`
- Modify: `tests/test_ka_smooth_cage.py`

**Interfaces:**
- Consumes: existing C3 parent restart, pinned LAMMPS executable, and `generator_response_lammps_input`.
- Produces: one `smooth_cage_tangent.npz` per member/epsilon pair and removes full dumps after successful reduction.

- [ ] **Step 1: Write failing CLI/source-contract tests**

Check that the script exposes `--parent-restart`, `--lammps`,
`--output-directory`, `--members`, `--epsilons`, `--run-steps`,
`--dump-interval`, and `--directional-step`; requests
`potential_protocol="ka_lj_c3_switch"`; and only unlinks trajectories after
the NPZ is atomically replaced.

- [ ] **Step 2: Implement the runner**

For each member and epsilon:

1. Generate plus/minus C3 LAMMPS inputs with common velocity/Langevin seeds.
2. Run both serially and require successful logs.
3. Extract both smooth-cage paths.
4. Reduce to central tangent arrays and save through a temporary NPZ followed
   by `Path.replace`.
5. Delete only the two full dumps after the reduced artifact exists.

Save protocol, target id, seeds, parent SHA-256, epsilon, frame time, and
`thermodynamic_claim_allowed=0`.

- [ ] **Step 3: Run CLI tests and compilation**

Run:

```bash
python3 -m unittest tests.test_ka_smooth_cage -v
python3 -m py_compile scripts/run_ka_smooth_cage_response.py
```

- [ ] **Step 4: Commit Task 4**

```bash
git add scripts/run_ka_smooth_cage_response.py tests/test_ka_smooth_cage.py
git commit -m "add low disk smooth cage response protocol"
```

---

### Task 5: Run the C3 experiment and apply preregistered gates

**Files:**
- Create: `scripts/analyze_ka_smooth_cage_tangent.py`
- Create: `data/renewal_cage_ka_smooth_cage_tangent_T058_summary.csv`
- Create: `data/renewal_cage_ka_smooth_cage_tangent_stride1_T058.csv`
- Create: `data/renewal_cage_ka_smooth_cage_tangent_stride2_T058.csv`
- Create: `data/renewal_cage_ka_smooth_cage_tangent_stride5_T058.csv`
- Modify: `tests/test_ka_smooth_cage.py`

**Interfaces:**
- Consumes: 8 members x 2 epsilons from Task 4.
- Produces: pooled and member-resolved covariance diagnostics plus gate booleans.

- [ ] **Step 1: Write failing analysis-contract test**

Require the source to call the existing
`tangent_noise_covariance_diagnostic`, report both epsilons separately, compute
cross-epsilon covariance convergence, and emit all preregistered gate columns.

- [ ] **Step 2: Implement analysis CLI**

Load every reduced NPZ and evaluate strides 1, 2, and 5. Select stride 5 as
the preregistered primary comparison, matching the earlier C3 tangent
checkpoint. Emit member rows plus pooled rows; do not select a stride after
looking at which one passes.

- [ ] **Step 3: Run the 32-path C3 protocol**

Run:

```bash
python3 scripts/run_ka_smooth_cage_response.py \
  --parent-restart tmp/ka_c3_parent_T058_tau10/equilibrated_c3.restart \
  --lammps tmp/lammps-lepton-build/lmp \
  --output-directory tmp/smooth_cage_c3_response_T058 \
  --members 8 --epsilons 0.001 0.002 \
  --run-steps 200 --dump-interval 1 --directional-step 1e-5
```

Expected: 16 reduced pair artifacts, 32 successful LAMMPS logs, and no retained
full dumps except an optional explicitly requested canary.

- [ ] **Step 4: Analyze all strides**

Run:

```bash
python3 scripts/analyze_ka_smooth_cage_tangent.py \
  --input-directory tmp/smooth_cage_c3_response_T058 \
  --summary-output data/renewal_cage_ka_smooth_cage_tangent_T058_summary.csv \
  --stride-output-prefix data/renewal_cage_ka_smooth_cage_tangent \
  --strides 1 2 5
```

Expected: output states pass/fail from the fixed design gates; no gate is
silently weakened.

- [ ] **Step 5: Run tests and CSV checks**

Run:

```bash
python3 -m unittest tests.test_ka_smooth_cage -v
python3 -m py_compile src/ka_smooth_cage.py scripts/run_ka_smooth_cage_response.py scripts/analyze_ka_smooth_cage_tangent.py
git diff --check
```

- [ ] **Step 6: Commit Task 5**

```bash
git add scripts/analyze_ka_smooth_cage_tangent.py tests/test_ka_smooth_cage.py data/renewal_cage_ka_smooth_cage_tangent_T058_summary.csv data/renewal_cage_ka_smooth_cage_tangent_stride1_T058.csv data/renewal_cage_ka_smooth_cage_tangent_stride2_T058.csv data/renewal_cage_ka_smooth_cage_tangent_stride5_T058.csv
git commit -m "validate microscopic smooth cage tangent dynamics"
```

---

### Task 6: Scientific record and next-decision boundary

**Files:**
- Create: `docs/microscopic-smooth-cage-projection.md`
- Modify: `tests/test_arxiv_package.py`

**Interfaces:**
- Consumes: design, formulas, CSV evidence, and exact commands from Tasks 1--5.
- Produces: a bounded scientific checkpoint and the next admissible experiment.

- [ ] **Step 1: Add an artifact-presence test**

Require the derivation document, summary CSV, all three stride tables, and
analysis scripts to be present in the arXiv package inventory test.

- [ ] **Step 2: Write the scientific record**

Document:

- the analytic cage map and Jacobian;
- exact projected SDE and matrix FDT;
- common-noise tangent protocol;
- all gate values, including failures;
- comparison with raw/fixed/hard-dynamic coordinates;
- why a passing instantaneous projection is not yet an autonomous GLE;
- the next decision: event-clock validation only if the tangent covariance
  passes, otherwise reject this coordinate.

- [ ] **Step 3: Run focused and full tests**

Run:

```bash
python3 -m unittest tests.test_ka_smooth_cage tests.test_arxiv_package -v
python3 -m unittest discover -s tests -v
git diff --check
```

Expected: all tests pass and no whitespace errors remain.

- [ ] **Step 4: Commit Task 6**

```bash
git add docs/microscopic-smooth-cage-projection.md tests/test_arxiv_package.py
git commit -m "document smooth microscopic cage projection"
```

## Plan self-review

- Every design requirement maps to a task: coordinate/Jacobian (Task 1), exact
  drift/FDT (Task 2), tangent reduction (Task 3), low-disk full-particle
  simulation (Task 4), preregistered gates (Task 5), claim boundary (Task 6).
- Function names and array shapes are consistent across tasks.
- The plan contains no event-rate or macro-observable fitting.
- The event-clock phase remains conditional on the tangent result and is not
  falsely included in this instantaneous-SDE checkpoint.
