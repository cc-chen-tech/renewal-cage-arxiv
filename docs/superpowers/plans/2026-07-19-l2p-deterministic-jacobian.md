# Deterministic L2p Jacobian Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed 32-probe Hutchinson estimate of the microscopic `L2p` conditional diffusion with a deterministic construction of `A_c = grad_V(L^2 p)` and `Q_c = 2 gamma T A_c A_c^T`.

**Architecture:** Extend the existing smooth cage batch evaluator to expose its analytic particle-block Jacobian. Build unordered active KA pair Hessians once per frame, contract them with the cage Jacobian to obtain `J DF`, and combine this exact force response with centered finite differences of the analytic cage Jacobian. A new resumable cache command writes deterministic `Q_c` with exact trajectory provenance; the held analyzer may consume it only after independent directional and step-convergence gates pass.

**Tech Stack:** Python 3, NumPy, `unittest`, existing KA trajectory/cache utilities.

## Global Constraints

- Work only in `/Users/luicy/AI/renewal-cage-arxiv/.worktrees/l2p-conditional-diffusion-run`.
- Treat `/Users/luicy/AI/renewal-cage-arxiv/.worktrees/prl-event-clock-closure` as read-only.
- Do not change trajectory inputs, target selection, held split, score tolerances, or physical model families.
- Do not retune the failed 4/8/16/32 probe gate.
- Validate deterministic numerical convergence before reading held closure scores.
- Preserve both `ka_lj_cut` and `ka_lj_c3_switch` protocols.
- Keep every broad physical and thermodynamic claim flag at zero.

---

### Task 1: Expose the analytic full cage Jacobian

**Files:**
- Modify: `src/ka_smooth_cage.py`
- Modify: `tests/test_ka_smooth_cage.py`

**Interfaces:**
- Consumes: `smooth_force_support_cage_batch(...)` and the scalar `smooth_force_support_cage(...)` reference.
- Produces: optional `return_jacobian: bool = False`; when true, result key `jacobian` has shape `(targets, particles, 3, 3)`.

- [ ] **Step 1: Write the failing scalar-equivalence test**

Add a test that requests `return_jacobian=True` for at least two targets and
checks every block against the scalar evaluator to `rtol=1e-13, atol=1e-13`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m unittest tests.test_ka_smooth_cage.SmoothCageTests.test_batch_full_jacobian_matches_scalar_blocks -v
```

Expected: fail because `return_jacobian` is not accepted.

- [ ] **Step 3: Implement the minimal optional output**

Inside each existing target batch, form neighbor blocks

```text
- [w I + centered outer radial_gradient] / total_weight
```

and replace the selected target block by the existing analytic
`target_jacobian_block`. Allocate the full tensor only when requested.

- [ ] **Step 4: Add projection and Gram identities**

Check

```python
np.einsum("tnab,knb->kta", jacobian, vectors)
```

using the correct simplified indices against `relative_velocity` and
`additional_relative_velocity`, and check `J J^T` against `jacobian_gram`.

- [ ] **Step 5: Run the smooth-cage suite and commit**

```bash
python -m unittest tests.test_ka_smooth_cage -v
git add src/ka_smooth_cage.py tests/test_ka_smooth_cage.py
git commit -m "expose full smooth-cage Jacobian"
```

---

### Task 2: Build unordered microscopic KA pair-Hessian geometry

**Files:**
- Modify: `src/ka_local_cage.py`
- Modify: `tests/test_ka_replicates.py`

**Interfaces:**
- Consumes: `ka_lj_radial_derivatives(...)`, KA type parameters, and periodic minimum-image geometry.
- Produces: `ka_lj_pair_hessian_geometry(positions, *, particle_types, box_lengths, potential_protocol) -> dict[str, np.ndarray | float]` with `particle_i`, `particle_j`, `pair_hessian`, `pair_count`, and `thermodynamic_claim_allowed`.

- [ ] **Step 1: Write the failing protocol-parametrized geometry test**

For `ka_lj_cut` and `ka_lj_c3_switch`, require `i < j`, finite symmetric 3x3
pair blocks, no inactive pairs, and exact agreement with the corresponding
blocks from `_ka_lj_target_pair_geometry` on a small configuration.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
python -m unittest tests.test_ka_replicates.KAReplicateTests.test_unordered_pair_hessian_geometry_matches_dense_reference -v
```

Expected: import failure for the new public function.

- [ ] **Step 3: Implement vectorized upper-triangle construction**

Generate `np.triu_indices(N, k=1)`, apply minimum image, evaluate radial
derivatives, filter to active force-support pairs, and only then allocate the
3x3 Hessian blocks. Do not construct an `(N,N,3,3)` tensor.

- [ ] **Step 4: Add translation and rotation covariance tests**

Require pair indices to remain fixed under a common translation and require
`K -> R K R^T` under a rigid rotation in a sufficiently large box.

- [ ] **Step 5: Run focused local-cage tests and commit**

```bash
python -m unittest tests.test_ka_replicates -v
git add src/ka_local_cage.py tests/test_ka_replicates.py
git commit -m "add sparse KA pair-Hessian geometry"
```

---

### Task 3: Construct and verify `J DF`

**Files:**
- Modify: `src/ka_smooth_cage.py`
- Modify: `tests/test_ka_smooth_cage.py`

**Interfaces:**
- Consumes: full cage `jacobian`, unordered `particle_i`, `particle_j`, and `pair_hessian`.
- Produces: `contract_cage_jacobian_force_jacobian(cage_jacobian, *, particle_i, particle_j, pair_hessian) -> np.ndarray` with shape `(targets, 3, particles, 3)`.

- [ ] **Step 1: Write a failing arbitrary-direction identity test**

For both force protocols, apply the returned matrix to a fixed Gaussian
`eta`. Compare it to

```text
J @ ka_lj_sparse_force_generator_multi(..., velocity_fields=eta)
```

at `rtol=2e-12, atol=2e-12` on a small non-cutoff configuration.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
python -m unittest tests.test_ka_smooth_cage.SmoothCageTests.test_cage_force_jacobian_contraction_matches_directional_force_response -v
```

- [ ] **Step 3: Implement unordered-pair accumulation**

For each pair `(i,j)`, compute

```text
B_ij = (J_j - J_i) K_ij
A[..., i, :] += B_ij
A[..., j, :] -= B_ij
```

with signs verified against the directional force-generator convention.

- [ ] **Step 4: Run the identity under both protocols and commit**

```bash
python -m unittest tests.test_ka_smooth_cage -v
git add src/ka_smooth_cage.py tests/test_ka_smooth_cage.py
git commit -m "contract cage and KA force Jacobians"
```

---

### Task 4: Build deterministic `A_c` and `Q_c`

**Files:**
- Modify: `src/ka_smooth_cage.py`
- Modify: `src/ka_l2p_conditional_diffusion.py`
- Modify: `tests/test_ka_smooth_cage.py`
- Modify: `tests/test_ka_l2p_conditional_diffusion.py`

**Interfaces:**
- Produces: `smooth_cage_l2p_velocity_jacobian_batch(..., jacobian_step, potential_protocol, target_batch_size=16)` returning `l2p_velocity_jacobian` and the fixed claim flag by default; `return_components=True` retains the five diagnostic derivative components at higher memory cost.
- Produces: `deterministic_conditional_diffusion(velocity_jacobian, *, friction, temperature)` returning symmetric `conditional_diffusion`.

- [ ] **Step 1: Write the failing `A_c @ eta` identity test**

Use a fixed small KA configuration and fixed random `eta`. Compare matrix
application with the existing
`smooth_cage_l2p_velocity_directional_derivative_batch`. Test a short
development step ladder without selecting held data.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
python -m unittest tests.test_ka_smooth_cage.SmoothCageTests.test_deterministic_l2p_velocity_jacobian_matches_directional_derivative -v
```

- [ ] **Step 3: Implement centered Jacobian derivatives**

Evaluate full analytic `J` at `R`, `R +/- hV`, and `R +/- hF`, then assemble

```text
A_c = J DF + 3 D_R J[F] - 6 gamma D_R J[V]
      + gamma^2 J + 3 D_R^2 J[V,V].
```

Return each component so a numerical failure can be localized.

- [ ] **Step 4: Write and pass PSD/no-probe tests for `Q_c`**

Require exact agreement with `2*gamma*T*A@A.T`, finite symmetry, eigenvalues
above `-1e-10 * max(trace,1)`, and no probe-count metadata.

- [ ] **Step 5: Run both focused suites and commit**

```bash
python -m unittest tests.test_ka_smooth_cage tests.test_ka_l2p_conditional_diffusion -v
git add src/ka_smooth_cage.py src/ka_l2p_conditional_diffusion.py tests/test_ka_smooth_cage.py tests/test_ka_l2p_conditional_diffusion.py
git commit -m "derive deterministic L2p conditional diffusion"
```

---

### Task 5: Add resumable deterministic caches and frozen numerical verdict

**Files:**
- Create: `scripts/cache_ka_l2p_deterministic_diffusion.py`
- Modify: `scripts/analyze_ka_l2p_conditional_diffusion.py`
- Modify: `tests/test_ka_l2p_conditional_diffusion.py`
- Modify: `docs/superpowers/specs/2026-07-19-l2p-deterministic-jacobian-design.md`

**Interfaces:**
- Consumes: exact trajectory, drift cache, second-generator cache, deterministic `Q_c` evaluator.
- Produces: resumable `clone_NNN_l2p_deterministic_diffusion.npz` files and a numerical-only convergence verdict.

- [ ] **Step 1: Freeze the real-frame step ladder and acceptance limits**

Based only on synthetic directional-identity conditioning, append exact
primary/sensitivity steps and adjacent-step median/p95 limits to the design
spec before running the real-frame ladder. Do not inspect held closure scores.

- [ ] **Step 2: Write failing CLI/alignment/checkpoint tests**

Require exact trajectory SHA, target list, potential protocol, derivative
step, estimator name, completed frame count, and all zero claim flags.

- [ ] **Step 3: Implement the resumable cache command**

Reuse loading and provenance validation from
`cache_ka_l2p_conditional_diffusion.py`; do not duplicate held-model fitting.

- [ ] **Step 4: Run one fixed real-frame convergence/performance canary**

Record componentwise errors, total `A_c` error, `Q_c` error, PSD diagnostics,
wall time, and peak cache size. If the frozen numerical gate fails, stop here.

- [ ] **Step 5: Commit numerical evidence**

```bash
python -m unittest tests.test_ka_l2p_conditional_diffusion -v
git add scripts/cache_ka_l2p_deterministic_diffusion.py scripts/analyze_ka_l2p_conditional_diffusion.py tests/test_ka_l2p_conditional_diffusion.py docs/superpowers/specs/2026-07-19-l2p-deterministic-jacobian-design.md
git commit -m "cache deterministic L2p diffusion tensors"
```

---

### Task 6: Run the held closure only after numerical PASS

**Files:**
- Create only on numerical PASS: deterministic T=0.58 CSV/SVG result artifacts.
- Modify: result documentation and arXiv recomputation tests only after the verdict is mechanically generated.

**Interfaces:**
- Consumes: four completed 200-frame deterministic caches.
- Produces: the existing constant/permuted/scalar/tensor held comparison with estimator provenance separated from the physical verdict.

- [ ] **Step 1: Generate all four caches without changing frozen inputs**

Checkpoint each frame and preserve raw caches under ignored `tmp/` paths.

- [ ] **Step 2: Run the unchanged held model families**

Do not alter covariance families or tolerances. Label the old 32-probe result
as numerically unresolved and the deterministic result by its own estimator
name.

- [ ] **Step 3: Mechanically generate verdict, CSV, and SVG**

All broad physical and thermodynamic claim flags remain zero regardless of
score outcome.

- [ ] **Step 4: Run full verification**

```bash
python -m unittest discover -s tests -v
python scripts/generate_results.py --check
bash scripts/build_arxiv_package.sh
git diff --check
git status --short
```

- [ ] **Step 5: Commit, push, and update PR #21**

```bash
git add <reviewed tracked result files>
git commit -m "test deterministic microscopic L2p diffusion closure"
git push origin codex/l2p-conditional-diffusion-run
gh pr edit 21 --body-file <updated-pr-body>
gh pr checks 21 --watch
```
