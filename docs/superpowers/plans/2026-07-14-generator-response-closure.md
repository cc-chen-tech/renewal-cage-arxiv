# Generator-Constrained Response Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate low-disk matched full-KA responses containing exact `F`, `LF`, and `L2F`, then test whether the 12-coordinate tagged Krylov state predicts held-out full-system responses.

**Architecture:** A focused source module builds deterministic LAMMPS inputs, extracts microscopic generator paths, and fits a continuous tangent generator whose first three block rows are fixed by the many-particle Langevin equation. A runner processes each raw dump immediately into a compressed path and deletes it after verification, except for one audit dump. A separate analysis CLI performs leave-one-member-out, cross-epsilon, temporal, and stability gates without fitting any macroscopic observable.

**Tech Stack:** Python 3.11, NumPy, `unittest`, serial LAMMPS 22 Jul 2025 Update 4, existing KA trajectory reader and force-generator APIs.

## Global Constraints

- Preserve `thermodynamic_claim_allowed = 0` in every manifest and output.
- Use one parent restart with SHA-256 `f6e235d87a9442c148a41b837a6ca5897707ff531e52dffe43e19a4d688408cc`.
- Use unit mass, `T=0.58`, `gamma=1`, LAMMPS integration step `0.001 tau`, saved interval `0.005 tau`, and duration `1 tau`.
- Use target particle id 821 and x-directed displacements `epsilon=0.001,0.002`.
- Use identical velocity and Langevin seeds within every matched `+/-` pair and across epsilon values for a given member.
- Fit no diffusion, NGP, scattering, event-clock, or thermodynamic observable.
- Keep peak new raw storage below 200 MiB by sequential extraction and verified deletion.
- Retain one raw audit dump only; never delete a raw dump until its compressed extraction reopens and passes shape/finite checks.
- Follow test-first development and use `apply_patch` for manual edits.

---

### Task 1: LAMMPS Protocol and Microscopic Path Extraction

**Files:**
- Create: `src/ka_generator_response.py`
- Modify: `tests/test_ka_replicates.py`

**Interfaces:**
- Produces: `generator_response_lammps_input(...) -> str`.
- Produces: `extract_generator_response_path(path, target_id, temperature, friction, integration_time_step, directional_step) -> dict[str, np.ndarray | float]`.
- Consumes: `load_lammps_custom_trajectory`, `ka_lj_force_generator_observables`, and `ka_lj_second_force_generator`.

- [ ] **Step 1: Add a failing protocol test**

The test constructs one input and requires one-particle displacement, common explicit seeds, a full sorted `R,V` dump every five integration steps, and no force fitting or macro-observable controls.

```python
text = generator_response_lammps_input(
    parent_restart=Path("/tmp/parent.restart"),
    target_id=821,
    displacement=0.001,
    temperature=0.58,
    friction=1.0,
    velocity_seed=82101,
    langevin_seed=83101,
    run_steps=1000,
    dump_interval_steps=5,
    trajectory_name="trajectory.lammpstrj",
)
self.assertIn("displace_atoms tagged move 0.001 0 0 units box", text)
self.assertIn("velocity all create 0.58 82101", text)
self.assertIn("fix bath all langevin 0.58 0.58 1 83101", text)
self.assertIn("dump trajectory all custom 5 trajectory.lammpstrj id type x y z ix iy iz vx vy vz", text)
self.assertIn("dump_modify trajectory sort id", text)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run the exact unittest name. Expected: import failure because `ka_generator_response` does not exist.

- [ ] **Step 3: Implement the input builder**

Validate positive ids, seeds, duration, dump interval, temperature, and friction. Return a complete serial `NVE + fix langevin` input using `read_restart`, `reset_timestep 0`, tagged displacement, full velocity resampling, sorted dump, and `run`.

- [ ] **Step 4: Add a failing extraction test with a tiny mocked loader boundary**

Patch only `ka_generator_response.load_lammps_custom_trajectory` to return a five-frame, two-particle 3D path. Require keys `time`, `position`, `velocity`, `force`, `force_generator`, `second_force_generator`, and `force_generator_noise_covariance_rate`, with shapes `(5,)`, `(5,3)`, `(5,3)`, `(5,3)`, `(5,3)`, `(5,3)`, and `(5,3,3)`.

- [ ] **Step 5: Implement extraction using only tested microscopic APIs**

Require uniform saved timesteps and velocity columns. For each frame call the exact first- and second-generator APIs for one target. Return physical times and arrays without fitting or smoothing.

- [ ] **Step 6: Run focused and neighboring generator tests**

Expected: protocol, extraction, `LF`, `L2F`, and finite-step diagnostic tests all pass.

---

### Task 2: Sequential Low-Disk Matched Response Runner

**Files:**
- Create: `scripts/run_ka_generator_response.py`
- Modify: `tests/test_ka_replicates.py`

**Interfaces:**
- Consumes: LAMMPS binary, parent restart, seed vectors, epsilon vector, and Task 1 APIs.
- Produces: one compressed `path.npz` per member/epsilon/sign, `manifest.json`, and optionally one retained audit dump.

- [ ] **Step 1: Add a failing `--help` test**

Require CLI controls for `--lammps-binary`, `--parent-restart`, `--output-directory`, `--velocity-seeds`, `--langevin-seeds`, `--epsilons`, `--duration`, `--dump-interval`, and `--retain-audit-raw`.

- [ ] **Step 2: Implement the runner**

For each member, epsilon, and sign: create a private run directory; write `in.response`; execute `[lmp, -in, in.response, -log, log.lammps, -screen, none]`; extract the generator path; write a compressed NPZ containing metadata and all microscopic arrays; reopen it and verify finite arrays and expected frame count; then remove the raw dump unless it is the single selected audit path.

- [ ] **Step 3: Write an atomic manifest**

The manifest records parent SHA-256, seeds, epsilon, sign, frame time, path SHA-256, raw retention, `fit_parameters_from_macro_observables=false`, and `thermodynamic_claim_allowed=false`. Write to `.tmp`, then rename only after all requested paths succeed.

- [ ] **Step 4: Run one-member/one-epsilon canary**

Use seeds `82101/83101`, epsilon `0.001`, two signs, and retain the positive raw dump. Expected: two 201-frame NPZ files, exact common seeds, one raw dump, and one deleted raw dump.

- [ ] **Step 5: Run the eight-member/two-epsilon ensemble**

Use velocity seeds `82101,82139,82157,82181,82203,82239,82277,82301` and Langevin seeds `83101,83139,83157,83181,83203,83239,83277,83301`. Expected: 32 compressed path files and one audit raw dump.

---

### Task 3: Generator-Constrained Tangent Propagator

**Files:**
- Modify: `src/ka_generator_response.py`
- Modify: `tests/test_ka_replicates.py`

**Interfaces:**
- Produces: `fit_generator_constrained_response(state_response, second_force_response, frame_time, friction, fit_frames) -> dict`.
- State ordering: `(delta x[3], delta v[3], delta F[3], delta LF[3])`.
- Only fitted object: a `3 x 12` block projecting `delta L2F` onto the retained response state.

- [ ] **Step 1: Add a failing synthetic exact-system test**

Construct a stable `3 x 12` final block, assemble the fixed continuous matrix

```text
d(delta x)/dt  = delta v
d(delta v)/dt  = delta F - gamma delta v
d(delta F)/dt  = delta LF
d(delta LF)/dt = A [delta x,delta v,delta F,delta LF],
```

generate a response with the same RK4 step used by production, and require recovery of `A`, the full response, and spectral radius below one to numerical tolerance.

- [ ] **Step 2: Verify RED**

Expected: missing `fit_generator_constrained_response`.

- [ ] **Step 3: Implement scaled least squares for the final block**

Fit `delta L2F = A delta state` on frames `[0, fit_frames)`, scaling each state coordinate by its training RMS. Reject rank-deficient designs. Assemble the first nine rows exactly from microscopic kinematics and friction. Use RK4 to build the one-frame transition and propagate from the measured initial state without resetting to later observations.

- [ ] **Step 4: Return falsifiable diagnostics**

Return fitted block, continuous generator, transition, predicted state, spectral radius, training `L2F` relative L2, training/held position relative L2, and residual correlation with each retained coordinate.

- [ ] **Step 5: Run synthetic and full generator tests**

Expected: exact synthetic recovery and no regression in Tasks 1-2.

---

### Task 4: Ensemble Closure and Preregistered Gates

**Files:**
- Create: `scripts/analyze_ka_generator_response_closure.py`
- Create: `data/renewal_cage_ka_generator_response_closure_T058_summary.csv`
- Create: `data/renewal_cage_ka_generator_response_closure_T058_curve.csv`
- Modify: `docs/microscopic-frozen-minimum-response.md`

**Interfaces:**
- Consumes: Task 2 manifest and 32 path NPZ files.
- Produces: cross-epsilon linearity, leave-one-member-out, temporal, stability, and empirical-four-state baseline rows.

- [ ] **Step 1: Add a failing CLI command-surface test**

Require `manifest`, `--output-prefix`, `--fit-times`, `--horizons`, and `--linearity-tolerance`.

- [ ] **Step 2: Implement matched responses**

For every microscopic array compute `(plus-minus)/(2 epsilon)`. Confirm identical time grids and seeds. Report cross-epsilon state mismatch through `0.2` and `1 tau`; do not fit a closure at a horizon whose position mismatch exceeds `0.02`.

- [ ] **Step 3: Implement leave-one-member-out fits**

Pool training members' response states and `L2F` targets to fit one final block, initialize propagation from the held member's measured first state, and score held position response at `0.2` and `1 tau`. Fit intervals are `0.05,0.1,0.2 tau` and must leave a temporal holdout.

- [ ] **Step 4: Compare the unconstrained empirical baseline**

On the identical folds and fit intervals, fit a free `12 x 12` discrete transition. Report its rank, spectral radius, and held position errors separately. It is a null comparator, not the microscopic model.

- [ ] **Step 5: Apply preregistered gates**

The generator closure passes only if every identified fold has position relative error `<=0.20` at `0.2 tau`, `<=0.30` at `1 tau`, transition spectral radius `<=1+1e-6`, and cross-epsilon mismatch `<=0.02`. Preserve failures and `thermodynamic_claim_allowed=0`.

- [ ] **Step 6: Run the complete ensemble and document the result**

Write full-precision CSVs. Document whether rank four in each Cartesian sector passes or fails, how it compares with the empirical baseline, and which next Krylov mode or cage-relative coordinate is activated by the result.

- [ ] **Step 7: Verify the complete subproject**

Run `python -m unittest discover -s tests`, compile all new/modified Python files, check whitespace, and verify the manifest contains exactly 32 path records with one retained raw dump.
