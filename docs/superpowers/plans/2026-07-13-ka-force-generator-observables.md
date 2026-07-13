# KA Force-Generator Observables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement exact first- and second-generator observables for the tagged KA pair force and turn the existing full-state trajectories into reproducible force-derivative and Hessian-noise validation data.

**Architecture:** A private vectorized pair-geometry routine supplies the exact KA pair force and Hessian row for selected particles. Public routines apply the underdamped Langevin backward generator to obtain `F`, `LF`, `L2F`, and the infinitesimal covariance rate of the stochastic part of `d(LF)`. A separate analysis helper and CLI validate these identities on existing trajectories without fitting any macroscopic observable.

**Tech Stack:** Python 3.11, NumPy, `unittest`, existing KA/LAMMPS trajectory readers.

## Global Constraints

- Preserve `thermodynamic_claim_allowed = 0` in every generated summary.
- Use the existing KA epsilon, sigma, cutoff, periodic minimum-image, mass `m=1`, and Langevin conventions.
- Do not fit diffusion, NGP, scattering, event clocks, or thermodynamic observables.
- Do not stage or commit pre-existing unrelated dirty/untracked research files.
- Use `apply_patch` for manual edits and test-first development for every production function.
- Treat the saved-frame `dt=0.005 tau` stochastic comparison as a finite-step canary, not an infinitesimal equality.

---

### Task 1: Exact First Force-Generator Mode and Noise Covariance

**Files:**
- Modify: `src/ka_local_cage.py` near `ka_lj_force_and_isotropic_curvature`
- Modify: `tests/test_ka_replicates.py` near the existing KA force/Hessian tests

**Interfaces:**
- Consumes: positions `(particles, 3)`, velocities `(particles, 3)`, KA particle types, box lengths, selected target indices, friction, and temperature.
- Produces: `ka_lj_force_generator_observables(...) -> dict[str, np.ndarray]` with keys `force`, `force_generator`, and `force_generator_noise_covariance_rate`.
- Produces privately: `_ka_lj_target_pair_geometry(...) -> dict[str, np.ndarray]` with `pair_force`, `pair_hessian`, and `active` arrays for use by Task 2.

- [ ] **Step 1: Add the failing two-particle generator test**

Add the import and this test to `tests/test_ka_replicates.py`:

```python
def test_ka_force_generator_matches_directional_force_derivative_and_hessian_noise(self):
    positions = np.array([[0.0, 0.0, 0.0], [1.13, 0.17, -0.08]])
    velocities = np.array([[0.4, -0.2, 0.1], [-0.3, 0.5, -0.4]])
    particle_types = np.array([0, 1])
    box_lengths = np.array([20.0, 20.0, 20.0])
    target = np.array([0])
    result = ka_lj_force_generator_observables(
        positions,
        velocities=velocities,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
        friction=1.0,
        temperature=0.58,
    )
    step = 1e-6
    force_plus = ka_lj_force_and_isotropic_curvature(
        positions + step * velocities,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
    )[0]
    force_minus = ka_lj_force_and_isotropic_curvature(
        positions - step * velocities,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
    )[0]
    np.testing.assert_allclose(
        result["force_generator"],
        (force_plus - force_minus) / (2.0 * step),
        rtol=2e-8,
        atol=2e-8,
    )
    cluster = ka_local_cluster_hessian(
        positions,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_index=0,
        cluster_cutoff=5.0,
    )["hessian"]
    target_row = cluster[:3]
    expected_covariance_rate = 2.0 * 1.0 * 0.58 * target_row @ target_row.T
    np.testing.assert_allclose(
        result["force_generator_noise_covariance_rate"][0],
        expected_covariance_rate,
        rtol=1e-12,
        atol=1e-12,
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m unittest tests.test_ka_replicates.KAReplicatePreparationTests.test_ka_force_generator_matches_directional_force_derivative_and_hessian_noise
```

Expected: import failure because `ka_lj_force_generator_observables` does not exist.

- [ ] **Step 3: Implement the shared pair geometry and public first-generator API**

Add this structure in `src/ka_local_cage.py`, reusing the exact formulas already present in `ka_lj_local_energy_force_hessian`:

```python
def _ka_lj_target_pair_geometry(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    positions = np.asarray(positions, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (particles, 3)")
    if particle_types.shape != (len(positions),) or np.any((particle_types < 0) | (particle_types > 1)):
        raise ValueError("particle_types must be 0/1 and align with positions")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if target.ndim != 1 or not len(target) or np.any(target < 0) or np.any(target >= len(positions)):
        raise ValueError("target_indices must select valid particles")
    wrapped = np.mod(positions, box_lengths)
    displacement = wrapped[target, None, :] - wrapped[None, :, :]
    displacement -= box_lengths * np.rint(displacement / box_lengths)
    squared_distance = np.sum(displacement**2, axis=2)
    distance = np.sqrt(np.maximum(squared_distance, 1e-24))
    epsilon = _EPSILON[particle_types[target, None], particle_types[None, :]]
    sigma = _SIGMA[particle_types[target, None], particle_types[None, :]]
    active = (distance > 1e-10) & (distance < _CUTOFF_SCALE * sigma)
    sigma_over_r2 = (sigma / np.maximum(distance, 1e-12)) ** 2
    sigma_over_r6 = sigma_over_r2**3
    sigma_over_r12 = sigma_over_r6**2
    force_coefficient = 24.0 * epsilon * (2.0 * sigma_over_r12 - sigma_over_r6) / np.maximum(squared_distance, 1e-24)
    pair_force = (force_coefficient * active)[:, :, None] * displacement
    potential_prime_over_r = 24.0 * epsilon * (-2.0 * sigma_over_r12 + sigma_over_r6) / np.maximum(squared_distance, 1e-24) * active
    potential_second = 24.0 * epsilon * (26.0 * sigma_over_r12 - 7.0 * sigma_over_r6) / np.maximum(squared_distance, 1e-24) * active
    unit = displacement / np.maximum(distance[:, :, None], 1e-12)
    pair_hessian = (
        (potential_second - potential_prime_over_r)[:, :, None, None]
        * unit[:, :, :, None]
        * unit[:, :, None, :]
        + potential_prime_over_r[:, :, None, None] * np.eye(3)
    )
    return {"pair_force": pair_force, "pair_hessian": pair_hessian, "active": active}


def ka_lj_force_generator_observables(
    positions: np.ndarray,
    *,
    velocities: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    friction: float,
    temperature: float,
) -> dict[str, np.ndarray]:
    velocities = np.asarray(velocities, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    if velocities.shape != np.asarray(positions).shape or np.any(~np.isfinite(velocities)):
        raise ValueError("velocities must be finite and align with positions")
    if not math.isfinite(friction) or friction < 0.0 or not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("friction and temperature must be finite and nonnegative")
    geometry = _ka_lj_target_pair_geometry(
        positions,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
    )
    pair_hessian = geometry["pair_hessian"]
    relative_velocity = velocities[target, None, :] - velocities[None, :, :]
    force_generator = -np.sum(np.einsum("pnab,pnb->pna", pair_hessian, relative_velocity), axis=1)
    diagonal_hessian = np.sum(pair_hessian, axis=1)
    covariance_geometry = np.einsum("pab,pcb->pac", diagonal_hessian, diagonal_hessian)
    covariance_geometry += np.einsum("pnab,pncb->pac", pair_hessian, pair_hessian)
    return {
        "force": np.sum(geometry["pair_force"], axis=1),
        "force_generator": force_generator,
        "force_generator_noise_covariance_rate": 2.0 * friction * temperature * covariance_geometry,
    }
```

- [ ] **Step 4: Run focused and neighboring force tests**

Run:

```bash
python -m unittest \
  tests.test_ka_replicates.KAReplicatePreparationTests.test_ka_force_generator_matches_directional_force_derivative_and_hessian_noise \
  tests.test_ka_replicates.KAReplicatePreparationTests.test_ka_pair_force_and_isotropic_curvature_match_analytic_pair_formula \
  tests.test_ka_replicates.KAReplicatePreparationTests.test_local_cluster_hessian_is_symmetric_and_has_isolated_translation_mode
```

Expected: all selected tests pass.

- [ ] **Step 5: Record the task boundary without committing unrelated files**

Run:

```bash
git diff --check -- src/ka_local_cage.py tests/test_ka_replicates.py
git status --short -- src/ka_local_cage.py tests/test_ka_replicates.py
```

Expected: no whitespace errors; only the already-dirty target files are listed. Do not stage them because both contain pre-existing uncommitted research work.

### Task 2: Exact Second Generator Drift

**Files:**
- Modify: `src/ka_local_cage.py` after `ka_lj_force_generator_observables`
- Modify: `tests/test_ka_replicates.py`

**Interfaces:**
- Consumes: Task 1 `_ka_lj_target_pair_geometry` and `ka_lj_force_generator_observables`.
- Produces: `ka_lj_second_force_generator(...) -> np.ndarray` with shape `(targets, 3)`.

- [ ] **Step 1: Add a deterministic two-particle finite-difference test**

Add:

```python
def test_ka_second_force_generator_matches_first_generator_drift(self):
    positions = np.array([[0.0, 0.0, 0.0], [1.13, 0.17, -0.08]])
    velocities = np.array([[0.4, -0.2, 0.1], [-0.3, 0.5, -0.4]])
    particle_types = np.array([0, 1])
    box_lengths = np.array([20.0, 20.0, 20.0])
    target = np.array([0])
    result = ka_lj_second_force_generator(
        positions,
        velocities=velocities,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
        friction=1.0,
        directional_step=1e-5,
    )
    force = ka_lj_force_and_isotropic_curvature(
        positions,
        particle_types=particle_types,
        box_lengths=box_lengths,
    )[0]
    acceleration = force - velocities
    step = 2e-6
    plus = ka_lj_force_generator_observables(
        positions + step * velocities,
        velocities=velocities + step * acceleration,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
        friction=1.0,
        temperature=0.0,
    )["force_generator"]
    minus = ka_lj_force_generator_observables(
        positions - step * velocities,
        velocities=velocities - step * acceleration,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
        friction=1.0,
        temperature=0.0,
    )["force_generator"]
    np.testing.assert_allclose(result, (plus - minus) / (2.0 * step), rtol=2e-5, atol=2e-5)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m unittest tests.test_ka_replicates.KAReplicatePreparationTests.test_ka_second_force_generator_matches_first_generator_drift
```

Expected: import failure for `ka_lj_second_force_generator`.

- [ ] **Step 3: Implement the second generator without a full `N x N` force allocation**

```python
def ka_lj_second_force_generator(
    positions: np.ndarray,
    *,
    velocities: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    friction: float,
    directional_step: float = 1e-5,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    if not math.isfinite(directional_step) or directional_step <= 0.0:
        raise ValueError("directional_step must be finite and positive")
    geometry = _ka_lj_target_pair_geometry(
        positions,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
    )
    plus = ka_lj_force_generator_observables(
        positions + directional_step * velocities,
        velocities=velocities,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
        friction=friction,
        temperature=0.0,
    )["force_generator"]
    minus = ka_lj_force_generator_observables(
        positions - directional_step * velocities,
        velocities=velocities,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
        friction=friction,
        temperature=0.0,
    )["force_generator"]
    position_drift = (plus - minus) / (2.0 * directional_step)
    velocity_drift = np.empty_like(position_drift)
    pair_hessian = geometry["pair_hessian"]
    for slot, particle in enumerate(target):
        neighbors = np.flatnonzero(geometry["active"][slot])
        selected = np.concatenate(([particle], neighbors))
        selected_force = ka_lj_force_and_isotropic_curvature(
            positions,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=selected,
        )[0]
        acceleration = selected_force - friction * velocities[selected]
        velocity_drift[slot] = -np.sum(
            np.einsum("jab,jb->ja", pair_hessian[slot, neighbors], acceleration[0] - acceleration[1:]),
            axis=0,
        )
    return position_drift + velocity_drift
```

- [ ] **Step 4: Verify directional-step convergence**

Add this assertion to the same test:

```python
step_results = [
    ka_lj_second_force_generator(
        positions,
        velocities=velocities,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
        friction=1.0,
        directional_step=directional_step,
    )
    for directional_step in (3e-6, 1e-5, 3e-5)
]
reference_norm = float(np.linalg.norm(step_results[1]))
for current in (step_results[0], step_results[2]):
    self.assertLess(float(np.linalg.norm(current - step_results[1]) / reference_norm), 2e-5)
```

Run the focused unittest from Step 2. Expected: PASS.

- [ ] **Step 5: Run Task 1 and Task 2 tests together**

Expected: both generator tests and neighboring Hessian tests pass; `git diff --check` reports no errors.

### Task 3: Reusable Increment Diagnostic

**Files:**
- Modify: `src/ka_local_cage.py`
- Modify: `tests/test_ka_replicates.py`

**Interfaces:**
- Consumes: arrays of `F`, `LF`, `L2F`, covariance-rate matrices, and physical frame time.
- Produces: `force_generator_increment_diagnostic(...) -> dict[str, float]`.

- [ ] **Step 1: Add a failing synthetic diagnostic test**

```python
def test_force_generator_increment_diagnostic_recovers_exact_derivative_and_known_innovation(self):
    frame_time = 0.1
    force = np.zeros((5, 3))
    force[:, 0] = [0.0, 0.0, 0.2, 0.0, 0.0]
    force_generator = np.zeros((5, 3))
    force_generator[:, 0] = [0.0, 1.0, 0.0, -1.0, 0.0]
    second_force_generator = np.zeros((5, 3))
    covariance_rate = np.repeat((np.eye(3) / frame_time)[None, :, :], 5, axis=0)

    result = force_generator_increment_diagnostic(
        force,
        force_generator,
        second_force_generator,
        covariance_rate,
        frame_time=frame_time,
    )

    self.assertLess(float(result["force_derivative_relative_l2"]), 1e-12)
    self.assertAlmostEqual(float(result["force_derivative_correlation"]), 1.0, delta=1e-12)
    self.assertAlmostEqual(float(result["innovation_trace_variance_ratio"]), 1.0 / 3.0, delta=1e-12)
    self.assertAlmostEqual(float(result["innovation_mean_squared_mahalanobis"]), 1.0, delta=1e-12)
    self.assertLess(float(result["innovation_normalized_mean"]), 1e-12)
    self.assertEqual(float(result["thermodynamic_claim_allowed"]), 0.0)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m unittest tests.test_ka_replicates.KAReplicatePreparationTests.test_force_generator_increment_diagnostic_recovers_exact_derivative_and_known_innovation
```

Expected: missing `force_generator_increment_diagnostic` import.

- [ ] **Step 3: Implement the diagnostic**

```python
def force_generator_increment_diagnostic(
    force: np.ndarray,
    force_generator: np.ndarray,
    second_force_generator: np.ndarray,
    force_generator_noise_covariance_rate: np.ndarray,
    *,
    frame_time: float,
) -> dict[str, float]:
    force = np.asarray(force, dtype=float)
    generator = np.asarray(force_generator, dtype=float)
    second = np.asarray(second_force_generator, dtype=float)
    covariance_rate = np.asarray(force_generator_noise_covariance_rate, dtype=float)
    if force.ndim != 2 or force.shape[1] != 3 or generator.shape != force.shape or second.shape != force.shape:
        raise ValueError("force and generator arrays must have matching (frames, 3) shapes")
    if covariance_rate.shape != (len(force), 3, 3) or frame_time <= 0.0:
        raise ValueError("covariance rates and frame_time must align with force frames")
    centered_force_derivative = (force[2:] - force[:-2]) / (2.0 * frame_time)
    centered_generator = generator[1:-1]
    derivative_difference = centered_force_derivative - centered_generator
    derivative_norm = float(np.linalg.norm(centered_generator))
    innovation = generator[1:] - generator[:-1] - frame_time * second[:-1]
    predicted_covariance = frame_time * covariance_rate[:-1]
    squared_mahalanobis = np.asarray(
        [value @ np.linalg.solve(covariance, value) for value, covariance in zip(innovation, predicted_covariance)]
    )
    innovation_rms = math.sqrt(float(np.mean(np.sum(innovation**2, axis=1))))
    return {
        "force_derivative_relative_l2": float(np.linalg.norm(derivative_difference) / derivative_norm),
        "force_derivative_correlation": float(np.corrcoef(centered_force_derivative.reshape(-1), centered_generator.reshape(-1))[0, 1]),
        "innovation_trace_variance_ratio": float(
            np.sum(innovation**2) / np.sum(np.trace(predicted_covariance, axis1=1, axis2=2))
        ),
        "innovation_mean_squared_mahalanobis": float(np.mean(squared_mahalanobis)),
        "innovation_normalized_mean": float(np.linalg.norm(np.mean(innovation, axis=0)) / innovation_rms),
        "thermodynamic_claim_allowed": 0.0,
    }
```

- [ ] **Step 4: Run the focused diagnostic and generator tests**

Expected: all pass with no warnings.

### Task 4: Existing-Trajectory Generator Validation CLI

**Files:**
- Create: `scripts/analyze_ka_force_generator.py`
- Create: `data/renewal_cage_ka_force_generator_T058_summary.csv`
- Create: `data/renewal_cage_ka_force_generator_T058_curve.csv`
- Modify: `docs/microscopic-frozen-minimum-response.md`

**Interfaces:**
- Consumes: four `tmp/isoconfigurational_force_velocity_T058/clone_*/trajectory.lammpstrj` files through `load_lammps_custom_trajectory`.
- Produces: clone-level force-derivative metrics, a preregistered 101-increment stochastic canary, and auditable curves for tagged particle id `821`.

- [ ] **Step 1: Create the CLI using only tested source APIs**

Create `scripts/analyze_ka_force_generator.py` with this implementation:

```python
#!/usr/bin/env python3
"""Validate KA force-generator observables on full-state Langevin paths."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_local_cage import (  # noqa: E402
    force_generator_increment_diagnostic,
    ka_lj_force_generator_observables,
    ka_lj_second_force_generator,
)
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", type=Path, nargs="+")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--target-id", type=int, default=821)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--integration-time-step", type=float, default=0.001)
    parser.add_argument("--directional-step", type=float, default=1e-5)
    parser.add_argument("--stochastic-frame-limit", type=int, default=102)
    args = parser.parse_args()
    if args.target_id < 1 or args.temperature < 0.0 or args.friction < 0.0:
        raise ValueError("target-id, temperature, and friction must be physical")
    if args.integration_time_step <= 0.0 or args.directional_step <= 0.0 or args.stochastic_frame_limit < 4:
        raise ValueError("time steps and stochastic-frame-limit must be positive")

    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    metric_names = (
        "force_derivative_relative_l2",
        "force_derivative_correlation",
        "innovation_trace_variance_ratio",
        "innovation_mean_squared_mahalanobis",
        "innovation_normalized_mean",
    )
    for clone_index, path in enumerate(args.trajectories, start=1):
        trajectory = load_lammps_custom_trajectory(path)
        if "velocities" not in trajectory:
            raise ValueError(f"{path}: trajectory must contain velocities")
        positions = np.asarray(trajectory["unwrapped_positions"])
        velocities = np.asarray(trajectory["velocities"])
        particle_types = np.asarray(trajectory["particle_types"])
        box_lengths = np.asarray(trajectory["box_lengths"])
        timesteps = np.asarray(trajectory["timesteps"])
        target_index = args.target_id - 1
        if target_index >= positions.shape[1]:
            raise ValueError(f"{path}: target id is outside the atom table")
        intervals = np.diff(timesteps)
        if len(intervals) == 0 or not np.all(intervals == intervals[0]):
            raise ValueError(f"{path}: saved timesteps must be uniform")
        frame_time = float(intervals[0]) * args.integration_time_step
        target = np.array([target_index])
        force: list[np.ndarray] = []
        generator: list[np.ndarray] = []
        covariance_rate: list[np.ndarray] = []
        for frame_positions, frame_velocities in zip(positions, velocities):
            observables = ka_lj_force_generator_observables(
                frame_positions,
                velocities=frame_velocities,
                particle_types=particle_types,
                box_lengths=box_lengths,
                target_indices=target,
                friction=args.friction,
                temperature=args.temperature,
            )
            force.append(observables["force"][0])
            generator.append(observables["force_generator"][0])
            covariance_rate.append(observables["force_generator_noise_covariance_rate"][0])
        force_array = np.asarray(force)
        generator_array = np.asarray(generator)
        covariance_array = np.asarray(covariance_rate)
        stochastic_frames = min(args.stochastic_frame_limit, len(positions))
        second_array = np.asarray(
            [
                ka_lj_second_force_generator(
                    positions[frame],
                    velocities=velocities[frame],
                    particle_types=particle_types,
                    box_lengths=box_lengths,
                    target_indices=target,
                    friction=args.friction,
                    directional_step=args.directional_step,
                )[0]
                for frame in range(stochastic_frames)
            ]
        )
        diagnostic = force_generator_increment_diagnostic(
            force_array[:stochastic_frames],
            generator_array[:stochastic_frames],
            second_array,
            covariance_array[:stochastic_frames],
            frame_time=frame_time,
        )
        summary_rows.append(
            {
                "record": "clone",
                "clone_index": clone_index,
                "trajectory": str(path),
                "target_id": args.target_id,
                "frame_time": frame_time,
                "frame_count": len(positions),
                "stochastic_frame_count": stochastic_frames,
                "directional_step": args.directional_step,
                **diagnostic,
            }
        )
        for frame, (timestep, force_value, generator_value) in enumerate(
            zip(timesteps, force_array, generator_array)
        ):
            second_value: np.ndarray | None = second_array[frame] if frame < stochastic_frames else None
            curve_rows.append(
                {
                    "clone_index": clone_index,
                    "time": float(timestep * args.integration_time_step),
                    **{f"force_{axis}": float(force_value[index]) for index, axis in enumerate("xyz")},
                    **{f"force_generator_{axis}": float(generator_value[index]) for index, axis in enumerate("xyz")},
                    **{
                        f"second_force_generator_{axis}": "" if second_value is None else float(second_value[index])
                        for index, axis in enumerate("xyz")
                    },
                    "thermodynamic_claim_allowed": 0,
                }
            )
    aggregate: dict[str, object] = {
        "record": "aggregate",
        "clone_index": "",
        "trajectory": "",
        "target_id": args.target_id,
        "frame_time": summary_rows[0]["frame_time"],
        "frame_count": summary_rows[0]["frame_count"],
        "stochastic_frame_count": summary_rows[0]["stochastic_frame_count"],
        "directional_step": args.directional_step,
        "thermodynamic_claim_allowed": 0,
    }
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in summary_rows])
        aggregate[metric] = float(np.mean(values))
        aggregate[f"{metric}_standard_error"] = (
            float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        )
    summary_rows.append(aggregate)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve_rows)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the CLI on one clone as a canary**

Run:

```bash
python scripts/analyze_ka_force_generator.py \
  tmp/isoconfigurational_force_velocity_T058/clone_001/trajectory.lammpstrj \
  --output-prefix tmp/ka_force_generator_canary
```

Expected: force-derivative correlation above `0.995`, relative L2 below `0.06`, finite covariance metrics, and no thermodynamic claim.

- [ ] **Step 3: Run all four clones and write durable outputs**

Run:

```bash
python scripts/analyze_ka_force_generator.py \
  tmp/isoconfigurational_force_velocity_T058/clone_001/trajectory.lammpstrj \
  tmp/isoconfigurational_force_velocity_T058/clone_002/trajectory.lammpstrj \
  tmp/isoconfigurational_force_velocity_T058/clone_003/trajectory.lammpstrj \
  tmp/isoconfigurational_force_velocity_T058/clone_004/trajectory.lammpstrj \
  --output-prefix data/renewal_cage_ka_force_generator_T058
```

Expected: four clone rows and one aggregate row. Preserve exact measured values rather than copying feasibility-audit numbers into code.

- [ ] **Step 4: Document the derivation and measured boundary**

Append a section headed `## Exact Force-Generator Layer` with this mathematical core:

````markdown
For the tagged conservative force `F_i=-grad_i U_KA`, the backward
underdamped Langevin generator gives

```text
G_i = L F_i = -sum_j H_ij v_j
    = -sum_(j != i) K_ij (v_i-v_j).
```

Since `G_i` is linear in all velocities, Ito's formula fixes both its drift
and infinitesimal conditional noise covariance,

```text
dG_i = L^2 F_i dt - sqrt(2 gamma T) sum_j H_ij dW_j,
Cov[dG_i | R,V] / dt = 2 gamma T sum_j H_ij H_ij^T.
```

These results validate the first microscopic generator layer and its
state-dependent multiplicative noise at the saved-frame resolution. They do
not establish an autonomous cage-relative response, renewal clock, Kramers
escape law, macro-observable closure, or thermodynamic glass transition.
````

Between the equations and boundary paragraph, add a five-row table copied
from the aggregate CSV. The rows, in this exact order, are
`force_derivative_relative_l2`, `force_derivative_correlation`,
`innovation_trace_variance_ratio`,
`innovation_mean_squared_mahalanobis`, and `innovation_normalized_mean`.
Round displayed values to six significant digits and retain the full-precision
CSV as the authoritative data.

Link [`analysis script`](../scripts/analyze_ka_force_generator.py),
[`summary`](../data/renewal_cage_ka_force_generator_T058_summary.csv), and
[`curves`](../data/renewal_cage_ka_force_generator_T058_curve.csv).

- [ ] **Step 5: Verify the complete first subproject**

Run:

```bash
python -m unittest discover -s tests
python -m py_compile src/ka_local_cage.py scripts/analyze_ka_force_generator.py
git diff --check
```

Expected: the full suite passes, both Python files compile, and no whitespace errors are reported.

- [ ] **Step 6: Record outputs without staging unrelated research state**

Run `git status --short` and report the exact modified/new files. Do not commit implementation changes while `src/ka_local_cage.py` and the shared test file contain pre-existing uncommitted work outside this plan.

## Deferred Follow-Up Plans

After this plan passes, create separate plans for:

1. sequential low-disk matched `+/-` full-state response extraction for eight clones and two epsilon values;
2. generator-constrained tangent-response fitting with leave-one-clone-out, cross-epsilon, temporal, and stability gates;
3. cage-relative autonomous coordinates, orthogonal-force law, first passage, and the full `D`/NGP/multi-k `F_s`/SE/heterogeneity closure.
