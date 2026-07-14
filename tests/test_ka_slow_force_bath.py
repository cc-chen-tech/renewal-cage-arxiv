import sys
import subprocess
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SlowForceBathTests(unittest.TestCase):
    def test_force_hankel_uses_newest_to_oldest_order(self):
        from ka_slow_force_bath import assemble_force_hankel

        force = np.repeat(np.arange(6, dtype=float)[:, None, None], 3, axis=2)
        result = assemble_force_hankel(force, history_length=3)

        self.assertEqual(result.shape, (4, 1, 3, 3))
        np.testing.assert_array_equal(result[:, 0, 0], [[2, 1, 0], [3, 2, 1], [4, 3, 2], [5, 4, 3]])

    def test_force_hankel_rejects_short_or_nonfinite_input(self):
        from ka_slow_force_bath import assemble_force_hankel

        with self.assertRaisesRegex(ValueError, "history_length"):
            assemble_force_hankel(np.zeros((2, 1, 3)), history_length=3)
        bad = np.zeros((4, 1, 3))
        bad[1, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            assemble_force_hankel(bad, history_length=2)

    def test_training_hankel_basis_projects_only_supplied_force_histories(self):
        from ka_slow_force_bath import fit_force_hankel_basis, project_force_hankel

        time = np.arange(40, dtype=float)
        training = np.zeros((40, 2, 3))
        training[:, :, 0] = np.sin(0.1 * time)[:, None]
        training[:, :, 1] = np.cos(0.1 * time)[:, None]
        training[:, :, 2] = 0.5 * np.sin(0.2 * time)[:, None]
        held = training + 100.0

        fit = fit_force_hankel_basis([training], history_length=8, mode_count=3)
        projected_training = project_force_hankel(training, fit)
        projected_held = project_force_hankel(held, fit)

        self.assertEqual(fit["basis"].shape, (8, 3))
        self.assertEqual(projected_training.shape, (33, 2, 3, 3))
        self.assertGreater(np.linalg.norm(projected_held - projected_training), 100.0)
        np.testing.assert_allclose(fit["basis"].T @ fit["basis"], np.eye(3), atol=1e-12)

    def test_covariance_contracted_fit_is_stable_and_stationary(self):
        from ka_slow_force_bath import fit_covariance_contracted_model

        rng = np.random.default_rng(7182)
        exact_transition = np.array([[0.94, 0.08], [-0.05, 0.90]])
        state = np.zeros((1200, 4, 2, 3))
        for frame in range(1, len(state)):
            state[frame] = np.einsum("ab,pbc->pac", exact_transition, state[frame - 1])
            state[frame] += rng.normal(scale=0.15, size=state[frame].shape)

        fit = fit_covariance_contracted_model([state[:800]], maximum_singular_value=0.999)

        self.assertLessEqual(fit["spectral_radius"], 1.0 + 1e-10)
        self.assertLess(fit["stationary_covariance_relative_error"], 1e-10)
        self.assertGreaterEqual(np.linalg.eigvalsh(fit["innovation_covariance"]).min(), -1e-10)
        np.testing.assert_allclose(fit["transition_matrix"], exact_transition, atol=0.03)

    def test_slow_hankel_mode_recovers_multiscale_force_state(self):
        from ka_slow_force_bath import (
            fit_covariance_contracted_model,
            fit_force_hankel_basis,
            project_force_hankel,
        )

        rng = np.random.default_rng(20260714)
        frames = 1800
        target_count = 6
        slow = np.zeros((frames, target_count, 3))
        force = np.zeros_like(slow)
        velocity = np.zeros_like(slow)
        for frame in range(1, frames):
            slow[frame] = 0.995 * slow[frame - 1] + rng.normal(scale=0.04, size=slow[frame].shape)
            force[frame] = slow[frame] + rng.normal(scale=0.30, size=slow[frame].shape)
            velocity[frame] = 0.96 * velocity[frame - 1] + 0.025 * slow[frame - 1]
            velocity[frame] += rng.normal(scale=0.03, size=velocity[frame].shape)

        split = 1100
        basis = fit_force_hankel_basis([force[:split]], history_length=32, mode_count=2)
        modes = project_force_hankel(force, basis)
        aligned_velocity = velocity[31:]
        state = np.concatenate([aligned_velocity[:, :, None, :], modes], axis=2)
        fit = fit_covariance_contracted_model([state[: split - 31]])

        held = state[split - 31 :]
        centered = held - fit["state_mean"][None, None, :, None]
        predicted = np.einsum("ab,tpbc->tpac", fit["transition_matrix"], centered[:-1])
        observed = centered[1:]
        velocity_error = np.mean((observed[:, :, 0] - predicted[:, :, 0]) ** 2)
        zero_error = np.mean(observed[:, :, 0] ** 2)

        self.assertGreater(float(basis["captured_variance_fraction"]), 0.55)
        self.assertLess(velocity_error / zero_error, 0.20)
        self.assertLessEqual(fit["spectral_radius"], 1.0 + 1e-10)

    def test_simulation_preserves_fitted_stationary_covariance(self):
        from ka_slow_force_bath import fit_covariance_contracted_model, simulate_slow_bath

        rng = np.random.default_rng(909)
        state = rng.normal(size=(2000, 3, 3, 3))
        for frame in range(1, len(state)):
            state[frame] += 0.75 * state[frame - 1]
        fit = fit_covariance_contracted_model([state])
        simulated = simulate_slow_bath(fit, step_count=400, simulation_count=3000, seed=91)
        centered = simulated[-1] - fit["state_mean"][None, :, None]
        covariance = np.einsum("sac,sbc->ab", centered, centered) / (centered.shape[0] * 3)
        relative_error = np.linalg.norm(covariance - fit["stationary_covariance"]) / np.linalg.norm(
            fit["stationary_covariance"]
        )

        self.assertLess(relative_error, 0.08)

    def test_heldout_diagnostic_and_trapezoid_displacement(self):
        from ka_slow_force_bath import (
            fit_covariance_contracted_model,
            heldout_state_diagnostics,
            state_paths_to_displacements,
        )

        rng = np.random.default_rng(44)
        state = np.zeros((800, 2, 2, 3))
        transition = np.array([[0.92, 0.04], [-0.02, 0.88]])
        for frame in range(1, len(state)):
            state[frame] = np.einsum("ab,pbc->pac", transition, state[frame - 1])
            state[frame] += rng.normal(scale=0.08, size=state[frame].shape)
        fit = fit_covariance_contracted_model([state[:500]])
        diagnostic = heldout_state_diagnostics(fit, state[500:])

        self.assertGreater(diagnostic["heldout_state_r_squared"], 0.7)
        self.assertGreater(diagnostic["heldout_velocity_r_squared"], 0.7)
        self.assertLess(diagnostic["maximum_held_residual_state_correlation"], 0.15)
        self.assertLess(diagnostic["maximum_held_residual_lag_correlation"], 0.15)

        paths = np.zeros((4, 1, 1, 3))
        paths[:, 0, 0, 0] = [0.0, 1.0, 2.0, 3.0]
        displacement = state_paths_to_displacements(paths, frame_time=0.5)
        np.testing.assert_allclose(displacement[:, 0, 0], [0.0, 0.25, 1.0, 2.25])

    def test_streaming_displacement_matches_full_state_simulation(self):
        from ka_slow_force_bath import (
            fit_covariance_contracted_model,
            simulate_slow_bath,
            simulate_slow_bath_displacements,
            state_paths_to_displacements,
        )

        rng = np.random.default_rng(808)
        state = rng.normal(size=(500, 2, 3, 3))
        for frame in range(1, len(state)):
            state[frame] += 0.6 * state[frame - 1]
        fit = fit_covariance_contracted_model([state])
        full = simulate_slow_bath(fit, step_count=30, simulation_count=40, seed=71)
        streamed = simulate_slow_bath_displacements(
            fit,
            step_count=30,
            simulation_count=40,
            frame_time=0.01,
            seed=71,
        )

        np.testing.assert_allclose(streamed, state_paths_to_displacements(full, frame_time=0.01))

    def test_real_data_cli_exposes_fixed_slow_bath_and_event_gates(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_hankel_slow_force_bath.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        for required in (
            "--history-length",
            "--primary-mode-count",
            "--mode-counts",
            "--half-window",
            "--phop-threshold",
            "--simulation-count",
            "--cache-directory",
        ):
            self.assertIn(required, completed.stdout)


if __name__ == "__main__":
    unittest.main()
