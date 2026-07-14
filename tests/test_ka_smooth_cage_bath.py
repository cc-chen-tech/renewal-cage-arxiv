import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SmoothCageBathTests(unittest.TestCase):
    def test_augmented_state_aligns_force_history_and_cage_vectors(self):
        from ka_smooth_cage_bath import assemble_augmented_hankel_state

        frames, particles, modes = 8, 2, 3
        velocity = np.arange(frames * particles * 3, dtype=float).reshape(
            frames, particles, 3
        )
        force_modes = np.full((6, particles, modes, 3), 7.0)
        relative_position = np.full_like(velocity, 11.0)
        relative_velocity = np.full_like(velocity, 13.0)

        state = assemble_augmented_hankel_state(
            velocity,
            force_modes,
            relative_position=relative_position,
            relative_velocity=relative_velocity,
            include_position=True,
            include_relative_velocity=True,
        )

        self.assertEqual(state.shape, (6, particles, 1 + modes + 2, 3))
        np.testing.assert_array_equal(state[:, :, 0], velocity[2:])
        np.testing.assert_array_equal(state[:, :, 1:4], force_modes)
        np.testing.assert_array_equal(state[:, :, 4], relative_position[2:])
        np.testing.assert_array_equal(state[:, :, 5], relative_velocity[2:])

    def test_trapezoidal_kinematic_error_vanishes_for_linear_velocity(self):
        from ka_smooth_cage_bath import cage_kinematic_diagnostic

        time = np.arange(101, dtype=float) * 0.01
        initial = np.array([0.2, -0.1, 0.3])
        acceleration = np.array([0.4, 0.2, -0.3])
        velocity = initial + time[:, None] * acceleration
        position = (
            time[:, None] * initial
            + 0.5 * time[:, None] ** 2 * acceleration
        )[:, None, :]
        velocity = velocity[:, None, :]

        result = cage_kinematic_diagnostic(
            position,
            velocity,
            frame_time=0.01,
        )

        self.assertLess(result["normalized_trapezoidal_kinematic_error"], 1e-12)
        self.assertLess(result["kinematic_residual_rms"], 1e-14)

    def test_mode_resolved_lag_profile_matches_generic_maximum(self):
        from ka_slow_force_bath import (
            fit_covariance_contracted_model,
            heldout_state_diagnostics,
        )
        from ka_smooth_cage_bath import heldout_residual_lag_profile

        rng = np.random.default_rng(20260716)
        state = rng.normal(size=(500, 4, 5, 3))
        for frame in range(1, len(state)):
            state[frame] += 0.7 * state[frame - 1]
        model = fit_covariance_contracted_model([state[:300]])
        held = state[300:]

        profile = heldout_residual_lag_profile(model, held, maximum_lag=16)
        generic = heldout_state_diagnostics(model, held)

        self.assertEqual(profile["maximum_by_residual_mode"].shape, (5,))
        self.assertAlmostEqual(
            float(profile["maximum_overall_residual_lag_correlation"]),
            generic["maximum_held_residual_lag_correlation"],
            places=12,
        )

        from ka_smooth_cage_bath import heldout_linear_velocity_diagnostic

        linear_velocity = heldout_linear_velocity_diagnostic(
            model,
            held,
            velocity_weights=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(
            linear_velocity["heldout_linear_velocity_r_squared"],
            generic["heldout_velocity_r_squared"],
            places=12,
        )

    def test_real_data_cli_exposes_fixed_cage_hankel_gates(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_smooth_cage_hankel_bath.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        for required in (
            "--cache-directory",
            "--cage-cache-directory",
            "--history-length",
            "--mode-count",
            "--lag-improvement-fraction",
            "--half-window",
            "--phop-threshold",
        ):
            self.assertIn(required, completed.stdout)

    def test_decomposed_drift_cli_exposes_fixed_split_bath_gates(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_decomposed_cage_drift_bath.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        for required in (
            "--drift-cache-directory",
            "--directional-step",
            "--sensitivity-steps",
            "--split-mode-counts",
            "--primary-split-mode-count",
            "--lag-improvement-fraction",
        ):
            self.assertIn(required, completed.stdout)

        source = (
            ROOT / "scripts" / "analyze_ka_decomposed_cage_drift_bath.py"
        ).read_text()
        for required in (
            "rerun {source_trajectory} dump x y z ix iy iz vx vy vz box yes",
            '"maximum_force_cache_absolute_error"',
            '"force_cache_relative_rms_error"',
            '"force_cache_correlation"',
            '"lammps_binary_sha256"',
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
