import sys
import subprocess
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class StateDependentMemoryTests(unittest.TestCase):
    @staticmethod
    def synthetic_state_dependent_bath():
        rng = np.random.default_rng(20260715)
        state = np.zeros((1800, 8, 3, 3))
        for frame in range(1, len(state)):
            previous = state[frame - 1]
            energy = np.mean(previous[:, 1:] ** 2, axis=(1, 2))
            state[frame, :, 0] = 0.86 * previous[:, 0] + 0.07 * previous[:, 1]
            state[frame, :, 0] -= 2.00 * (energy - 0.055)[:, None] * previous[:, 0]
            state[frame, :, 1] = 0.965 * previous[:, 1]
            state[frame, :, 2] = 0.82 * previous[:, 2]
            state[frame, :, 0] += rng.normal(scale=0.03, size=state[frame, :, 0].shape)
            state[frame, :, 1:] += rng.normal(scale=0.08, size=state[frame, :, 1:].shape)
        return state

    def test_state_invariants_are_rotation_invariant(self):
        from ka_state_dependent_memory import fit_state_invariant_scaling, state_invariants

        rng = np.random.default_rng(91)
        state = rng.normal(size=(100, 4, 5, 3))
        scaling = fit_state_invariant_scaling([state])
        angle = 0.71
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated = np.einsum("tpmc,dc->tpmd", state, rotation)

        np.testing.assert_allclose(
            state_invariants(state, scaling),
            state_invariants(rotated, scaling),
            atol=1e-12,
        )

    def test_bilinear_model_improves_known_state_dependent_bath(self):
        from ka_slow_force_bath import fit_covariance_contracted_model, heldout_state_diagnostics
        from ka_state_dependent_memory import (
            bilinear_heldout_diagnostics,
            fit_bilinear_state_dependent_model,
        )

        state = self.synthetic_state_dependent_bath()
        training = state[:1200]
        held = state[1200:]
        linear = fit_covariance_contracted_model([training])
        linear_diagnostic = heldout_state_diagnostics(linear, held)
        bilinear = fit_bilinear_state_dependent_model(
            [training],
            invariant_names=("bath_energy",),
        )
        bilinear_diagnostic = bilinear_heldout_diagnostics(bilinear, held)

        self.assertGreater(
            bilinear_diagnostic["heldout_velocity_r_squared"],
            linear_diagnostic["heldout_velocity_r_squared"] + 0.001,
        )
        self.assertLess(
            bilinear_diagnostic["heldout_velocity_residual_mean_squared"],
            0.98 * linear_diagnostic["heldout_velocity_residual_mean_squared"],
        )

    def test_full_bilinear_design_uses_energy_and_power_blocks(self):
        from ka_state_dependent_memory import fit_bilinear_state_dependent_model

        state = self.synthetic_state_dependent_bath()[:500]
        fit = fit_bilinear_state_dependent_model(
            [state],
            invariant_names=("bath_energy", "velocity_bath_power"),
        )

        self.assertEqual(fit["coefficient_matrix"].shape, (9, 3))
        self.assertEqual(tuple(fit["invariant_names"]), ("bath_energy", "velocity_bath_power"))
        self.assertEqual(float(fit["fit_parameters_from_macro_observables"]), 0.0)
        self.assertEqual(float(fit["thermodynamic_claim_allowed"]), 0.0)

    def test_real_data_cli_exposes_preregistered_bilinear_gate(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_bilinear_state_dependent_memory.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        for required in (
            "--cache-directory",
            "--history-length",
            "--mode-count",
            "--ridge-relative",
            "--lag-improvement-fraction",
        ):
            self.assertIn(required, completed.stdout)


if __name__ == "__main__":
    unittest.main()
