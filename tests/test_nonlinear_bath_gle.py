import sys
import subprocess
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


class NonlinearBathGleTests(unittest.TestCase):
    @staticmethod
    def controls(**updates):
        from nonlinear_bath_gle import NonlinearBathControls

        values = {
            "temperature": 0.58,
            "friction": 1.0,
            "period": 1.0,
            "barrier": 1.74,
            "rates": np.array([0.20, 1.00]),
            "amplitudes": np.array([1.00, 0.55]),
            "modulation": np.array([0.45, 0.25]),
            "phases": np.array([0.0, 0.5 * np.pi]),
            "time_step": 0.001,
        }
        values.update(updates)
        return NonlinearBathControls(**values)

    def test_periodic_potential_gradient_matches_centered_difference(self):
        from nonlinear_bath_gle import (
            periodic_potential,
            periodic_potential_gradient,
        )

        position = np.array([-0.37, -0.11, 0.0, 0.19, 0.44])
        step = 1e-7
        numerical = (
            periodic_potential(position + step, barrier=1.74, period=1.0)
            - periodic_potential(position - step, barrier=1.74, period=1.0)
        ) / (2.0 * step)
        actual = periodic_potential_gradient(position, barrier=1.74, period=1.0)
        np.testing.assert_allclose(actual, numerical, rtol=2e-9, atol=2e-9)

    def test_periodic_coupling_uses_every_frozen_mode(self):
        from nonlinear_bath_gle import periodic_coupling

        controls = self.controls()
        position = np.array([0.0, 0.25])
        actual = periodic_coupling(position, controls=controls)
        expected = controls.amplitudes[None] * (
            1.0
            + controls.modulation[None]
            * np.cos(
                2.0 * np.pi * position[:, None] / controls.period
                + controls.phases[None]
            )
        )
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(actual.shape, (2, 2))

    def test_antisymmetric_coupling_energy_error_is_second_order(self):
        from nonlinear_bath_gle import nonlinear_bath_step

        position = np.array([0.13])
        momentum = np.array([0.8])
        auxiliary = np.array([[0.4, -0.7]])
        normal_p = np.zeros_like(momentum)
        normal_z = np.zeros_like(auxiliary)
        initial_energy = 0.5 * (
            momentum[0] ** 2 + float(np.sum(auxiliary[0] ** 2))
        )
        errors = []
        for step in (1e-3, 5e-4):
            controls = self.controls(
                temperature=0.0,
                friction=0.0,
                barrier=0.0,
                rates=np.zeros(2),
                modulation=np.zeros(2),
                time_step=step,
            )
            result = nonlinear_bath_step(
                position,
                momentum,
                auxiliary,
                normal_p=normal_p,
                normal_z=normal_z,
                controls=controls,
            )
            final_energy = 0.5 * (
                float(result["momentum"][0] ** 2)
                + float(np.sum(result["auxiliary"][0] ** 2))
            )
            errors.append(abs(final_energy - initial_energy))
        self.assertGreater(errors[0], 0.0)
        self.assertLess(errors[1] / errors[0], 0.26)

    def test_exact_ou_reconstruction_matches_supplied_noise_path(self):
        from nonlinear_bath_gle import (
            nonlinear_bath_step,
            reconstruct_auxiliary_path,
        )

        controls = self.controls()
        rng = np.random.default_rng(20260811)
        steps, trajectories = 25, 4
        normal_p = rng.normal(size=(steps, trajectories))
        normal_z = rng.normal(size=(steps, trajectories, 2))
        position = np.linspace(-0.2, 0.2, trajectories)
        momentum = rng.normal(size=trajectories)
        auxiliary = rng.normal(size=(trajectories, 2))
        positions = [position.copy()]
        momenta = [momentum.copy()]
        auxiliaries = [auxiliary.copy()]
        for index in range(steps):
            result = nonlinear_bath_step(
                position,
                momentum,
                auxiliary,
                normal_p=normal_p[index],
                normal_z=normal_z[index],
                controls=controls,
            )
            position = np.asarray(result["position"])
            momentum = np.asarray(result["momentum"])
            auxiliary = np.asarray(result["auxiliary"])
            positions.append(position.copy())
            momenta.append(momentum.copy())
            auxiliaries.append(auxiliary.copy())
        reconstructed = reconstruct_auxiliary_path(
            np.asarray(auxiliaries[0]),
            positions=np.asarray(positions[:-1]),
            momenta=np.asarray(momenta[:-1]),
            normal_increments=normal_z,
            controls=controls,
        )
        np.testing.assert_allclose(
            reconstructed,
            np.asarray(auxiliaries),
            rtol=0.0,
            atol=5e-14,
        )

    def test_eliminated_kernel_matches_mode_sum_and_keeps_claims_closed(self):
        from nonlinear_bath_gle import eliminated_memory_kernel

        controls = self.controls()
        left = np.array([0.10, 0.20])
        right = np.array([-0.15, 0.25])
        lag = 0.37
        result = eliminated_memory_kernel(
            left,
            right,
            lag=lag,
            controls=controls,
        )
        coupling_left = controls.amplitudes[None] * (
            1.0
            + controls.modulation[None]
            * np.cos(
                2.0 * np.pi * left[:, None] / controls.period
                + controls.phases[None]
            )
        )
        coupling_right = controls.amplitudes[None] * (
            1.0
            + controls.modulation[None]
            * np.cos(
                2.0 * np.pi * right[:, None] / controls.period
                + controls.phases[None]
            )
        )
        expected = np.sum(
            coupling_left
            * np.exp(-controls.rates[None] * lag)
            * coupling_right,
            axis=1,
        )
        np.testing.assert_allclose(result["kernel"], expected)
        self.assertEqual(result["exact_nonlinear_bath_elimination_supported"], 0.0)
        self.assertEqual(result["autonomous_single_particle_gle_allowed"], 0.0)
        self.assertEqual(result["thermodynamic_claim_allowed"], 0.0)

    def test_fokker_planck_audit_proves_gibbs_stationarity_term_by_term(self):
        from nonlinear_bath_gle import gibbs_stationarity_audit

        controls = self.controls()
        rng = np.random.default_rng(20260815)
        position = rng.uniform(-1.0, 1.0, size=17)
        momentum = rng.normal(size=17)
        auxiliary = rng.normal(size=(17, 2))
        result = gibbs_stationarity_audit(
            position,
            momentum,
            auxiliary,
            controls=controls,
        )
        for key in (
            "hamiltonian_energy_rate",
            "antisymmetric_bath_energy_rate",
            "conservative_phase_space_divergence",
            "momentum_thermostat_stationarity_residual",
            "auxiliary_thermostat_stationarity_residual",
        ):
            np.testing.assert_allclose(result[key], 0.0, rtol=0.0, atol=2e-14)
        self.assertLess(result["maximum_normalized_stationarity_residual"], 2e-14)
        self.assertEqual(
            result["periodic_quotient_gibbs_invariant_density_derived"],
            1.0,
        )
        self.assertEqual(result["unwrapped_position_gibbs_probability_allowed"], 0.0)
        self.assertNotIn("gibbs_invariant_density_derived", result)
        self.assertEqual(result["thermodynamic_claim_allowed"], 0.0)

    def test_remote_simulator_cli_exposes_only_frozen_modes(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "simulate_nonlinear_bath_elimination.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for phrase in (
            "--output-path",
            "--mode",
            "canary",
            "canary-half-step",
            "production",
            "null-constant-coupling",
            "null-no-bath",
            "--checkpoint-interval",
            "--resume",
        ):
            self.assertIn(phrase, completed.stdout)

    def test_frozen_remote_protocol_keeps_primary_and_null_streams_paired(self):
        from simulate_nonlinear_bath_elimination import frozen_simulation_protocol

        primary = frozen_simulation_protocol("production")
        constant = frozen_simulation_protocol("null-constant-coupling")
        no_bath = frozen_simulation_protocol("null-no-bath")
        self.assertEqual(primary["seed"], 20260811)
        self.assertEqual(primary["trajectory_count"], 256)
        self.assertEqual(primary["burn_in_steps"], 100000)
        self.assertEqual(primary["production_steps"], 400000)
        self.assertEqual(primary["event_sample_stride"], 10)
        self.assertEqual(primary["equilibrium_sample_stride"], 100)
        self.assertEqual(primary["potential_amplitude"], 1.74)
        self.assertEqual(primary["physical_barrier_height"], 3.48)
        self.assertEqual(constant["seed"], primary["seed"])
        self.assertEqual(no_bath["seed"], primary["seed"])
        np.testing.assert_array_equal(
            constant["controls"].modulation,
            np.zeros(2),
        )
        np.testing.assert_array_equal(
            no_bath["controls"].amplitudes,
            np.zeros(2),
        )

    def test_checkpoint_metadata_rejects_any_provenance_change(self):
        from simulate_nonlinear_bath_elimination import (
            checkpoint_metadata,
            validate_checkpoint_metadata,
        )

        expected = checkpoint_metadata(
            frozen_simulation_protocol="production",
            source_sha256="abc123",
            requested_step_count=500000,
        )
        self.assertIn("gle_source_sha256", expected)
        self.assertEqual(expected["potential_amplitude"], 1.74)
        self.assertEqual(expected["physical_barrier_height"], 3.48)
        validate_checkpoint_metadata(dict(expected), expected)
        for key, replacement in (
            ("frozen_simulation_protocol", "canary"),
            ("source_sha256", "changed"),
            ("gle_source_sha256", "changed-dependency"),
            ("requested_step_count", 499999.0),
            ("seed", 1.0),
            ("physical_barrier_height", 1.74),
        ):
            changed = dict(expected)
            changed[key] = replacement
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "checkpoint provenance"):
                    validate_checkpoint_metadata(changed, expected)

    def test_simulation_guard_requires_remote_environment_marker(self):
        from simulate_nonlinear_bath_elimination import (
            require_remote_execution,
            run_simulation,
        )

        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "remote-only"):
                require_remote_execution()
            with mock.patch(
                "simulate_nonlinear_bath_elimination._sample_equilibrium_positions",
                side_effect=AssertionError("remote guard bypassed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "remote-only"):
                    run_simulation(
                        Path("must-not-be-created.npz"),
                        mode="canary",
                        checkpoint_interval=100,
                        resume=False,
                    )
        with mock.patch.dict(
            "os.environ",
            {"RENEWAL_CAGE_REMOTE_COMPUTE": "1"},
            clear=True,
        ):
            require_remote_execution()
            with mock.patch(
                "simulate_nonlinear_bath_elimination._sample_equilibrium_positions",
                side_effect=AssertionError("invalid interval entered simulation"),
            ):
                with self.assertRaisesRegex(ValueError, "checkpoint interval"):
                    run_simulation(
                        Path("must-not-be-created.npz"),
                        mode="canary",
                        checkpoint_interval=0,
                        resume=False,
                    )

    def test_checkpoint_payload_rejects_inconsistent_or_nonfinite_state(self):
        from simulate_nonlinear_bath_elimination import (
            checkpoint_metadata,
            frozen_simulation_protocol,
            validate_checkpoint_payload,
        )

        mode = "canary"
        protocol = frozen_simulation_protocol(mode)
        provenance = checkpoint_metadata(
            frozen_simulation_protocol=mode,
            source_sha256="abc123",
            requested_step_count=int(protocol["requested_step_count"]),
        )
        trajectories = int(protocol["trajectory_count"])
        steps = int(protocol["requested_step_count"])
        modes = len(protocol["controls"].rates)
        payload = {
            **provenance,
            "completed_step_count": float(steps),
            "stored_event_sample_count": float(steps + 1),
            "stored_equilibrium_sample_count": float(steps + 1),
            "cache_complete": 1.0,
            "current_position": np.zeros(trajectories),
            "current_momentum": np.zeros(trajectories),
            "current_auxiliary": np.zeros((trajectories, modes)),
            "event_positions": np.zeros((steps + 1, trajectories)),
            "equilibrium_positions": np.zeros((steps + 1, trajectories)),
            "equilibrium_momenta": np.zeros((steps + 1, trajectories)),
            "equilibrium_auxiliary": np.zeros(
                (steps + 1, trajectories, modes)
            ),
            "canary_normal_p": np.zeros((steps, trajectories)),
            "canary_normal_z": np.zeros((steps, trajectories, modes)),
            "rng_state_json": np.asarray(
                __import__("json").dumps(
                    np.random.default_rng(3).bit_generator.state
                )
            ),
            "exact_nonlinear_bath_elimination_supported": 0.0,
            "synthetic_bath_level_fdt_replay_supported": 0.0,
            "synthetic_delayed_hazard_emerges": 0.0,
            "real_ka_position_dependent_kernel_authorized": 0.0,
            "autonomous_single_particle_gle_allowed": 0.0,
            "complete_event_clock_closure_allowed": 0.0,
            "kramers_escape_claim_allowed": 0.0,
            "spatial_facilitation_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        validated = validate_checkpoint_payload(
            payload,
            provenance=provenance,
            protocol=protocol,
        )
        self.assertEqual(validated["completed_step_count"], steps)
        self.assertEqual(validated["event_sample_count"], steps + 1)

        corruptions = {
            "fractional completed count": ("completed_step_count", steps - 0.5),
            "event count mismatch": ("stored_event_sample_count", steps),
            "bad state shape": ("current_position", np.zeros(trajectories - 1)),
            "nonfinite state": (
                "current_momentum",
                np.full(trajectories, np.nan),
            ),
            "truncated normals": (
                "canary_normal_p",
                np.zeros((steps - 1, trajectories)),
            ),
            "invalid rng json": ("rng_state_json", np.asarray("not-json")),
            "open broad claim": ("thermodynamic_claim_allowed", 1.0),
        }
        for label, (key, value) in corruptions.items():
            changed = dict(payload)
            changed[key] = value
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "checkpoint payload"):
                    validate_checkpoint_payload(
                        changed,
                        provenance=provenance,
                        protocol=protocol,
                    )


if __name__ == "__main__":
    unittest.main()
