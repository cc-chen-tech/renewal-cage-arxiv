import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


class NonlinearBathAnalysisTests(unittest.TestCase):
    def test_cli_requires_the_complete_frozen_cache_grid(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_nonlinear_bath_elimination.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for phrase in (
            "--canary-cache",
            "--half-step-cache",
            "--production-cache",
            "--constant-coupling-cache",
            "--no-bath-cache",
            "--output-prefix",
        ):
            self.assertIn(phrase, completed.stdout)

    def test_bundle_header_gate_requires_all_modes_and_exact_source_hashes(self):
        from analyze_nonlinear_bath_elimination import validate_bundle_headers

        modes = {
            "canary": 16,
            "canary-half-step": 16,
            "production": 256,
            "null-constant-coupling": 256,
            "null-no-bath": 256,
        }
        headers = {
            mode: {
                "frozen_simulation_protocol": mode,
                "source_sha256": "simulator-sha",
                "gle_source_sha256": "gle-sha",
                "cache_complete": 1.0,
                "trajectory_count": float(count),
            }
            for mode, count in modes.items()
        }
        result = validate_bundle_headers(
            headers,
            simulator_sha256="simulator-sha",
            gle_sha256="gle-sha",
        )
        self.assertEqual(result["cache_grid_complete"], 1.0)
        self.assertEqual(result["training_trajectory_count"], 128.0)
        self.assertEqual(result["held_trajectory_count"], 128.0)

        missing = dict(headers)
        del missing["null-no-bath"]
        with self.assertRaisesRegex(ValueError, "cache bundle"):
            validate_bundle_headers(
                missing,
                simulator_sha256="simulator-sha",
                gle_sha256="gle-sha",
            )
        changed = {mode: dict(row) for mode, row in headers.items()}
        changed["production"]["source_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "cache bundle"):
            validate_bundle_headers(
                changed,
                simulator_sha256="simulator-sha",
                gle_sha256="gle-sha",
            )

    def test_complete_cache_loader_recomputes_frozen_canary_provenance(self):
        from analyze_nonlinear_bath_elimination import load_complete_cache
        from simulate_nonlinear_bath_elimination import (
            checkpoint_metadata,
            file_sha256,
            frozen_simulation_protocol,
        )

        protocol = frozen_simulation_protocol("canary")
        steps = int(protocol["requested_step_count"])
        trajectories = int(protocol["trajectory_count"])
        mode_count = len(protocol["controls"].rates)
        provenance = checkpoint_metadata(
            frozen_simulation_protocol="canary",
            source_sha256=file_sha256(
                ROOT / "scripts" / "simulate_nonlinear_bath_elimination.py"
            ),
            requested_step_count=steps,
        )
        payload = {
            **provenance,
            "completed_step_count": float(steps),
            "stored_event_sample_count": float(steps + 1),
            "stored_equilibrium_sample_count": float(steps + 1),
            "cache_complete": 1.0,
            "current_position": np.zeros(trajectories),
            "current_momentum": np.zeros(trajectories),
            "current_auxiliary": np.zeros((trajectories, mode_count)),
            "event_positions": np.zeros((steps + 1, trajectories)),
            "equilibrium_positions": np.zeros((steps + 1, trajectories)),
            "equilibrium_momenta": np.zeros((steps + 1, trajectories)),
            "equilibrium_auxiliary": np.zeros(
                (steps + 1, trajectories, mode_count)
            ),
            "canary_normal_p": np.zeros((steps, trajectories)),
            "canary_normal_z": np.zeros((steps, trajectories, mode_count)),
            "rng_state_json": np.asarray(
                json.dumps(np.random.default_rng(11).bit_generator.state)
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
        with tempfile.TemporaryDirectory() as directory:
            valid_path = Path(directory) / "valid.npz"
            np.savez_compressed(valid_path, **payload)
            loaded = load_complete_cache(valid_path, expected_mode="canary")
            self.assertEqual(loaded["completed_step_count"], steps)

            changed = dict(payload)
            changed["barrier"] = 1.75
            invalid_path = Path(directory) / "invalid.npz"
            np.savez_compressed(invalid_path, **changed)
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                load_complete_cache(invalid_path, expected_mode="canary")

    def test_wait_table_adds_one_right_censored_interval_per_trajectory(self):
        from analyze_nonlinear_bath_elimination import event_wait_table

        records = [
            {"trajectory_index": 0.0, "event_time": 1.0},
            {"trajectory_index": 0.0, "event_time": 3.0},
            {"trajectory_index": 1.0, "event_time": 2.5},
        ]
        table = event_wait_table(records, horizon=4.0, trajectory_count=3)
        np.testing.assert_allclose(
            table["waiting_time"],
            np.array([1.0, 2.0, 1.0, 2.5, 1.5, 4.0]),
        )
        np.testing.assert_array_equal(
            table["censored"],
            np.array([False, False, True, False, True, True]),
        )
        np.testing.assert_array_equal(
            table["trajectory_index"],
            np.array([0, 0, 0, 1, 1, 2]),
        )

    def test_hazard_scoring_uses_only_training_fit_and_fixed_survival_grid(self):
        from analyze_nonlinear_bath_elimination import score_hazard_models

        waits = []
        censored = []
        owners = []
        for trajectory in range(256):
            waits.extend([1.0 + 0.01 * (trajectory % 5), 3.0])
            censored.extend([False, True])
            owners.extend([trajectory, trajectory])
        result = score_hazard_models(
            {
                "waiting_time": np.asarray(waits),
                "censored": np.asarray(censored),
                "trajectory_index": np.asarray(owners),
            },
            bootstrap_count=8,
            bootstrap_seed=20260812,
        )
        self.assertEqual(result["fit_uses_held_samples"], 0.0)
        self.assertEqual(result["training_trajectory_count"], 128.0)
        self.assertEqual(result["held_trajectory_count"], 128.0)
        np.testing.assert_allclose(
            result["survival_time"],
            np.linspace(0.0, 100.0, 201),
        )
        self.assertEqual(result["bootstrap_resamples"], 8.0)
        self.assertGreaterEqual(result["delay_time_bootstrap_ci95_low"], 1e-3)

    def test_svg_states_unclipped_axis_semantics_and_claim_boundary(self):
        from analyze_nonlinear_bath_elimination import render_gate_svg

        svg = render_gate_svg(
            equilibrium_rows=[
                {"metric": "momentum", "normalized_error": 7.0},
                {"metric": "position", "normalized_error": 0.5},
            ],
            replay_rows=[
                {"lag_time": 0.01, "normalized_error": 3.0},
                {"lag_time": 1.0, "normalized_error": 0.2},
            ],
            survival_time=np.array([0.0, 50.0, 100.0]),
            empirical_survival=np.array([1.0, 0.4, 0.1]),
            constant_survival=np.array([1.0, 0.5, 0.2]),
            delayed_survival=np.array([1.0, 0.42, 0.11]),
        )
        for phrase in (
            "normalized equilibrium error",
            "bath-level FDT replay normalized error",
            "held survival probability",
            "time (tau)",
            "tolerance = 1",
            "broad microscopic and thermodynamic claims remain closed",
        ):
            self.assertIn(phrase, svg)
        self.assertNotIn("clipped", svg.lower())
        self.assertNotIn("clipPath", svg)


if __name__ == "__main__":
    unittest.main()
