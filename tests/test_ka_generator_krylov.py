import csv
import hashlib
import json
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


class SecondGeneratorKrylovTests(unittest.TestCase):
    @staticmethod
    def synthetic_second_generator_states():
        frame_time = 0.005
        friction = 1.0
        third_block = np.zeros((3, 15))
        for block, coefficient in enumerate((-0.2, -0.5, -1.0, -2.0, -3.0)):
            third_block[:, 3 * block : 3 * (block + 1)] = coefficient * np.eye(3)
        generator = np.zeros((15, 15))
        generator[0:3, 3:6] = np.eye(3)
        generator[3:6, 3:6] = -friction * np.eye(3)
        generator[3:6, 6:9] = np.eye(3)
        generator[6:9, 9:12] = np.eye(3)
        generator[9:12, 12:15] = np.eye(3)
        generator[12:15] = third_block
        identity = np.eye(15)
        transition = np.linalg.solve(
            identity - 0.5 * frame_time * generator,
            identity + 0.5 * frame_time * generator,
        )
        rng = np.random.default_rng(90210)
        states = np.empty((8, 201, 15))
        states[:, 0] = rng.normal(size=(8, 15))
        for frame in range(1, states.shape[1]):
            states[:, frame] = np.einsum(
                "ab,pb->pa", transition, states[:, frame - 1]
            )
        return states, third_block, frame_time, friction

    def test_assemble_second_generator_state_preserves_microscopic_order(self):
        from ka_generator_krylov import assemble_second_generator_state

        state = np.arange(2 * 7 * 12, dtype=float).reshape(2, 7, 12)
        second = np.arange(2 * 7 * 3, dtype=float).reshape(2, 7, 3)

        result = assemble_second_generator_state(state, second)

        self.assertEqual(result.shape, (2, 7, 15))
        np.testing.assert_array_equal(result[..., :12], state)
        np.testing.assert_array_equal(result[..., 12:15], second)

    def test_assemble_second_generator_state_rejects_misaligned_second_mode(self):
        from ka_generator_krylov import assemble_second_generator_state

        with self.assertRaisesRegex(ValueError, "second_force_response"):
            assemble_second_generator_state(
                np.zeros((7, 12)),
                np.zeros((6, 3)),
            )

    def test_weak_form_fit_recovers_exact_second_generator_chain(self):
        from ka_generator_krylov import fit_second_generator_constrained_response

        states, exact_block, frame_time, friction = (
            self.synthetic_second_generator_states()
        )

        fit = fit_second_generator_constrained_response(
            states,
            frame_time=frame_time,
            friction=friction,
            fit_frames=41,
        )

        np.testing.assert_allclose(
            fit["fitted_third_generator_block"],
            exact_block,
            rtol=2e-3,
            atol=2e-3,
        )
        self.assertEqual(fit["design_rank"], 15.0)
        self.assertLess(fit["heldout_position_relative_l2_error"], 2e-3)
        self.assertLessEqual(fit["spectral_radius"], 1.0 + 1e-6)

    def test_free_transition_and_propagation_recover_synthetic_chain(self):
        from ka_generator_krylov import (
            fit_free_second_generator_transition,
            propagate_linear_response,
        )

        states, _, _, _ = self.synthetic_second_generator_states()

        fit = fit_free_second_generator_transition(states, fit_frames=41)
        predicted = propagate_linear_response(
            fit["transition_matrix"],
            states[0, 0],
            states.shape[1],
        )

        self.assertEqual(fit["design_rank"], 15.0)
        self.assertEqual(predicted.shape, states[0].shape)
        np.testing.assert_array_equal(predicted[0], states[0, 0])
        np.testing.assert_allclose(predicted, states[0], rtol=1e-9, atol=1e-9)

    def test_second_generator_residual_vanishes_for_exact_weak_chain(self):
        from ka_generator_krylov import second_generator_residual_diagnostic

        states, exact_block, frame_time, _ = self.synthetic_second_generator_states()

        diagnostic = second_generator_residual_diagnostic(
            states,
            exact_block,
            frame_time=frame_time,
        )

        self.assertLess(diagnostic["residual_relative_l2"], 1e-10)
        self.assertEqual(diagnostic["maximum_abs_residual_state_correlation"], 0.0)

    def test_second_generator_cli_exposes_preregistered_gates(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_second_generator_response.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--fit-times", completed.stdout)
        self.assertIn("--horizons", completed.stdout)
        self.assertIn("--linearity-tolerance", completed.stdout)
        self.assertIn("--expected-member-count", completed.stdout)

    def test_second_generator_cli_validates_manifest_and_keeps_claims_limited(self):
        states, _, frame_time, _ = self.synthetic_second_generator_states()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            time = np.arange(states.shape[1]) * frame_time
            for member in range(1, 9):
                response = states[member - 1]
                for epsilon in (0.001, 0.002):
                    for sign in (-1, 1):
                        path = root / f"member{member}_epsilon{epsilon}_sign{sign}.npz"
                        np.savez_compressed(
                            path,
                            time=time,
                            position=sign * epsilon * response[:, 0:3],
                            velocity=sign * epsilon * response[:, 3:6],
                            force=sign * epsilon * response[:, 6:9],
                            force_generator=sign * epsilon * response[:, 9:12],
                            second_force_generator=sign * epsilon * response[:, 12:15],
                            potential_protocol=np.asarray("ka_lj_c3_switch"),
                            thermodynamic_claim_allowed=0.0,
                        )
                        records.append(
                            {
                                "member_index": member,
                                "epsilon": epsilon,
                                "sign": sign,
                                "velocity_seed": 100 + member,
                                "langevin_seed": 200 + member,
                                "path": path.name,
                                "path_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "frame_count": len(time),
                                "frame_time_tau": frame_time,
                                "potential_protocol": "ka_lj_c3_switch",
                                "fit_parameters_from_macro_observables": False,
                                "thermodynamic_claim_allowed": False,
                            }
                        )
            manifest = {
                "protocol": "full_KA_common_noise_generator_response",
                "potential_protocol": "ka_lj_c3_switch",
                "member_count": 8,
                "record_count": len(records),
                "epsilons": [0.001, 0.002],
                "saved_frame_interval_tau": frame_time,
                "duration_tau": float(time[-1]),
                "friction": 1.0,
                "fit_parameters_from_macro_observables": False,
                "thermodynamic_claim_allowed": False,
                "records": records,
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            prefix = root / "result"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "analyze_ka_second_generator_response.py"),
                    str(manifest_path),
                    "--output-prefix",
                    str(prefix),
                    "--fit-times",
                    "0.20",
                    "--horizons",
                    "0.20",
                    "0.50",
                    "1.00",
                    "--expected-member-count",
                    "8",
                ],
                check=True,
            )

            with prefix.with_name(prefix.name + "_summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["model"] for row in rows if row["record"] == "leave_one_member_out"},
                {
                    "first_generator_constrained",
                    "second_generator_constrained",
                    "free_second_generator_transition",
                },
            )
            for row in rows:
                self.assertEqual(row["autonomous_stochastic_single_particle_gle_allowed"], "0.0")
                self.assertEqual(row["event_clock_claim_allowed"], "0.0")
                self.assertEqual(row["kramers_escape_claim_allowed"], "0.0")
                self.assertEqual(row["thermodynamic_claim_allowed"], "0.0")


if __name__ == "__main__":
    unittest.main()
