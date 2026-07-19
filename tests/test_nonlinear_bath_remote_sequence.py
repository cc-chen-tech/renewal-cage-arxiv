import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class NonlinearBathRemoteSequenceTests(unittest.TestCase):
    def test_cli_exposes_only_remote_sequence_paths(self):
        import subprocess

        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_nonlinear_bath_remote_sequence.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for phrase in ("--output-dir", "--artifact-prefix"):
            self.assertIn(phrase, completed.stdout)

    def test_proc_audit_rejects_l3p_and_existing_nonlinear_workers(self):
        from run_nonlinear_bath_remote_sequence import find_conflicting_processes

        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            commands = {
                101: "python3 simulate_ka_l3p_generator_quotient.py --clone 4",
                102: "python3 simulate_nonlinear_bath_elimination.py --mode canary",
                103: "sshd: root@pts/0",
                999: "python3 run_nonlinear_bath_remote_sequence.py",
            }
            for pid, command in commands.items():
                process = proc / str(pid)
                process.mkdir()
                (process / "cmdline").write_bytes(command.replace(" ", "\0").encode())
            conflicts = find_conflicting_processes(proc, current_pid=999)
        self.assertEqual([row[0] for row in conflicts], [101, 102])
        self.assertIn("l3p", conflicts[0][1])
        self.assertIn("nonlinear_bath", conflicts[1][1])

    def test_frozen_commands_are_sequential_and_resume_existing_cache(self):
        from run_nonlinear_bath_remote_sequence import build_frozen_commands

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "nonlinear_bath_canary.npz").touch()
            commands = build_frozen_commands(
                output,
                artifact_prefix=output / "gate",
                python_executable="/usr/bin/python3",
            )
        self.assertEqual(len(commands), 7)
        labels = [label for label, _ in commands]
        self.assertEqual(
            labels,
            [
                "canary",
                "canary-half-step",
                "canary-preflight",
                "production",
                "null-constant-coupling",
                "null-no-bath",
                "analysis",
            ],
        )
        canary = commands[0][1]
        self.assertIn("--resume", canary)
        self.assertNotIn("--resume", commands[1][1])
        self.assertIn("check_nonlinear_bath_canary.py", " ".join(commands[2][1]))
        self.assertIn("analyze_nonlinear_bath_elimination.py", " ".join(commands[-1][1]))
        for (_, earlier), (_, later) in zip(commands, commands[1:]):
            self.assertIsNot(earlier, later)

    def test_remote_marker_is_required_before_process_or_file_work(self):
        from unittest import mock

        from run_nonlinear_bath_remote_sequence import run_remote_sequence

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch(
                "run_nonlinear_bath_remote_sequence.find_conflicting_processes",
                side_effect=AssertionError("process audit ran before remote guard"),
            ):
                with self.assertRaisesRegex(RuntimeError, "remote-only"):
                    run_remote_sequence(
                        Path("must-not-exist"),
                        artifact_prefix=Path("must-not-exist/gate"),
                    )

    def test_artifact_prefix_must_stay_inside_the_locked_output_directory(self):
        from run_nonlinear_bath_remote_sequence import validate_remote_paths

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            valid_output, valid_prefix = validate_remote_paths(
                output,
                output / "gate",
            )
            self.assertEqual(valid_prefix.parent, valid_output)
            with self.assertRaisesRegex(ValueError, "artifact prefix"):
                validate_remote_paths(output, Path(directory) / "outside" / "gate")

    def test_manifest_claim_schema_rejects_missing_nonbinary_or_open_flags(self):
        from run_nonlinear_bath_remote_sequence import validate_claim_flags

        flags = {
            "exact_nonlinear_bath_elimination_supported": "1.0",
            "synthetic_bath_level_fdt_replay_supported": "1.0",
            "synthetic_delayed_hazard_emerges": "0.0",
            "real_ka_position_dependent_kernel_authorized": "1.0",
            "real_ka_kernel_identifiability_test_required": "1.0",
            "positive_prony_kernel_identified_in_ka": "0.0",
            "finite_auxiliary_rank_identified_in_ka": "0.0",
            "oscillatory_matrix_bath_authorized": "0.0",
            "autonomous_single_particle_gle_allowed": "0.0",
            "complete_event_clock_closure_allowed": "0.0",
            "kramers_escape_claim_allowed": "0.0",
            "spatial_facilitation_claim_allowed": "0.0",
            "thermodynamic_claim_allowed": "0.0",
        }
        validated = validate_claim_flags(flags)
        self.assertEqual(validated["exact_nonlinear_bath_elimination_supported"], 1.0)
        self.assertEqual(validated["thermodynamic_claim_allowed"], 0.0)

        for label, key, value in (
            ("missing", "synthetic_delayed_hazard_emerges", None),
            ("nonbinary", "synthetic_delayed_hazard_emerges", "0.5"),
            ("broad open", "thermodynamic_claim_allowed", "1.0"),
            ("family open", "positive_prony_kernel_identified_in_ka", "1.0"),
            ("test disabled", "real_ka_kernel_identifiability_test_required", "0.0"),
        ):
            changed = dict(flags)
            if value is None:
                del changed[key]
            else:
                changed[key] = value
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "claim schema"):
                    validate_claim_flags(changed)


if __name__ == "__main__":
    unittest.main()
