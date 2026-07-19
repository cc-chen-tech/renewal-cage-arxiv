import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.ka_parent_completion import (
    build_completion_record,
    file_sha256,
    snapshot_parent_completion,
    validate_completion_record,
)


ROOT = Path(__file__).resolve().parents[1]


def write_dump(path: Path, *, frames: int = 3, atoms: int = 2) -> None:
    chunks = []
    for frame in range(frames):
        chunks.extend(
            [
                "ITEM: TIMESTEP\n",
                f"{frame * 1000}\n",
                "ITEM: NUMBER OF ATOMS\n",
                f"{atoms}\n",
                "ITEM: BOX BOUNDS pp pp pp\n",
                "0 10\n0 10\n0 10\n",
                "ITEM: ATOMS id type x y z ix iy iz\n",
            ]
        )
        for atom in range(1, atoms + 1):
            chunks.append(f"{atom} 1 {atom + frame / 10} 0 0 0 0 0\n")
    path.write_text("".join(chunks))


class ParentCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        write_dump(self.run / "trajectory.lammpstrj")
        (self.run / "final.restart").write_bytes(b"restart")
        (self.run / "log.lammps").write_text("Loop time of 1 on 1 procs\n")
        (self.run / "parent_manifest.json").write_text(
            json.dumps({"parent_id": "parent-1"}) + "\n"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_unknown_exit_is_blocked_even_when_outputs_are_complete(self):
        row = snapshot_parent_completion(
            parent_id="parent-1",
            pid=999_999_999,
            output_directory=self.run,
            expected_frame_count=3,
            expected_atom_count=2,
            expected_first_timestep=0,
            expected_last_timestep=2000,
        )
        self.assertIsNone(row["exit_code"])
        self.assertEqual(row["completion_state"], "blocked_missing_observed_exit_code")
        self.assertEqual(row["scientific_ingestion_allowed"], 0)
        self.assertEqual(row["production_frame_count"], 3)
        self.assertEqual(len(row["trajectory_sha256"]), 64)
        self.assertEqual(row["final_restart_present"], 1)

    def test_explicit_zero_exit_and_complete_outputs_are_verified(self):
        (self.run / "run.exitcode").write_text("0\n")
        row = snapshot_parent_completion(
            parent_id="parent-1",
            pid=999_999_999,
            output_directory=self.run,
            expected_frame_count=3,
            expected_atom_count=2,
            expected_first_timestep=0,
            expected_last_timestep=2000,
        )
        self.assertEqual(row["exit_code"], 0)
        self.assertEqual(row["completion_state"], "complete_verified")
        self.assertEqual(row["scientific_ingestion_allowed"], 1)
        self.assertEqual(
            row["final_restart_sha256"], hashlib.sha256(b"restart").hexdigest()
        )

    def test_nonzero_exit_or_error_signature_fails_closed(self):
        (self.run / "run.exitcode").write_text("7\n")
        (self.run / "log.lammps").write_text("ERROR: Lost atoms\n")
        row = snapshot_parent_completion(
            parent_id="parent-1",
            pid=999_999_999,
            output_directory=self.run,
            expected_frame_count=3,
            expected_atom_count=2,
            expected_first_timestep=0,
            expected_last_timestep=2000,
        )
        self.assertEqual(row["completion_state"], "failed_explicit_nonzero_exit")
        self.assertEqual(row["error_match_count"], 2)
        self.assertEqual(row["scientific_ingestion_allowed"], 0)

    def test_aggregate_record_validates_and_keeps_claims_closed(self):
        (self.run / "run.exitcode").write_text("0\n")
        row = snapshot_parent_completion(
            parent_id="parent-1",
            pid=999_999_999,
            output_directory=self.run,
            expected_frame_count=3,
            expected_atom_count=2,
            expected_first_timestep=0,
            expected_last_timestep=2000,
        )
        record = build_completion_record([row], acquisition_manifest_sha256="a" * 64)
        validated = validate_completion_record(record)
        self.assertEqual(validated["completion_state"], "complete_verified")
        self.assertTrue(all(value == 0 for value in validated["claim_flags"].values()))

    def test_committed_watcher_deployment_joins_exact_source_and_completion_hashes(self):
        deployment = json.loads(
            (
                ROOT
                / "data"
                / "renewal_cage_ka_prl_T045_parent_completion_watcher_deployment.json"
            ).read_text()
        )
        completion = (
            ROOT
            / "data"
            / "renewal_cage_ka_prl_T045_parent_acquisition_completion.json"
        )
        self.assertEqual(
            deployment["remote_watcher_module_sha256"],
            file_sha256(ROOT / "src" / "ka_parent_completion.py"),
        )
        self.assertEqual(
            deployment["remote_watcher_script_sha256"],
            file_sha256(ROOT / "scripts" / "watch_ka_parent_completion.py"),
        )
        self.assertEqual(
            deployment["local_completion_status_sha256"], file_sha256(completion)
        )
        self.assertFalse(deployment["watcher_modified_run_outputs"])
        self.assertFalse(deployment["credentials_persisted"])
        self.assertFalse(deployment["scientific_ingestion_allowed"])


if __name__ == "__main__":
    unittest.main()
