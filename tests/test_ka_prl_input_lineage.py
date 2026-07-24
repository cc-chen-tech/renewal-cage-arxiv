import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.ka_prl_input_lineage import (
    bind_parent_lineage,
    trajectory_identities_from_ensemble,
)


class PrlInputLineageBindingTests(unittest.TestCase):
    @staticmethod
    def provenance_rows():
        return [
            {
                "temperature": "0.45",
                "replicate": "1",
                "source_doi": "10.0/example",
                "source_sha256": "a" * 64,
                "source_frame_index": "5000",
                "velocity_seed": "45117",
            },
            {
                "temperature": "0.58",
                "replicate": "1",
                "source_doi": "10.0/example",
                "source_sha256": "b" * 64,
                "source_frame_index": "3000",
                "velocity_seed": "58117",
            },
        ]

    @staticmethod
    def trajectory_rows():
        return [
            {
                "temperature": "0.45",
                "replicate": "1",
                "trajectory_sha256": "c" * 64,
                "trajectory_size_bytes": "123",
                "trajectory_hash_scope": "complete_file",
            },
            {
                "temperature": "0.58",
                "replicate": "1",
                "trajectory_sha256": "d" * 64,
                "trajectory_size_bytes": "456",
                "trajectory_hash_scope": "complete_file",
            },
        ]

    def test_binds_parent_and_complete_trajectory_hash_without_changing_science(self):
        rows = [
            {
                "temperature": "0.45",
                "replicate": "1.0",
                "observed_msd": "0.125",
                "positive_memory_closure_claim_allowed": "0",
            }
        ]
        bound = bind_parent_lineage(
            rows,
            provenance_rows=self.provenance_rows()[:1],
            trajectory_rows=self.trajectory_rows()[:1],
            table_kind="temperature",
        )
        self.assertEqual(bound[0]["observed_msd"], "0.125")
        self.assertEqual(bound[0]["parent_id"], f"10.0/example:{'a' * 64}")
        self.assertEqual(bound[0]["source_sha256"], "a" * 64)
        self.assertEqual(bound[0]["trajectory_sha256"], "c" * 64)
        self.assertEqual(bound[0]["trajectory_size_bytes"], 123)
        self.assertEqual(bound[0]["trajectory_hash_scope"], "complete_file")
        self.assertEqual(bound[0]["positive_memory_closure_claim_allowed"], "0")

    def test_environment_group_maps_to_frozen_temperature_parent(self):
        rows = [
            {"temperature_group": "low", "replicate": "1", "block_size": "20"},
            {"temperature_group": "high", "replicate": "1", "block_size": "20"},
        ]
        bound = bind_parent_lineage(
            rows,
            provenance_rows=self.provenance_rows(),
            trajectory_rows=self.trajectory_rows(),
            table_kind="temperature_group",
        )
        self.assertEqual([row["source_sha256"] for row in bound], ["a" * 64, "b" * 64])

    def test_rejects_conflicting_embedded_identity_or_uncovered_restart(self):
        rows = [
            {
                "temperature": "0.45",
                "replicate": "1",
                "parent_id": "wrong-parent",
            }
        ]
        with self.assertRaisesRegex(ValueError, "conflicts"):
            bind_parent_lineage(
                rows,
                provenance_rows=self.provenance_rows()[:1],
                trajectory_rows=self.trajectory_rows()[:1],
                table_kind="temperature",
            )

    def test_ensemble_loader_hashes_complete_trajectory_after_manifest_joins(self):
        provenance = self.provenance_rows()[:1]
        with tempfile.TemporaryDirectory() as directory:
            ensemble = Path(directory)
            child = ensemble / "replicate_01"
            child.mkdir()
            (ensemble / "ensemble_manifest.json").write_text(
                json.dumps(
                    {
                        "temperature": 0.45,
                        "source_doi": "10.0/example",
                        "source_sha256": "a" * 64,
                        "replicates": [
                            {
                                "replicate": 1,
                                "directory": "replicate_01",
                                "source_frame_index": 5000,
                                "velocity_seed": 45117,
                            }
                        ],
                    }
                )
            )
            (child / "manifest.json").write_text(
                json.dumps(
                    {
                        "temperature": 0.45,
                        "source_doi": "10.0/example",
                        "source_sha256": "a" * 64,
                        "source_frame_index": 5000,
                        "velocity_seed": 45117,
                    }
                )
            )
            trajectory = child / "trajectory.lammpstrj"
            trajectory.write_bytes(b"complete trajectory\n")

            rows = trajectory_identities_from_ensemble(
                provenance,
                ensemble_directory=ensemble,
                temperature=0.45,
            )
            self.assertEqual(
                rows[0]["trajectory_sha256"],
                hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            )
            self.assertEqual(rows[0]["trajectory_size_bytes"], trajectory.stat().st_size)
            self.assertEqual(rows[0]["trajectory_hash_scope"], "complete_file")

            child_manifest = json.loads((child / "manifest.json").read_text())
            child_manifest["velocity_seed"] = 999
            (child / "manifest.json").write_text(json.dumps(child_manifest))
            with self.assertRaisesRegex(ValueError, "child manifest"):
                trajectory_identities_from_ensemble(
                    provenance,
                    ensemble_directory=ensemble,
                    temperature=0.45,
                )

        with self.assertRaisesRegex(ValueError, "unknown parent"):
            bind_parent_lineage(
                [{"temperature": "0.45", "replicate": "2"}],
                provenance_rows=self.provenance_rows()[:1],
                trajectory_rows=self.trajectory_rows()[:1],
                table_kind="temperature",
            )


if __name__ == "__main__":
    unittest.main()
