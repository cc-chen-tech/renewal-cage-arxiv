import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ka_prl_memory_closure as closure


class ParentProvenanceTests(unittest.TestCase):
    def provenance_rows(self):
        rows = []
        for temperature, digest, frames in (
            (0.45, "a" * 64, (5000, 35000, 65000)),
            (0.58, "b" * 64, (3000, 8000, 13000, 18000, 23000)),
        ):
            for replicate, frame in enumerate(frames, start=1):
                rows.append(
                    {
                        "temperature": str(temperature),
                        "replicate": str(replicate),
                        "source_doi": "10.5281/zenodo.7469766",
                        "source_sha256": digest,
                        "source_frame_index": str(frame),
                        "velocity_seed": str(45000 + replicate),
                        "production_time_tau": "10000" if temperature == 0.45 else "1500",
                        "independence_class": "decorrelated_parent_frames_plus_velocity_seeds",
                        "independently_prepared_parent_samples": "False",
                    }
                )
        return rows

    @staticmethod
    def stationarity(passing):
        return [
            {"comparison": comparison, "curve_transfer_pass": str(float(passing))}
            for comparison in ("early_late", "early_heldout", "late_heldout")
        ]

    def test_parent_audit_counts_shared_restart_parent_once(self):
        ledger, blockers = closure.audit_parent_provenance(
            provenance_rows=self.provenance_rows(),
            stationarity_by_temperature={
                0.45: self.stationarity(True),
                0.58: self.stationarity(False),
            },
        )

        low_ledger = [row for row in ledger if row["temperature"] == 0.45]
        low_blocker = next(row for row in blockers if row["temperature"] == 0.45)
        self.assertEqual({row["parent_id"] for row in low_ledger}, {f"10.5281/zenodo.7469766:{'a' * 64}"})
        self.assertEqual(low_blocker["available_parent_count"], 1)
        self.assertEqual(low_blocker["missing_parent_count"], 2)
        self.assertEqual(low_blocker["blocker_state"], "missing_independent_parents")
        self.assertEqual(sum(row["parent_unit_contribution"] for row in low_ledger), 1)
        self.assertTrue(all(row["restart_is_independent_sample"] == 0.0 for row in low_ledger))

    def test_parent_audit_keeps_failed_warm_stationarity_as_canary(self):
        _, blockers = closure.audit_parent_provenance(
            provenance_rows=self.provenance_rows(),
            stationarity_by_temperature={
                0.45: self.stationarity(True),
                0.58: self.stationarity(False),
            },
        )

        warm = next(row for row in blockers if row["temperature"] == 0.58)
        self.assertEqual(warm["blocker_state"], "stationarity_and_independent_parents")
        self.assertEqual(warm["evidence_role"], "canary_only")
        self.assertEqual(warm["available_parent_count"], 1)
        self.assertEqual(warm["missing_parent_count"], 4)
        self.assertEqual(warm["stationarity_pass"], 0.0)

    def test_parent_audit_rejects_malformed_parent_hash_and_missing_comparison(self):
        rows = self.provenance_rows()
        rows[0]["source_sha256"] = "not-a-sha"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            closure.audit_parent_provenance(
                provenance_rows=rows,
                stationarity_by_temperature={
                    0.45: self.stationarity(True),
                    0.58: self.stationarity(False),
                },
            )

        with self.assertRaisesRegex(ValueError, "stationarity comparisons"):
            closure.audit_parent_provenance(
                provenance_rows=self.provenance_rows(),
                stationarity_by_temperature={
                    0.45: self.stationarity(True)[:2],
                    0.58: self.stationarity(False),
                },
            )


if __name__ == "__main__":
    unittest.main()
