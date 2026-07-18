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


class MemoryKernelTests(unittest.TestCase):
    @staticmethod
    def labelled_blocks(particles=4, block_count=12):
        blocks = np.zeros((particles, block_count, 3), dtype=float)
        for particle in range(particles):
            for block in range(block_count):
                blocks[particle, block] = (
                    1000.0 * particle + block,
                    10.0 * particle + block,
                    particle - block,
                )
        return blocks

    def test_full_candidate_uses_contiguous_blocks_until_exchange(self):
        _, audit = closure.generate_ablation_path(
            self.labelled_blocks(),
            model="full_candidate",
            environment_time=1.0e12,
            block_size=20.0,
            rng=np.random.default_rng(7),
        )

        sources = audit["source_particle"]
        source_blocks = audit["source_block"]
        for particle in range(sources.shape[0]):
            for index in range(1, sources.shape[1]):
                if sources[particle, index] == sources[particle, index - 1]:
                    self.assertEqual(
                        source_blocks[particle, index],
                        source_blocks[particle, index - 1] + 1,
                    )
        self.assertEqual(audit["ordered_path_memory_retained"], 1.0)
        self.assertEqual(audit["finite_exchange_environment_retained"], 1.0)
        self.assertEqual(audit["source_wrap_count"], 0.0)

    def test_finite_exchange_ablation_retains_identity_but_destroys_order(self):
        _, audit = closure.generate_ablation_path(
            self.labelled_blocks(block_count=40),
            model="finite_exchange_environment",
            environment_time=20.0,
            block_size=20.0,
            rng=np.random.default_rng(11),
        )

        self.assertEqual(audit["finite_exchange_environment_retained"], 1.0)
        self.assertEqual(audit["ordered_path_memory_retained"], 0.0)
        self.assertGreater(audit["environment_exchange_count"], 0.0)

    def test_static_environment_never_changes_source_particle(self):
        _, audit = closure.generate_ablation_path(
            self.labelled_blocks(),
            model="static_particle_environment",
            environment_time=40.0,
            block_size=20.0,
            rng=np.random.default_rng(13),
        )

        expected = np.broadcast_to(
            np.arange(4, dtype=int)[:, None], audit["source_particle"].shape
        )
        np.testing.assert_array_equal(audit["source_particle"], expected)
        self.assertEqual(audit["environment_exchange_count"], 0.0)
        self.assertEqual(audit["static_particle_environment_retained"], 1.0)
        self.assertEqual(audit["ordered_path_memory_retained"], 0.0)

    def test_terminal_source_block_forces_recorded_exchange_without_wrap(self):
        _, audit = closure.generate_ablation_path(
            self.labelled_blocks(particles=3, block_count=8),
            model="full_candidate",
            environment_time=1.0e12,
            block_size=20.0,
            rng=np.random.default_rng(5),
        )

        self.assertEqual(audit["source_wrap_count"], 0.0)
        self.assertGreater(audit["forced_terminal_exchange_count"], 0.0)

    def test_full_candidate_has_no_shared_global_source_schedule(self):
        _, audit = closure.generate_ablation_path(
            self.labelled_blocks(particles=8, block_count=20),
            model="full_candidate",
            environment_time=60.0,
            block_size=20.0,
            rng=np.random.default_rng(19),
        )

        self.assertEqual(audit["global_source_segment_schedule_preserved"], 0.0)
        self.assertTrue(
            any(
                not np.array_equal(
                    audit["source_particle"][0],
                    audit["source_particle"][particle],
                )
                for particle in range(1, 8)
            )
        )

    def test_mean_rate_and_one_step_ablation_information_flags_are_exact(self):
        blocks = self.labelled_blocks()
        for model, one_step in (("mean_rate_null", 0.0), ("one_step_jump_law", 1.0)):
            with self.subTest(model=model):
                generated, audit = closure.generate_ablation_path(
                    blocks,
                    model=model,
                    environment_time=40.0,
                    block_size=20.0,
                    rng=np.random.default_rng(23),
                )
                self.assertEqual(generated.shape, blocks.shape)
                self.assertTrue(np.all(np.isfinite(generated)))
                self.assertEqual(audit["one_step_jump_law_retained"], one_step)
                self.assertEqual(audit["particle_identity_retained"], 0.0)
                self.assertEqual(audit["ordered_path_memory_retained"], 0.0)

    def test_invalid_kernel_controls_fail_closed(self):
        blocks = self.labelled_blocks()
        with self.assertRaisesRegex(ValueError, "unknown"):
            closure.generate_ablation_path(
                blocks,
                model="posthoc_model",
                environment_time=40.0,
                block_size=20.0,
                rng=np.random.default_rng(1),
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            closure.generate_ablation_path(
                blocks,
                model="full_candidate",
                environment_time=0.0,
                block_size=20.0,
                rng=np.random.default_rng(1),
            )


if __name__ == "__main__":
    unittest.main()
