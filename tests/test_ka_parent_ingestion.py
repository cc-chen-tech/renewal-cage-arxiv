import json
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.ka_parent_completion import file_sha256
from src.ka_parent_ingestion import (
    completion_import_blockers,
    heldout_target_rows,
    run_frozen_six_ablation_gate,
    split_parent_trajectory,
    stationarity_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_trajectory() -> dict[str, np.ndarray]:
    frames = 10001
    particles = 4
    rng = np.random.default_rng(20260719)
    increments = rng.normal(scale=0.04, size=(frames - 1, particles, 3))
    positions = np.concatenate(
        [np.zeros((1, particles, 3)), np.cumsum(increments, axis=0)], axis=0
    )
    return {
        "timesteps": np.arange(frames, dtype=np.int64) * 1000,
        "particle_types": np.array([0, 0, 0, 1]),
        "unwrapped_positions": positions,
    }


class ParentIngestionTests(unittest.TestCase):
    def test_split_uses_exact_calibration_and_heldout_windows(self):
        trajectory = synthetic_trajectory()
        split = split_parent_trajectory(trajectory)
        self.assertEqual(split["calibration_blocks"].shape, (3, 250, 3))
        self.assertEqual(split["heldout_blocks"].shape, (3, 250, 3))
        np.testing.assert_allclose(
            split["calibration_blocks"][:, -1],
            trajectory["unwrapped_positions"][5000, :3]
            - trajectory["unwrapped_positions"][4980, :3],
        )
        np.testing.assert_allclose(
            split["heldout_blocks"][:, 0],
            trajectory["unwrapped_positions"][5020, :3]
            - trajectory["unwrapped_positions"][5000, :3],
        )
        self.assertEqual(split["calibration_stop_tau"], 5000.0)
        self.assertEqual(split["heldout_start_tau"], 5000.0)

    def test_targets_and_stationarity_are_parent_keyed_and_claim_closed(self):
        split = split_parent_trajectory(synthetic_trajectory())
        targets = heldout_target_rows(
            split["heldout_blocks"],
            parent_id="parent-a",
            replicate=1,
            trajectory_sha256="a" * 64,
            parent_manifest_sha256="b" * 64,
        )
        self.assertEqual([row["lag"] for row in targets], [20, 100, 200, 500, 1000, 2000, 3000])
        self.assertTrue(all(row["parent_id"] == "parent-a" for row in targets))
        self.assertTrue(all(row["heldout_observables_used_as_model_inputs"] == 0 for row in targets))
        rows = stationarity_rows(
            split["calibration_blocks"],
            targets,
            parent_id="parent-a",
            replicate=1,
            trajectory_sha256="a" * 64,
            parent_manifest_sha256="b" * 64,
        )
        self.assertEqual(
            {row["comparison"] for row in rows},
            {"early_late", "early_heldout", "late_heldout"},
        )
        self.assertTrue(all(row["positive_memory_closure_claim_allowed"] == 0 for row in rows))

    def test_committed_remote_completion_blocks_import_without_exit_status(self):
        manifest_path = ROOT / "data" / "renewal_cage_ka_prl_T045_parent_acquisition_manifest.json"
        completion_path = ROOT / "data" / "renewal_cage_ka_prl_T045_parent_acquisition_completion.json"
        ledger, blocker = completion_import_blockers(
            json.loads(manifest_path.read_text()),
            json.loads(completion_path.read_text()),
            acquisition_manifest_sha256=file_sha256(manifest_path),
        )
        self.assertEqual(len(ledger), 2)
        self.assertTrue(all(row["scientific_ingestion_allowed"] == 0 for row in ledger))
        self.assertEqual(blocker["blocker_state"], "blocked_missing_observed_exit_code")
        self.assertEqual(blocker["available_parent_count"], 0)
        self.assertEqual(blocker["missing_parent_count"], 3)
        self.assertEqual(blocker["positive_memory_closure_claim_allowed"], 0)

    def test_one_click_cli_writes_machine_readable_fail_closed_bundle(self):
        manifest_path = ROOT / "data" / "renewal_cage_ka_prl_T045_parent_acquisition_manifest.json"
        completion_path = ROOT / "data" / "renewal_cage_ka_prl_T045_parent_acquisition_completion.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "import_ka_independent_parent_acquisition.py"),
                    "--acquisition-manifest",
                    str(manifest_path),
                    "--completion-artifact",
                    str(completion_path),
                    "--output-directory",
                    str(output),
                ],
                check=True,
                cwd=ROOT,
            )
            status = json.loads((output / "run_status.json").read_text())
            with (output / "gate.csv").open(newline="") as handle:
                gate = next(csv.DictReader(handle))
            self.assertEqual(status["pipeline_state"], "completion_blocked_fail_closed")
            self.assertEqual(status["trajectory_files_opened"], 0)
            self.assertEqual(gate["mechanism_state"], "blocked_independent_parent_validation")
            self.assertEqual(float(gate["positive_memory_closure_claim_allowed"]), 0.0)

    def test_fixture_runs_exact_64_realization_six_ablation_contract(self):
        rng = np.random.default_rng(45064)
        prepared = []
        for replicate in (1, 2, 3):
            calibration = rng.normal(scale=0.2, size=(4, 160, 3))
            heldout = rng.normal(scale=0.2, size=(4, 160, 3))
            prepared.append(
                {
                    "temperature": 0.45,
                    "replicate": replicate,
                    "parent_id": f"fixture-parent-{replicate}",
                    "velocity_seed": 1000 + replicate,
                    "trajectory_sha256": str(replicate) * 64,
                    "trajectory_size_bytes": 1000 + replicate,
                    "parent_manifest_sha256": chr(96 + replicate) * 64,
                    "calibration_blocks": calibration,
                    "heldout_blocks": heldout,
                }
            )
        result = run_frozen_six_ablation_gate(
            prepared,
            realizations=64,
            workers=1,
            fixture_only=True,
        )
        tested_models = {
            "mean_rate_null",
            "one_step_jump_law",
            "two_point_path_spectrum",
            "static_particle_environment",
            "finite_exchange_environment",
            "full_candidate",
        }
        self.assertEqual(
            {row["model"] for row in result["realization_rows"]}.intersection(tested_models),
            tested_models,
        )
        for model in tested_models - {"two_point_path_spectrum"}:
            labels = {
                int(row["realization"])
                for row in result["realization_rows"]
                if row["model"] == model and int(row["restart"]) == 1
            }
            self.assertEqual(labels, set(range(64)))
        spectral_labels = {
            int(row["realization"])
            for row in result["realization_rows"]
            if row["model"] == "two_point_path_spectrum" and int(row["restart"]) == 1
        }
        self.assertEqual(spectral_labels, set(range(8)))
        self.assertEqual(len(result["parent_ledger"]), 3)
        self.assertEqual(len(result["stationarity_rows"]), 9)
        self.assertEqual(result["gate"]["fixture_only"], 1)
        self.assertEqual(result["gate"]["positive_memory_closure_claim_allowed"], 0)


if __name__ == "__main__":
    unittest.main()
