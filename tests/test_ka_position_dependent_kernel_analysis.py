import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class PositionDependentKernelAnalysisTests(unittest.TestCase):
    def test_cli_and_loader_require_frozen_four_clone_provenance(self):
        from analyze_ka_position_dependent_kernel import (
            file_sha256,
            load_frozen_kernel_clones,
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_position_dependent_kernel.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for phrase in (
            "--drift-cache-directory",
            "--trajectory-directory",
            "--output-prefix",
            "--memory-supports",
            "4 16 40 100",
            "--auxiliary-ranks",
            "1 2 4 8",
            "--ridge-grid",
            "0.0 1e-10 1e-08 1e-06 0.0001 0.01",
            "--decay-grid-count",
            "32",
        ):
            self.assertIn(phrase, completed.stdout)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drift_directory = root / "drift"
            trajectory_directory = root / "trajectories"
            drift_directory.mkdir()
            targets = np.array([2, 5], dtype=int)
            for clone in range(1, 5):
                clone_directory = trajectory_directory / f"clone_{clone:03d}"
                clone_directory.mkdir(parents=True)
                trajectory_path = clone_directory / "trajectory.lammpstrj"
                trajectory_path.write_bytes(f"trajectory-{clone}".encode())
                shape = (8, 2, 3)
                rng = np.random.default_rng(clone)
                np.savez_compressed(
                    drift_directory / f"clone_{clone:03d}_decomposed_drift.npz",
                    relative_position=rng.normal(size=shape),
                    relative_velocity=rng.normal(size=shape),
                    center_velocity=rng.normal(size=shape),
                    relative_drift=rng.normal(size=shape),
                    center_drift=rng.normal(size=shape),
                    target_indices=targets,
                    trajectory_sha256=np.asarray(file_sha256(trajectory_path)),
                    thermodynamic_claim_allowed=np.asarray(0.0),
                )
            clones = load_frozen_kernel_clones(
                drift_directory,
                trajectory_directory,
                expected_clone_count=4,
                target_count=2,
            )
            self.assertEqual(len(clones), 4)
            self.assertEqual(clones[0]["relative_position"].shape, (8, 2, 3))
            self.assertEqual(clones[0]["trajectory_sha256"], file_sha256(
                trajectory_directory / "clone_001" / "trajectory.lammpstrj"
            ))
            self.assertEqual(clones[0]["fit_uses_held_clone"], 0.0)

            changed = trajectory_directory / "clone_003" / "trajectory.lammpstrj"
            changed.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "trajectory hash"):
                load_frozen_kernel_clones(
                    drift_directory,
                    trajectory_directory,
                    expected_clone_count=4,
                    target_count=2,
                )


if __name__ == "__main__":
    unittest.main()
