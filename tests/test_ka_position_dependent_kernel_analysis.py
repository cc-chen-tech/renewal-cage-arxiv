import subprocess
import sys
import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class PositionDependentKernelAnalysisTests(unittest.TestCase):
    @staticmethod
    def _synthetic_kernel_clones() -> list[dict[str, object]]:
        from ka_position_dependent_kernel import (
            radial_vector_basis,
            radial_vector_basis_jacobian,
        )

        clones = []
        reference_scale = {
            "mu_r2": 3.0,
            "sigma_r2": 2.0,
            "epsilon_r2": 0.05,
        }
        for clone_index in range(1, 5):
            rng = np.random.default_rng(900 + clone_index)
            position = rng.normal(size=(32, 7, 3))
            velocity = rng.normal(size=position.shape)
            basis = radial_vector_basis(position, reference_scale)
            jacobian = radial_vector_basis_jacobian(position, reference_scale)
            jacobian_velocity = np.einsum(
                "tpbik,tpk->tpbi",
                jacobian,
                velocity,
            )
            acceleration = np.einsum(
                "b,tpbi->tpi",
                np.array([0.25, -0.14, 0.08]),
                basis,
            )
            acceleration[1:] -= np.einsum(
                "b,tpbi->tpi",
                np.array([0.18, 0.11, -0.07]),
                jacobian_velocity[:-1],
            )
            clones.append(
                {
                    "clone_index": float(clone_index),
                    "relative_position": position,
                    "relative_velocity": velocity,
                    "relative_drift": acceleration,
                }
            )
        return clones

    def test_nested_nonparametric_selection_is_whole_clone_and_outer_held_blind(self):
        from analyze_ka_position_dependent_kernel import (
            select_nonparametric_hierarchy,
        )

        clones = self._synthetic_kernel_clones()
        candidates, selections = select_nonparametric_hierarchy(
            clones,
            supports=(2, 3),
            ridge_grid=(0.0, 1e-6),
            permutation_seed=20260719,
        )
        self.assertEqual(len(candidates), 4 * 3 * 2 * 2)
        self.assertEqual(len(selections), 4 * 3)
        self.assertTrue(all(row["fit_uses_outer_held_clone"] == 0.0 for row in candidates))
        self.assertTrue(all(row["inner_validation_fold_count"] == 3.0 for row in candidates))
        self.assertTrue(all(row["fit_uses_outer_held_clone"] == 0.0 for row in selections))
        for held_clone in range(1, 5):
            selected = {
                row["model"]: row
                for row in selections
                if row["held_clone_index"] == float(held_clone)
            }
            self.assertLess(
                selected["finite_basis_mz_position_kernel"]["mean_inner_normalized_rmse"],
                selected["stationary_scalar_nonparametric_volterra"]["mean_inner_normalized_rmse"],
            )
            self.assertLess(
                selected["finite_basis_mz_position_kernel"]["mean_inner_normalized_rmse"],
                selected["time_permuted_position_null"]["mean_inner_normalized_rmse"],
            )

        mutated = [dict(clone) for clone in clones]
        mutated[3] = {
            **mutated[3],
            "relative_position": 50.0 * np.asarray(mutated[3]["relative_position"]),
            "relative_drift": -30.0 * np.asarray(mutated[3]["relative_drift"]),
        }
        _, changed = select_nonparametric_hierarchy(
            mutated,
            supports=(2, 3),
            ridge_grid=(0.0, 1e-6),
            permutation_seed=20260719,
        )
        original_held_four = [
            row for row in selections if row["held_clone_index"] == 4.0
        ]
        changed_held_four = [row for row in changed if row["held_clone_index"] == 4.0]
        self.assertEqual(original_held_four, changed_held_four)

    def test_nested_auxiliary_selection_derives_poles_inside_each_training_fold(self):
        from analyze_ka_position_dependent_kernel import (
            select_auxiliary_hierarchy,
            select_nonparametric_hierarchy,
        )

        clones = self._synthetic_kernel_clones()
        _, nonparametric = select_nonparametric_hierarchy(
            clones,
            supports=(2, 3),
            ridge_grid=(1e-6,),
            permutation_seed=20260719,
        )
        candidates, selections = select_auxiliary_hierarchy(
            clones,
            nonparametric,
            ranks=(1, 2),
            ridge_grid=(1e-6, 1e-4),
            decay_grid=np.array([0.1, 0.3, 1.0, 3.0]),
            frame_time=0.01,
        )
        self.assertEqual(len(candidates), 4 * 2 * 2 * 2)
        self.assertEqual(len(selections), 4 * 2)
        self.assertTrue(all(row["fit_uses_outer_held_clone"] == 0.0 for row in candidates))
        self.assertTrue(all(row["pole_selection_uses_outer_held_clone"] == 0.0 for row in candidates))
        self.assertTrue(all(row["inner_validation_fold_count"] == 3.0 for row in candidates))
        self.assertTrue(all(row["all_selected_decay_rates_positive"] == 1.0 for row in selections))

        mutated = [dict(clone) for clone in clones]
        mutated[3] = {
            **mutated[3],
            "relative_position": 40.0 * np.asarray(mutated[3]["relative_position"]),
            "relative_drift": -20.0 * np.asarray(mutated[3]["relative_drift"]),
        }
        _, changed = select_auxiliary_hierarchy(
            mutated,
            nonparametric,
            ranks=(1, 2),
            ridge_grid=(1e-6, 1e-4),
            decay_grid=np.array([0.1, 0.3, 1.0, 3.0]),
            frame_time=0.01,
        )
        self.assertEqual(
            [row for row in selections if row["held_clone_index"] == 4.0],
            [row for row in changed if row["held_clone_index"] == 4.0],
        )

    def test_outer_refit_scores_one_common_held_horizon_and_fail_closed_claims(self):
        from analyze_ka_position_dependent_kernel import (
            fit_and_score_outer_hierarchy,
            select_auxiliary_hierarchy,
            select_nonparametric_hierarchy,
        )
        from ka_position_dependent_kernel import (
            classify_position_dependent_kernel_gate,
        )

        clones = self._synthetic_kernel_clones()
        _, nonparametric = select_nonparametric_hierarchy(
            clones,
            supports=(2, 3),
            ridge_grid=(1e-6,),
            permutation_seed=20260719,
        )
        decay_grid = np.array([0.1, 0.3, 1.0, 3.0])
        _, auxiliary = select_auxiliary_hierarchy(
            clones,
            nonparametric,
            ranks=(1, 2),
            ridge_grid=(1e-6,),
            decay_grid=decay_grid,
            frame_time=0.01,
        )
        rows, kernels = fit_and_score_outer_hierarchy(
            clones,
            nonparametric + auxiliary,
            decay_grid=decay_grid,
            frame_time=0.01,
            temperature=0.58,
            maximum_diagnostic_lag=2,
            permutation_seed=20260719,
        )
        self.assertEqual(len(rows), 4 * 5)
        self.assertGreater(len(kernels), 0)
        for held_clone in range(1, 5):
            fold = [row for row in rows if row["held_clone_index"] == float(held_clone)]
            self.assertEqual(len({row["held_scalar_component_count"] for row in fold}), 1)
            self.assertTrue(all(row["fit_uses_outer_held_clone"] == 0.0 for row in fold))
            m3 = next(row for row in fold if row["model"] == "past_position_real_pole")
            m4 = next(row for row in fold if row["model"] == "two_position_positive_prony")
            self.assertEqual(m3["second_fdt_diagnostics_applicable"], 0.0)
            self.assertEqual(m4["second_fdt_diagnostics_applicable"], 1.0)
            self.assertEqual(m4["positive_prony_factorization"], 1.0)
        verdict = classify_position_dependent_kernel_gate(rows)
        for claim in (
            "latent_auxiliary_innovation_identified_in_ka",
            "stochastic_auxiliary_bath_identified_in_ka",
            "autonomous_single_particle_gle_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "spatial_facilitation_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(verdict[claim], 0.0)

        from analyze_ka_position_dependent_kernel import (
            file_sha256,
            write_kernel_artifacts,
        )

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "kernel_gate"
            paths = write_kernel_artifacts(
                prefix,
                details=rows,
                selections=nonparametric + auxiliary,
                kernels=kernels,
                verdict=verdict,
                source_paths=[
                    ROOT / "scripts" / "analyze_ka_position_dependent_kernel.py",
                    ROOT / "src" / "ka_position_dependent_kernel.py",
                ],
            )
            self.assertEqual(set(paths), {"details", "selection", "kernel", "summary", "svg", "sha256"})
            self.assertTrue(all(path.is_file() for path in paths.values()))
            svg = paths["svg"].read_text()
            self.assertIn("unclipped held diagnostics", svg)
            self.assertIn("residual-correlation tolerance 0.20", svg)
            self.assertIn("second-FDT tolerance 0.30", svg)
            with paths["sha256"].open(newline="") as handle:
                manifest = list(csv.DictReader(handle))
            artifact_rows = [row for row in manifest if row["record"] == "artifact"]
            self.assertEqual(len(artifact_rows), 5)
            for row in artifact_rows:
                self.assertEqual(file_sha256(Path(row["path"])), row["sha256"])

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
