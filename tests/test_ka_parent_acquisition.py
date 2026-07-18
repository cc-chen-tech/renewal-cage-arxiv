import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.ka_parent_acquisition import (
    prepare_parent_acquisition,
    validate_acquisition_spec,
    validate_prelaunch_spec,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "manifest_state": "build_pending",
        "implementation_commit": "a" * 40,
        "temperature": 0.45,
        "system": {
            "particle_count": 1000,
            "type_a_count": 800,
            "type_b_count": 200,
            "density": 1.2,
            "lattice_shape": [10, 10, 10],
        },
        "protocol": {
            "timestep_tau": 0.001,
            "melt_temperature": 1.0,
            "melt_time_tau": 1000.0,
            "cool_time_tau": 4000.0,
            "target_hold_time_tau": 5000.0,
            "production_time_tau": 10000.0,
            "calibration_time_tau": 5000.0,
            "heldout_time_tau": 5000.0,
            "dump_interval_tau": 1.0,
            "restart_interval_tau": 100.0,
            "thermostat_damping_tau": 10.0,
        },
        "potential": {
            "pair_style": "lj/cut 2.5",
            "pair_modify": "shift yes",
            "pair_coefficients": [
                "1 1 1.0 1.0 2.5",
                "1 2 1.5 0.8 2.0",
                "2 2 0.5 0.88 2.2",
            ],
        },
        "lammps": {
            "version": "22 Jul 2025 - Update 4",
            "binary_sha256": "build_pending",
        },
        "parents": [
            {
                "parent_id": "ka-t045-independent-p02-20260719",
                "type_assignment_seed": 4502001,
                "velocity_seed": 4502002,
                "remote_output_directory": "/root/prl-memory-closure-acquisition/runs/p02",
            },
            {
                "parent_id": "ka-t045-independent-p03-20260719",
                "type_assignment_seed": 4503001,
                "velocity_seed": 4503002,
                "remote_output_directory": "/root/prl-memory-closure-acquisition/runs/p03",
            },
        ],
        "claim_flags": {
            "positive_memory_closure_claim_allowed": 0,
            "complete_microscopic_closure_claim_allowed": 0,
            "spatial_facilitation_claim_allowed": 0,
            "thermodynamic_glass_transition_claim_allowed": 0,
        },
    }


class ParentAcquisitionTests(unittest.TestCase):
    def test_prepare_writes_two_distinct_hashed_independent_parent_inputs(self):
        spec = frozen_spec()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec))
            output = root / "prepared"

            rows = prepare_parent_acquisition(
                spec_path,
                output,
                repository_commit="a" * 40,
            )

            self.assertEqual(len(rows), 2)
            type_vectors = []
            for row in rows:
                parent = output / str(row["parent_id"])
                data = parent / "initial.data"
                input_path = parent / "in.production"
                manifest = json.loads((parent / "parent_manifest.json").read_text())
                self.assertEqual(manifest["initial_data_sha256"], sha256(data))
                self.assertEqual(manifest["lammps_input_sha256"], sha256(input_path))
                self.assertEqual(manifest["implementation_commit"], "a" * 40)
                text = input_path.read_text()
                for command in (
                    "run 1000000",
                    "run 4000000",
                    "run 5000000",
                    "run 10000000",
                    "reset_timestep 0",
                    "dump trajectory all custom 1000 trajectory.lammpstrj",
                ):
                    self.assertIn(command, text)
                atom_lines = data.read_text().split("Atoms # atomic\n\n", 1)[1].splitlines()
                types = np.array([int(line.split()[1]) for line in atom_lines if line.strip()])
                self.assertEqual(np.count_nonzero(types == 1), 800)
                self.assertEqual(np.count_nonzero(types == 2), 200)
                type_vectors.append(types)
            self.assertFalse(np.array_equal(type_vectors[0], type_vectors[1]))

    def test_spec_rejects_shared_seeds_or_changed_frozen_budget(self):
        spec = frozen_spec()
        spec["parents"][1]["velocity_seed"] = spec["parents"][0]["velocity_seed"]
        with self.assertRaisesRegex(ValueError, "seeds"):
            validate_acquisition_spec(spec)

        spec = frozen_spec()
        spec["protocol"]["production_time_tau"] = 9999.0
        with self.assertRaisesRegex(ValueError, "production"):
            validate_acquisition_spec(spec)

    def test_prelaunch_requires_binary_and_generated_input_hashes(self):
        spec = validate_acquisition_spec(frozen_spec())
        spec["manifest_state"] = "frozen_prelaunch"
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_prelaunch_spec(spec)

        spec["lammps"]["binary_sha256"] = "b" * 64
        for parent in spec["parents"]:
            parent["initial_data_sha256"] = "c" * 64
            parent["lammps_input_sha256"] = "d" * 64
        validated = validate_prelaunch_spec(spec)
        self.assertEqual(validated["manifest_state"], "frozen_prelaunch")


if __name__ == "__main__":
    unittest.main()
