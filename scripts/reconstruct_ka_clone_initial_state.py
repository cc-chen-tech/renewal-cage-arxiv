#!/usr/bin/env python3
"""Reconstruct full frame-zero states from recorded isoconfigurational seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import load_lammps_custom_trajectory  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruction_input(
    *,
    parent_restart: Path,
    temperature: float,
    velocity_seed: int,
    langevin_seed: int,
    damping: float,
) -> str:
    return f"""units lj
atom_style atomic
read_restart {parent_restart}

reset_timestep 0
velocity all create {temperature:g} {velocity_seed} mom yes rot no dist gaussian
fix integrator all nve
fix bath all langevin {temperature:g} {temperature:g} {damping:g} {langevin_seed}
timestep 0.001
thermo 1
thermo_style custom step time temp pe ke etotal press

dump initial all custom 1 initial.lammpstrj id type x y z ix iy iz vx vy vz fx fy fz
dump_modify initial sort id first yes
run 0
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clone_directory", type=Path)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--clone-count", type=int, default=8)
    parser.add_argument("--maximum-position-error", type=float, default=2e-5)
    args = parser.parse_args()

    clone_directory = args.clone_directory.resolve()
    lammps = args.lammps.resolve()
    output_directory = args.output_directory.resolve()
    manifest_path = clone_directory / "manifest.json"
    if not manifest_path.is_file() or not lammps.is_file():
        raise ValueError("clone manifest and LAMMPS binary must exist")
    if args.clone_count < 1 or args.maximum_position_error <= 0.0:
        raise ValueError("clone-count and maximum-position-error must be positive")
    if output_directory.exists():
        raise ValueError("output-directory must not already exist")
    manifest = json.loads(manifest_path.read_text())
    clone_rows = list(manifest.get("clones", []))
    if args.clone_count > len(clone_rows):
        raise ValueError("clone-count exceeds completed manifest rows")
    if manifest.get("axis_semantics") != "isoconfigurational_langevin_clones":
        raise ValueError("manifest axis semantics are not isoconfigurational clones")
    parent_restart = Path(manifest["parent_restart_path"]).resolve()
    if not parent_restart.is_file():
        raise ValueError("parent restart in manifest does not exist")
    parent_hash = file_sha256(parent_restart)
    if parent_hash != manifest.get("parent_restart_sha256"):
        raise ValueError("parent restart does not match parent_restart_sha256")
    temperature = float(manifest["temperature"])
    damping = float(manifest["langevin_damping"])

    output_directory.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for row in clone_rows[: args.clone_count]:
        clone_index = int(row["clone_index"])
        source_path = (
            clone_directory / f"clone_{clone_index:03d}" / "trajectory.lammpstrj"
        )
        if not source_path.is_file():
            raise ValueError(f"missing completed clone trajectory {source_path}")
        work_directory = output_directory / f"clone_{clone_index:03d}_work"
        work_directory.mkdir()
        input_path = work_directory / "in.initial"
        dump_path = work_directory / "initial.lammpstrj"
        log_path = work_directory / "log.lammps"
        input_path.write_text(
            reconstruction_input(
                parent_restart=parent_restart,
                temperature=temperature,
                velocity_seed=int(row["velocity_seed"]),
                langevin_seed=int(row["langevin_seed"]),
                damping=damping,
            )
        )
        command = [
            str(lammps),
            "-log",
            str(log_path),
            "-screen",
            "none",
            "-in",
            str(input_path),
        ]
        subprocess.run(
            command,
            cwd=work_directory,
            check=True,
            capture_output=True,
            text=True,
        )
        reconstructed = load_lammps_custom_trajectory(dump_path)
        original = load_lammps_custom_trajectory(source_path)
        if len(reconstructed["timesteps"]) != 1 or int(reconstructed["timesteps"][0]) != 0:
            raise ValueError("run 0 reconstruction must contain exactly frame zero")
        if not np.array_equal(
            reconstructed["particle_types"], original["particle_types"]
        ) or not np.allclose(
            reconstructed["box_lengths"], original["box_lengths"], rtol=0.0, atol=1e-12
        ):
            raise ValueError("reconstructed particle types or box differ from original")
        displacement = np.asarray(
            reconstructed["unwrapped_positions"][0]
            - original["unwrapped_positions"][0],
            dtype=float,
        )
        box_lengths = np.asarray(original["box_lengths"], dtype=float)
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        maximum_position_reconstruction_error = float(np.max(np.abs(displacement)))
        if maximum_position_reconstruction_error > args.maximum_position_error:
            raise ValueError(
                "maximum_position_reconstruction_error exceeds declared tolerance"
            )
        output_path = output_directory / f"clone_{clone_index:03d}_initial.npz"
        temporary_path = output_directory / f"clone_{clone_index:03d}_initial.tmp.npz"
        np.savez_compressed(
            temporary_path,
            positions=np.asarray(reconstructed["unwrapped_positions"][0], dtype=float),
            velocities=np.asarray(reconstructed["velocities"][0], dtype=float),
            forces=np.asarray(reconstructed["forces"][0], dtype=float),
            particle_types=np.asarray(reconstructed["particle_types"], dtype=int),
            box_lengths=box_lengths,
            clone_index=np.asarray(clone_index),
            velocity_seed=np.asarray(int(row["velocity_seed"])),
            langevin_seed=np.asarray(int(row["langevin_seed"])),
            temperature=np.asarray(temperature),
            damping=np.asarray(damping),
            parent_restart_sha256=np.asarray(parent_hash),
            source_trajectory_sha256=np.asarray(file_sha256(source_path)),
            maximum_position_reconstruction_error=np.asarray(
                maximum_position_reconstruction_error
            ),
            thermodynamic_claim_allowed=np.asarray(0.0),
        )
        os.replace(temporary_path, output_path)
        records.append(
            {
                "clone_index": clone_index,
                "velocity_seed": int(row["velocity_seed"]),
                "langevin_seed": int(row["langevin_seed"]),
                "reduced_state_path": str(output_path),
                "source_trajectory_path": str(source_path),
                "source_trajectory_sha256": file_sha256(source_path),
                "maximum_position_reconstruction_error": maximum_position_reconstruction_error,
            }
        )
        for transient in (dump_path, input_path, log_path):
            if transient.exists():
                transient.unlink()
        work_directory.rmdir()

    reduced_manifest = {
        "axis_semantics": "isoconfigurational_clone_initial_full_state",
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "parent_restart_path": str(parent_restart),
        "parent_restart_sha256": parent_hash,
        "temperature": temperature,
        "langevin_damping": damping,
        "clone_count": len(records),
        "maximum_position_reconstruction_error": max(
            float(row["maximum_position_reconstruction_error"]) for row in records
        ),
        "maximum_position_error_allowed": args.maximum_position_error,
        "raw_dump_retained": False,
        "thermodynamic_claim_allowed": False,
        "records": records,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(reduced_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(reduced_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
