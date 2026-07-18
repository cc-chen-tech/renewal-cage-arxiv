"""Frozen preparation protocol for independently prepared KA parent trajectories."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FROZEN_PARENT_IDS = {
    "ka-t045-independent-p02-20260719",
    "ka-t045-independent-p03-20260719",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hex(value: object, length: int) -> bool:
    text = str(value)
    return len(text) == length and all(character in "0123456789abcdef" for character in text)


def validate_acquisition_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate the immutable acquisition contract without requiring a built binary."""

    result = copy.deepcopy(spec)
    if result.get("schema_version") != 1:
        raise ValueError("schema_version must equal one")
    if result.get("manifest_state") not in {"build_pending", "frozen_prelaunch"}:
        raise ValueError("manifest_state is invalid")
    if not _is_hex(result.get("implementation_commit"), 40):
        raise ValueError("implementation_commit must be a lowercase 40-character Git hash")
    if not math.isclose(float(result.get("temperature", math.nan)), 0.45):
        raise ValueError("temperature must remain frozen at 0.45")

    system = result.get("system", {})
    expected_system = {
        "particle_count": 1000,
        "type_a_count": 800,
        "type_b_count": 200,
        "density": 1.2,
        "lattice_shape": [10, 10, 10],
    }
    if system != expected_system:
        raise ValueError("system composition, density, and lattice shape are frozen")

    protocol = result.get("protocol", {})
    frozen_protocol = {
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
    }
    if set(protocol) != set(frozen_protocol):
        raise ValueError("protocol fields do not match the frozen acquisition schema")
    for key, expected in frozen_protocol.items():
        if not math.isclose(float(protocol[key]), expected):
            label = "production" if key == "production_time_tau" else key
            raise ValueError(f"{label} is not the frozen value")
    if not math.isclose(
        float(protocol["calibration_time_tau"]) + float(protocol["heldout_time_tau"]),
        float(protocol["production_time_tau"]),
    ):
        raise ValueError("calibration and heldout windows must span production")

    potential = result.get("potential", {})
    if potential != {
        "pair_style": "lj/cut 2.5",
        "pair_modify": "shift yes",
        "pair_coefficients": [
            "1 1 1.0 1.0 2.5",
            "1 2 1.5 0.8 2.0",
            "2 2 0.5 0.88 2.2",
        ],
    }:
        raise ValueError("Kob-Andersen potential parameters are frozen")

    lammps = result.get("lammps", {})
    binary_hash = lammps.get("binary_sha256")
    if binary_hash != "build_pending" and not _is_hex(binary_hash, 64):
        raise ValueError("LAMMPS binary hash must be pending or a SHA256 digest")

    parents = result.get("parents")
    if not isinstance(parents, list) or len(parents) != 2:
        raise ValueError("exactly two T=0.45 parents are required")
    if {parent.get("parent_id") for parent in parents} != FROZEN_PARENT_IDS:
        raise ValueError("parent IDs differ from the frozen acquisition")
    seeds = [
        int(parent[key])
        for parent in parents
        for key in ("type_assignment_seed", "velocity_seed")
    ]
    if min(seeds) <= 0 or len(set(seeds)) != len(seeds):
        raise ValueError("all preparation and velocity seeds must be positive and distinct")
    directories = [str(parent.get("remote_output_directory", "")) for parent in parents]
    if len(set(directories)) != 2 or any(not value.startswith("/") for value in directories):
        raise ValueError("remote output directories must be distinct absolute paths")
    if any(float(value) != 0.0 for value in result.get("claim_flags", {}).values()):
        raise ValueError("all acquisition claim flags must remain zero")
    return result


def validate_prelaunch_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Require the hashes that must be frozen before LAMMPS is launched."""

    result = validate_acquisition_spec(spec)
    if result["manifest_state"] != "frozen_prelaunch":
        raise ValueError("prelaunch manifest_state must be frozen_prelaunch")
    if not _is_hex(result["lammps"].get("binary_sha256"), 64):
        raise ValueError("binary SHA256 must be frozen before launch")
    for parent in result["parents"]:
        if not _is_hex(parent.get("initial_data_sha256"), 64):
            raise ValueError("initial-data SHA256 must be frozen before launch")
        if not _is_hex(parent.get("lammps_input_sha256"), 64):
            raise ValueError("LAMMPS-input SHA256 must be frozen before launch")
    return result


def _initial_state(spec: dict[str, Any], parent: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    system = spec["system"]
    particle_count = int(system["particle_count"])
    box_length = float((particle_count / float(system["density"])) ** (1.0 / 3.0))
    shape = tuple(int(value) for value in system["lattice_shape"])
    spacing = box_length / shape[0]
    grid = np.indices(shape).reshape(3, -1).T
    positions = -0.5 * box_length + (grid + 0.5) * spacing
    types = np.ones(particle_count, dtype=int)
    generator = np.random.default_rng(int(parent["type_assignment_seed"]))
    type_b = generator.choice(particle_count, size=int(system["type_b_count"]), replace=False)
    types[type_b] = 2
    return positions, types, box_length


def _data_text(positions: np.ndarray, types: np.ndarray, box_length: float) -> str:
    lines = [
        "Independent Kob-Andersen parent initialization",
        "",
        f"{len(types)} atoms",
        "2 atom types",
        "",
        f"{-box_length / 2.0:.10f} {box_length / 2.0:.10f} xlo xhi",
        f"{-box_length / 2.0:.10f} {box_length / 2.0:.10f} ylo yhi",
        f"{-box_length / 2.0:.10f} {box_length / 2.0:.10f} zlo zhi",
        "",
        "Masses",
        "",
        "1 1.0",
        "2 1.0",
        "",
        "Atoms # atomic",
        "",
    ]
    for atom_id, (particle_type, position) in enumerate(zip(types, positions), start=1):
        lines.append(
            f"{atom_id} {particle_type} {position[0]:.9f} {position[1]:.9f} {position[2]:.9f}"
        )
    return "\n".join(lines) + "\n"


def _steps(time_tau: float, timestep: float) -> int:
    value = int(round(time_tau / timestep))
    if not math.isclose(value * timestep, time_tau):
        raise ValueError("protocol time must be an integer number of timesteps")
    return value


def _input_text(spec: dict[str, Any], parent: dict[str, Any]) -> str:
    protocol = spec["protocol"]
    timestep = float(protocol["timestep_tau"])
    melt_steps = _steps(float(protocol["melt_time_tau"]), timestep)
    cool_steps = _steps(float(protocol["cool_time_tau"]), timestep)
    hold_steps = _steps(float(protocol["target_hold_time_tau"]), timestep)
    production_steps = _steps(float(protocol["production_time_tau"]), timestep)
    dump_steps = _steps(float(protocol["dump_interval_tau"]), timestep)
    restart_steps = _steps(float(protocol["restart_interval_tau"]), timestep)
    target = float(spec["temperature"])
    melt = float(protocol["melt_temperature"])
    damping = float(protocol["thermostat_damping_tau"])
    coefficients = "\n".join(
        f"pair_coeff {value}" for value in spec["potential"]["pair_coefficients"]
    )
    return f"""units lj
atom_style atomic
boundary p p p
read_data initial.data

pair_style {spec['potential']['pair_style']}
pair_modify {spec['potential']['pair_modify']}
{coefficients}
neighbor 0.3 bin
neigh_modify delay 0 every 1 check yes

velocity all create {melt:g} {int(parent['velocity_seed'])} mom yes rot no dist gaussian
timestep {timestep:g}
thermo {restart_steps}
thermo_style custom step time temp pe ke etotal press

fix thermostat all nvt temp {melt:g} {melt:g} {damping:g}
run {melt_steps}
unfix thermostat
fix thermostat all nvt temp {melt:g} {target:g} {damping:g}
run {cool_steps}
unfix thermostat
fix thermostat all nvt temp {target:g} {target:g} {damping:g}
run {hold_steps}
write_restart preparation_end.restart

reset_timestep 0
dump trajectory all custom {dump_steps} trajectory.lammpstrj id type x y z ix iy iz
dump_modify trajectory sort id
restart {restart_steps} restart.*
run {production_steps}
write_restart final.restart
"""


def prepare_parent_acquisition(
    spec_path: Path,
    output_directory: Path,
    *,
    repository_commit: str,
) -> list[dict[str, Any]]:
    """Render deterministic LAMMPS inputs and their per-parent provenance."""

    spec_path = Path(spec_path).resolve()
    output_directory = Path(output_directory).resolve()
    spec = validate_acquisition_spec(json.loads(spec_path.read_text()))
    if repository_commit != spec["implementation_commit"]:
        raise ValueError("repository commit does not match the frozen implementation commit")
    if output_directory.exists():
        raise ValueError("output_directory must not exist")
    output_directory.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for parent in spec["parents"]:
        parent_directory = output_directory / str(parent["parent_id"])
        parent_directory.mkdir()
        positions, types, box_length = _initial_state(spec, parent)
        data_path = parent_directory / "initial.data"
        input_path = parent_directory / "in.production"
        data_path.write_text(_data_text(positions, types, box_length))
        input_path.write_text(_input_text(spec, parent))
        manifest = {
            **parent,
            "temperature": spec["temperature"],
            "implementation_commit": spec["implementation_commit"],
            "spec_sha256": _sha256(spec_path),
            "initial_data_sha256": _sha256(data_path),
            "lammps_input_sha256": _sha256(input_path),
            "particle_count": spec["system"]["particle_count"],
            "type_a_count": spec["system"]["type_a_count"],
            "type_b_count": spec["system"]["type_b_count"],
            "density": spec["system"]["density"],
            "box_length": box_length,
            "production_time_tau": spec["protocol"]["production_time_tau"],
            "calibration_time_tau": spec["protocol"]["calibration_time_tau"],
            "heldout_time_tau": spec["protocol"]["heldout_time_tau"],
            "saved_frame_interval_tau": spec["protocol"]["dump_interval_tau"],
            "expected_production_frame_count": 10001,
            "independence_class": "independent_type_assignment_melt_cool_hold_history",
            "independently_prepared_parent_sample": True,
            **spec["claim_flags"],
        }
        manifest_path = parent_directory / "parent_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        rows.append(manifest)
    return rows
