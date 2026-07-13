#!/usr/bin/env python3
"""Compare analytic Python KA C3 forces with LAMMPS Lepton dimers."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_generator_response import ka_c3_lepton_pair_commands  # noqa: E402
from ka_local_cage import ka_lj_force_and_isotropic_curvature  # noqa: E402


PAIR_CASES = (
    ("AA", (1, 1), (0, 0), 1.0),
    ("AB", (1, 2), (0, 1), 0.8),
    ("BB", (2, 2), (1, 1), 0.88),
)
SCALED_RADII = (1.2, 2.0, 2.25, 2.49, 2.4999)


def dimer_input(*, atom_types: tuple[int, int], distance: float) -> str:
    """Return a zero-step LAMMPS input for one isolated periodic dimer."""

    if distance <= 0.0 or not math.isfinite(distance):
        raise ValueError("distance must be finite and positive")
    return f"""units lj
atom_style atomic
boundary p p p
region box block 0 20 0 20 0 20
create_box 2 box
create_atoms {atom_types[0]} single 10 10 10 units box
create_atoms {atom_types[1]} single {10.0 + distance:.17g} 10 10 units box
mass * 1
{ka_c3_lepton_pair_commands()}
neighbor 0.3 bin
neigh_modify delay 0 every 1 check yes
run 0 post no
write_dump all custom force.dump id type x y z fx fy fz modify sort id format float %.17g
"""


def load_first_force(path: Path) -> np.ndarray:
    """Read the force on atom id 1 from the compact parity dump."""

    lines = path.read_text().splitlines()
    header = "ITEM: ATOMS id type x y z fx fy fz"
    try:
        start = lines.index(header) + 1
    except ValueError as error:
        raise ValueError("unexpected parity dump schema") from error
    rows = np.asarray([[float(value) for value in line.split()] for line in lines[start : start + 2]])
    rows = rows[np.argsort(rows[:, 0])]
    if rows.shape != (2, 8) or int(rows[0, 0]) != 1:
        raise ValueError("parity dump must contain sorted atom ids 1 and 2")
    return rows[0, 5:8]


def python_first_force(*, particle_types: tuple[int, int], distance: float) -> np.ndarray:
    positions = np.asarray([[10.0, 10.0, 10.0], [10.0 + distance, 10.0, 10.0]])
    forces, _ = ka_lj_force_and_isotropic_curvature(
        positions,
        particle_types=np.asarray(particle_types),
        box_lengths=np.asarray([20.0, 20.0, 20.0]),
        potential_protocol="ka_lj_c3_switch",
    )
    return forces[0]


def run_case(
    *,
    lammps_binary: Path,
    atom_types: tuple[int, int],
    particle_types: tuple[int, int],
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="ka-c3-parity-") as directory_name:
        directory = Path(directory_name)
        input_path = directory / "in.parity"
        input_path.write_text(dimer_input(atom_types=atom_types, distance=distance))
        completed = subprocess.run(
            [str(lammps_binary), "-log", "none", "-screen", "none", "-in", str(input_path)],
            cwd=directory,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"LAMMPS parity case failed at r={distance:.17g}:\n{completed.stdout}\n{completed.stderr}"
            )
        lammps_force = load_first_force(directory / "force.dump")
    return lammps_force, python_first_force(particle_types=particle_types, distance=distance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lammps-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-11)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-10)
    args = parser.parse_args()

    binary = args.lammps_binary.resolve()
    if not binary.is_file():
        raise ValueError("lammps binary does not exist")
    if args.absolute_tolerance < 0.0 or args.relative_tolerance < 0.0:
        raise ValueError("tolerances must be nonnegative")

    rows: list[dict[str, object]] = []
    all_pass = True
    for pair, atom_types, particle_types, sigma in PAIR_CASES:
        for scaled_radius in SCALED_RADII:
            distance = sigma * scaled_radius
            observed, expected = run_case(
                lammps_binary=binary,
                atom_types=atom_types,
                particle_types=particle_types,
                distance=distance,
            )
            difference = observed - expected
            absolute_error = float(np.max(np.abs(difference)))
            reference = float(np.max(np.abs(expected)))
            threshold = args.absolute_tolerance + args.relative_tolerance * reference
            passed = absolute_error <= threshold
            all_pass = all_pass and passed
            rows.append(
                {
                    "pair": pair,
                    "scaled_radius": scaled_radius,
                    "distance": distance,
                    "lammps_fx": observed[0],
                    "python_fx": expected[0],
                    "absolute_error": absolute_error,
                    "relative_error": absolute_error / max(reference, args.absolute_tolerance),
                    "threshold": threshold,
                    "parity_pass": int(passed),
                    "potential_protocol": "ka_lj_c3_switch",
                    "thermodynamic_claim_allowed": 0,
                }
            )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    maximum_error = max(float(row["absolute_error"]) for row in rows)
    print(f"checked {len(rows)} dimer forces; max absolute error={maximum_error:.3e}; pass={all_pass}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
