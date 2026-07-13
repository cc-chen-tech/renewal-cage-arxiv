#!/usr/bin/env python3
"""Re-equilibrate a KA restart under the analytic C3-switched pair potential."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_generator_response import ka_c3_lepton_pair_commands  # noqa: E402


THERMO_COLUMNS = ("step", "time", "temperature", "potential_energy", "kinetic_energy", "total_energy", "pressure")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def switched_parent_lammps_input(
    *,
    source_restart: Path,
    temperature: float,
    friction: float,
    seed: int,
    timestep: float,
    run_steps: int,
    thermo_interval: int,
) -> str:
    """Return a hard-restart to C3-potential re-equilibration input."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(friction) or friction <= 0.0:
        raise ValueError("friction must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
        raise ValueError("seed must be a positive integer")
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("timestep must be finite and positive")
    if isinstance(run_steps, bool) or not isinstance(run_steps, int) or run_steps < 1:
        raise ValueError("run_steps must be a positive integer")
    if isinstance(thermo_interval, bool) or not isinstance(thermo_interval, int) or thermo_interval < 1:
        raise ValueError("thermo_interval must be a positive integer")
    return f"""units lj
atom_style atomic
processors 1 1 1
read_restart {Path(source_restart).resolve()}

{ka_c3_lepton_pair_commands()}
reset_timestep 0
timestep {timestep:.17g}
thermo {thermo_interval}
thermo_style custom step time temp pe ke etotal press
run 0

fix integrator all nve
fix bath all langevin {temperature:g} {temperature:g} {friction:g} {seed}
run {run_steps}
write_restart equilibrated_c3.restart
"""


def parse_thermodynamic_log(path: Path) -> list[dict[str, float]]:
    """Extract all seven-column thermo rows from a LAMMPS log."""

    rows: list[dict[str, float]] = []
    in_table = False
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields == ["Step", "Time", "Temp", "PotEng", "KinEng", "TotEng", "Press"]:
            in_table = True
            continue
        if in_table and (line.startswith("Loop time") or line.startswith("ERROR")):
            in_table = False
            continue
        if not in_table or len(fields) != len(THERMO_COLUMNS):
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            continue
        rows.append(dict(zip(THERMO_COLUMNS, values)))
    if not rows:
        raise ValueError("LAMMPS log contains no thermodynamic rows")
    return rows


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - np.mean(x)
    denominator = float(centered @ centered)
    if denominator == 0.0:
        return 0.0
    return float(centered @ (y - np.mean(y)) / denominator)


def equilibration_summary(rows: list[dict[str, float]]) -> dict[str, float | bool]:
    values = np.asarray([[row[column] for column in THERMO_COLUMNS] for row in rows], dtype=float)
    finite = bool(np.all(np.isfinite(values)))
    dynamic = values[values[:, 0] > 0.0]
    if not len(dynamic):
        dynamic = values[-1:]
    final_half = dynamic[len(dynamic) // 2 :]
    time = final_half[:, 1]
    temperature = final_half[:, 2]
    total_energy = final_half[:, 5]
    return {
        "thermo_row_count": float(len(values)),
        "final_half_row_count": float(len(final_half)),
        "all_thermodynamic_values_finite": finite,
        "final_temperature": float(dynamic[-1, 2]),
        "final_pressure": float(dynamic[-1, 6]),
        "final_half_temperature_mean": float(np.mean(temperature)),
        "final_half_temperature_std": float(np.std(temperature)),
        "final_half_temperature_slope_per_tau": linear_slope(time, temperature),
        "final_half_total_energy_mean": float(np.mean(total_energy)),
        "final_half_total_energy_std": float(np.std(total_energy)),
        "final_half_total_energy_slope_per_tau": linear_slope(time, total_energy),
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-restart", type=Path, required=True)
    parser.add_argument("--lammps-binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=712451)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--duration", type=float, required=True)
    args = parser.parse_args()

    source_restart = args.source_restart.resolve()
    binary = args.lammps_binary.resolve()
    output_directory = args.output_directory.resolve()
    if not source_restart.is_file():
        raise ValueError("source restart does not exist")
    if not binary.is_file():
        raise ValueError("lammps binary does not exist")
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        raise ValueError("duration must be finite and positive")
    run_steps = int(round(args.duration / args.timestep))
    if not math.isclose(run_steps * args.timestep, args.duration, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("duration must be an integer multiple of timestep")
    thermo_interval = max(1, run_steps // 100)
    output_directory.mkdir(parents=True, exist_ok=True)
    input_path = output_directory / "in.c3_parent"
    log_path = output_directory / "thermodynamic.log"
    restart_path = output_directory / "equilibrated_c3.restart"
    trace_path = output_directory / "thermodynamic_trace.csv"
    summary_path = output_directory / "equilibration_summary.csv"
    for stale in (log_path, restart_path, trace_path, summary_path):
        stale.unlink(missing_ok=True)

    input_text = switched_parent_lammps_input(
        source_restart=source_restart,
        temperature=args.temperature,
        friction=args.friction,
        seed=args.seed,
        timestep=args.timestep,
        run_steps=run_steps,
        thermo_interval=thermo_interval,
    )
    input_path.write_text(input_text)
    completed = subprocess.run(
        [str(binary), "-log", str(log_path), "-screen", "none", "-in", str(input_path)],
        cwd=output_directory,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"LAMMPS C3 parent preparation failed:\n{completed.stdout}\n{completed.stderr}")
    if not restart_path.is_file():
        raise ValueError("LAMMPS did not produce the C3 parent restart")

    rows = parse_thermodynamic_log(log_path)
    with trace_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(THERMO_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = equilibration_summary(rows)
    if not summary["all_thermodynamic_values_finite"]:
        raise ValueError("C3 parent thermodynamic trace contains nonfinite values")
    summary_payload = {
        **summary,
        "duration_tau": args.duration,
        "potential_protocol": "ka_lj_c3_switch",
        "stationarity_diagnostic_only": True,
        "equilibrium_glass_claim_allowed": False,
        "thermodynamic_claim_allowed": False,
    }
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_payload), lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary_payload)

    manifest = {
        "protocol": "ka_c3_switched_parent_rethermalization",
        "potential_protocol": "ka_lj_c3_switch",
        "source_restart": str(source_restart),
        "source_restart_sha256": file_sha256(source_restart),
        "lammps_binary": str(binary),
        "lammps_binary_sha256": file_sha256(binary),
        "output_restart": str(restart_path),
        "output_restart_sha256": file_sha256(restart_path),
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "thermodynamic_log_path": str(log_path),
        "thermodynamic_trace_path": str(trace_path),
        "equilibration_summary_path": str(summary_path),
        "pair_commands": ka_c3_lepton_pair_commands(),
        "temperature": args.temperature,
        "friction": args.friction,
        "seed": args.seed,
        "timestep_tau": args.timestep,
        "duration_tau": args.duration,
        "run_steps": run_steps,
        "stationarity_diagnostic_only": True,
        "equilibrium_glass_claim_allowed": False,
        "thermodynamic_claim_allowed": False,
    }
    atomic_json(output_directory / "manifest.json", manifest)
    print(
        f"prepared {restart_path}; rows={int(summary['thermo_row_count'])}; "
        f"T_final={summary['final_temperature']:.6g}; finite=True"
    )


if __name__ == "__main__":
    main()
