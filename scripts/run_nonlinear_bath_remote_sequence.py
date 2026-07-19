#!/usr/bin/env python3
"""Run the frozen nonlinear-bath cache grid sequentially on remote compute."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from simulate_nonlinear_bath_elimination import (  # noqa: E402
    file_sha256,
    require_remote_execution,
)


_BROAD_CLAIMS = (
    "autonomous_single_particle_gle_allowed",
    "complete_event_clock_closure_allowed",
    "kramers_escape_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)


def find_conflicting_processes(
    proc_root: Path = Path("/proc"),
    *,
    current_pid: int | None = None,
) -> list[tuple[int, str]]:
    """Return active L3p or nonlinear-bath workers from a Linux proc tree."""

    own_pid = os.getpid() if current_pid is None else current_pid
    conflicts: list[tuple[int, str]] = []
    for process in sorted(
        (entry for entry in Path(proc_root).iterdir() if entry.name.isdigit()),
        key=lambda entry: int(entry.name),
    ):
        pid = int(process.name)
        if pid == own_pid:
            continue
        try:
            raw = (process / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode(errors="replace").strip()
        lowered = command.lower()
        if "l3p" in lowered or any(
            name in lowered
            for name in (
                "simulate_nonlinear_bath_elimination.py",
                "analyze_nonlinear_bath_elimination.py",
                "check_nonlinear_bath_canary.py",
            )
        ):
            conflicts.append((pid, command))
    return conflicts


def _simulation_command(
    python_executable: str,
    *,
    output_path: Path,
    mode: str,
) -> list[str]:
    command = [
        python_executable,
        str(ROOT / "scripts" / "simulate_nonlinear_bath_elimination.py"),
        "--output-path",
        str(output_path),
        "--mode",
        mode,
        "--checkpoint-interval",
        "50000",
    ]
    if output_path.exists():
        command.append("--resume")
    return command


def build_frozen_commands(
    output_dir: Path,
    *,
    artifact_prefix: Path,
    python_executable: str | None = None,
) -> list[tuple[str, list[str]]]:
    """Return the immutable, serial canary-to-analysis command graph."""

    python = sys.executable if python_executable is None else python_executable
    output = Path(output_dir)
    canary = output / "nonlinear_bath_canary.npz"
    half_step = output / "nonlinear_bath_canary_half_step.npz"
    production = output / "nonlinear_bath_production.npz"
    constant = output / "nonlinear_bath_null_constant_coupling.npz"
    no_bath = output / "nonlinear_bath_null_no_bath.npz"
    preflight = output / "nonlinear_bath_canary_preflight.json"
    return [
        (
            "canary",
            _simulation_command(python, output_path=canary, mode="canary"),
        ),
        (
            "canary-half-step",
            _simulation_command(
                python,
                output_path=half_step,
                mode="canary-half-step",
            ),
        ),
        (
            "canary-preflight",
            [
                python,
                str(ROOT / "scripts" / "check_nonlinear_bath_canary.py"),
                "--canary-cache",
                str(canary),
                "--half-step-cache",
                str(half_step),
                "--output-json",
                str(preflight),
            ],
        ),
        (
            "production",
            _simulation_command(
                python,
                output_path=production,
                mode="production",
            ),
        ),
        (
            "null-constant-coupling",
            _simulation_command(
                python,
                output_path=constant,
                mode="null-constant-coupling",
            ),
        ),
        (
            "null-no-bath",
            _simulation_command(
                python,
                output_path=no_bath,
                mode="null-no-bath",
            ),
        ),
        (
            "analysis",
            [
                python,
                str(ROOT / "scripts" / "analyze_nonlinear_bath_elimination.py"),
                "--canary-cache",
                str(canary),
                "--half-step-cache",
                str(half_step),
                "--production-cache",
                str(production),
                "--constant-coupling-cache",
                str(constant),
                "--no-bath-cache",
                str(no_bath),
                "--output-prefix",
                str(artifact_prefix),
            ],
        ),
    ]


def validate_remote_paths(
    output_dir: Path,
    artifact_prefix: Path,
) -> tuple[Path, Path]:
    """Keep all generated artifacts inside the directory protected by the lock."""

    output = Path(output_dir).resolve()
    prefix = Path(artifact_prefix).resolve()
    if prefix.parent != output or not prefix.name:
        raise ValueError("artifact prefix must be directly inside output directory")
    return output, prefix


def _run_logged(label: str, command: list[str], output_dir: Path) -> None:
    log_path = output_dir / f"{label}.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("remote sequence child has no output stream")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _write_manifests(output_dir: Path, artifact_prefix: Path) -> None:
    summary_path = Path(f"{artifact_prefix}_summary.csv")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("remote sequence requires exactly one verdict row")
    verdict = rows[0]
    for claim in _BROAD_CLAIMS:
        if claim not in verdict or float(verdict[claim]) != 0.0:
            raise ValueError(f"remote sequence found an open broad claim: {claim}")
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    manifest_path = output_dir / "nonlinear_bath_environment_manifest.json"
    manifest = {
        "git_commit": git_commit,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "remote_compute_marker": os.environ.get("RENEWAL_CAGE_REMOTE_COMPUTE"),
        "runner_sha256": file_sha256(Path(__file__)),
        "simulator_sha256": file_sha256(
            ROOT / "scripts" / "simulate_nonlinear_bath_elimination.py"
        ),
        "analyzer_sha256": file_sha256(
            ROOT / "scripts" / "analyze_nonlinear_bath_elimination.py"
        ),
        "gle_sha256": file_sha256(ROOT / "src" / "nonlinear_bath_gle.py"),
        "diagnostics_sha256": file_sha256(
            ROOT / "src" / "nonlinear_bath_diagnostics.py"
        ),
        "broad_claims": {claim: float(verdict[claim]) for claim in _BROAD_CLAIMS},
    }
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    sha_path = output_dir / "nonlinear_bath_sha256.txt"
    artifact_paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path != sha_path and not path.name.endswith(".lock")
    )
    sha_path.write_text(
        "".join(
            f"{file_sha256(path)}  {path.name}\n" for path in artifact_paths
        ),
        encoding="utf-8",
    )


def run_remote_sequence(output_dir: Path, *, artifact_prefix: Path) -> None:
    """Audit, lock, execute one numerical child at a time, and manifest."""

    require_remote_execution()
    output, prefix = validate_remote_paths(output_dir, artifact_prefix)
    conflicts = find_conflicting_processes()
    if conflicts:
        rendered = "; ".join(f"PID {pid}: {command}" for pid, command in conflicts)
        raise RuntimeError(f"remote process audit found active work: {rendered}")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".nonlinear_bath.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("nonlinear-bath remote sequence lock is active") from error
        for label, command in build_frozen_commands(
            output,
            artifact_prefix=prefix,
        ):
            _run_logged(label, command, output)
        _write_manifests(output, prefix)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_remote_sequence(args.output_dir, artifact_prefix=args.artifact_prefix)


if __name__ == "__main__":
    main()
