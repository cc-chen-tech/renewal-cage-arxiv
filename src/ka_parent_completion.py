"""Passive, fail-closed completion records for independent KA parent runs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


CLAIM_FLAGS = {
    "positive_memory_closure_claim_allowed": 0,
    "complete_microscopic_closure_claim_allowed": 0,
    "spatial_facilitation_claim_allowed": 0,
    "thermodynamic_glass_transition_claim_allowed": 0,
}
ERROR_PATTERNS = (
    re.compile(r"\bERROR\b", re.IGNORECASE),
    re.compile(r"\bLost atoms\b", re.IGNORECASE),
    re.compile(r"\bSegmentation(?: fault)?\b", re.IGNORECASE),
    re.compile(r"\bKilled\b", re.IGNORECASE),
    re.compile(r"\bNon-numeric\b", re.IGNORECASE),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _exit_code(path: Path) -> tuple[int | None, str]:
    if not path.is_file():
        return None, "unavailable_original_launcher_did_not_persist_wait_status"
    try:
        text = path.read_text().strip()
        value = int(text)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid persisted exit code: {path}") from error
    return value, "persisted_launcher_wait_status"


def _trajectory_metadata(path: Path) -> dict[str, object]:
    frame_count = 0
    atom_counts: set[int] = set()
    first_timestep: int | None = None
    last_timestep: int | None = None
    truncated = False
    try:
        with path.open() as handle:
            while True:
                marker = handle.readline()
                if marker == "":
                    break
                if marker.strip() != "ITEM: TIMESTEP":
                    truncated = True
                    break
                timestep_text = handle.readline()
                if timestep_text == "":
                    truncated = True
                    break
                timestep = int(timestep_text)
                if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                    truncated = True
                    break
                count_text = handle.readline()
                if count_text == "":
                    truncated = True
                    break
                atom_count = int(count_text)
                if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                    truncated = True
                    break
                if any(handle.readline() == "" for _ in range(3)):
                    truncated = True
                    break
                if not handle.readline().startswith("ITEM: ATOMS"):
                    truncated = True
                    break
                if any(handle.readline() == "" for _ in range(atom_count)):
                    truncated = True
                    break
                first_timestep = timestep if first_timestep is None else first_timestep
                last_timestep = timestep
                atom_counts.add(atom_count)
                frame_count += 1
    except (OSError, ValueError):
        truncated = True
    return {
        "production_frame_count": frame_count,
        "first_production_timestep": first_timestep,
        "last_production_timestep": last_timestep,
        "observed_atom_counts": sorted(atom_counts),
        "trajectory_parse_complete": int(not truncated and frame_count > 0),
    }


def _path_metadata(path: Path, prefix: str) -> dict[str, object]:
    if not path.is_file():
        return {
            f"{prefix}_present": 0,
            f"{prefix}_size_bytes": 0,
            f"{prefix}_sha256": "missing",
        }
    return {
        f"{prefix}_present": 1,
        f"{prefix}_size_bytes": path.stat().st_size,
        f"{prefix}_sha256": file_sha256(path),
    }


def snapshot_parent_completion(
    *,
    parent_id: str,
    pid: int,
    output_directory: Path,
    expected_frame_count: int,
    expected_atom_count: int,
    expected_first_timestep: int,
    expected_last_timestep: int,
    observed_at_utc: str | None = None,
) -> dict[str, object]:
    """Inspect one run without changing it or inferring an unrecorded exit code."""

    root = Path(output_directory)
    trajectory_path = root / "trajectory.lammpstrj"
    log_path = root / "log.lammps"
    final_restart_path = root / "final.restart"
    manifest_path = root / "parent_manifest.json"
    exit_code, exit_code_source = _exit_code(root / "run.exitcode")
    trajectory = _trajectory_metadata(trajectory_path)
    trajectory.update(_path_metadata(trajectory_path, "trajectory"))
    final_restart = _path_metadata(final_restart_path, "final_restart")
    parent_manifest = _path_metadata(manifest_path, "parent_manifest")
    log = _path_metadata(log_path, "lammps_log")
    log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    error_count = sum(len(pattern.findall(log_text)) for pattern in ERROR_PATTERNS)
    process_state = "running" if _pid_is_alive(int(pid)) else "exited"
    output_checks_pass = (
        int(trajectory["trajectory_parse_complete"]) == 1
        and int(trajectory["production_frame_count"]) == int(expected_frame_count)
        and trajectory["first_production_timestep"] == int(expected_first_timestep)
        and trajectory["last_production_timestep"] == int(expected_last_timestep)
        and trajectory["observed_atom_counts"] == [int(expected_atom_count)]
        and int(final_restart["final_restart_present"]) == 1
        and int(parent_manifest["parent_manifest_present"]) == 1
        and error_count == 0
    )
    if exit_code is not None and exit_code != 0:
        completion_state = "failed_explicit_nonzero_exit"
    elif process_state == "running":
        completion_state = "running"
    elif not output_checks_pass:
        completion_state = "failed_output_completeness"
    elif exit_code is None:
        completion_state = "blocked_missing_observed_exit_code"
    else:
        completion_state = "complete_verified"
    return {
        "parent_id": str(parent_id),
        "pid": int(pid),
        "process_state": process_state,
        "observed_at_utc": observed_at_utc
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "remote_output_directory": str(root),
        "exit_code": exit_code,
        "exit_code_source": exit_code_source,
        **trajectory,
        **final_restart,
        **parent_manifest,
        **log,
        "error_match_count": error_count,
        "expected_production_frame_count": int(expected_frame_count),
        "expected_atom_count": int(expected_atom_count),
        "expected_first_production_timestep": int(expected_first_timestep),
        "expected_last_production_timestep": int(expected_last_timestep),
        "output_completeness_pass": int(output_checks_pass),
        "completion_state": completion_state,
        "scientific_ingestion_allowed": int(completion_state == "complete_verified"),
        **CLAIM_FLAGS,
    }


def build_completion_record(
    jobs: Sequence[Mapping[str, object]], *, acquisition_manifest_sha256: str
) -> dict[str, object]:
    if not jobs:
        raise ValueError("completion record requires at least one parent")
    if len(acquisition_manifest_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in acquisition_manifest_sha256
    ):
        raise ValueError("acquisition manifest SHA256 is invalid")
    states = {str(job["completion_state"]) for job in jobs}
    if states == {"complete_verified"}:
        state = "complete_verified"
    elif any(value.startswith("failed_") for value in states):
        state = "failed"
    elif "running" in states:
        state = "running"
    else:
        state = "blocked"
    return {
        "schema_version": 1,
        "completion_state": state,
        "acquisition_manifest_sha256": acquisition_manifest_sha256,
        "observed_at_utc": max(str(job["observed_at_utc"]) for job in jobs),
        "watcher_mode": "passive_non_destructive",
        "credentials_persisted": False,
        "jobs": [dict(job) for job in jobs],
        "claim_flags": dict(CLAIM_FLAGS),
    }


def validate_completion_record(record: Mapping[str, object]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    if result.get("schema_version") != 1:
        raise ValueError("completion schema_version must equal one")
    if result.get("watcher_mode") != "passive_non_destructive":
        raise ValueError("completion watcher mode is invalid")
    if result.get("credentials_persisted") is not False:
        raise ValueError("completion artifact must not persist credentials")
    digest = str(result.get("acquisition_manifest_sha256", ""))
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("completion manifest SHA256 is invalid")
    jobs = result.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("completion jobs must be a nonempty list")
    parent_ids = [str(job.get("parent_id", "")) for job in jobs]
    if any(not value for value in parent_ids) or len(set(parent_ids)) != len(parent_ids):
        raise ValueError("completion parent IDs must be nonempty and unique")
    for job in jobs:
        allowed = int(job.get("scientific_ingestion_allowed", -1))
        expected = int(job.get("completion_state") == "complete_verified")
        if allowed != expected:
            raise ValueError("completion ingestion flag disagrees with state")
        if expected:
            if int(job.get("exit_code", -1)) != 0 or int(
                job.get("output_completeness_pass", 0)
            ) != 1:
                raise ValueError("verified completion lacks explicit zero exit or outputs")
            for key in (
                "trajectory_sha256",
                "final_restart_sha256",
                "parent_manifest_sha256",
                "lammps_log_sha256",
            ):
                value = str(job.get(key, ""))
                if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                    raise ValueError(f"verified completion has invalid {key}")
        if any(int(job.get(key, 0)) != 0 for key in CLAIM_FLAGS):
            raise ValueError("completion job claim flags must remain zero")
    if result.get("completion_state") == "complete_verified" and not all(
        int(job["scientific_ingestion_allowed"]) == 1 for job in jobs
    ):
        raise ValueError("aggregate completion state disagrees with parent jobs")
    if any(int(result.get("claim_flags", {}).get(key, 0)) != 0 for key in CLAIM_FLAGS):
        raise ValueError("aggregate completion claim flags must remain zero")
    return result


def write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
