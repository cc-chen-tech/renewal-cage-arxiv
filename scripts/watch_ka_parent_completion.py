#!/usr/bin/env python3
"""Passively wait for independent-parent PIDs and record completion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_parent_completion import (  # noqa: E402
    build_completion_record,
    snapshot_parent_completion,
    validate_completion_record,
    write_json_atomic,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-record", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1.0 <= args.poll_seconds <= 300.0:
        raise ValueError("poll seconds must be between one and 300")
    launch = json.loads(args.launch_record.read_text())
    manifest = json.loads(args.acquisition_manifest.read_text())
    expected_manifest_sha = sha256(args.acquisition_manifest)
    if launch.get("acquisition_manifest_sha256") != expected_manifest_sha:
        raise ValueError("launch record does not join acquisition manifest")
    parents = {parent["parent_id"]: parent for parent in manifest["parents"]}
    protocol = manifest["protocol"]
    expected_last = round(
        float(protocol["production_time_tau"]) / float(protocol["timestep_tau"])
    )
    while True:
        rows = [
            snapshot_parent_completion(
                parent_id=str(job["parent_id"]),
                pid=int(job["pid"]),
                output_directory=Path(str(job["remote_output_directory"])),
                expected_frame_count=int(
                    manifest["completion_checks"]["expected_production_frame_count"]
                ),
                expected_atom_count=int(manifest["system"]["particle_count"]),
                expected_first_timestep=0,
                expected_last_timestep=expected_last,
            )
            for job in launch["jobs"]
            if str(job["parent_id"]) in parents
        ]
        record = build_completion_record(
            rows, acquisition_manifest_sha256=expected_manifest_sha
        )
        validate_completion_record(record)
        write_json_atomic(args.output, record)
        if not args.wait or all(row["process_state"] == "exited" for row in rows):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
