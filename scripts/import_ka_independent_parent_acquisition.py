#!/usr/bin/env python3
"""Import completed KA parents and run the frozen parent-first PRL gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.ka_parent_completion import file_sha256, write_json_atomic  # noqa: E402
from src.ka_parent_ingestion import (  # noqa: E402
    completion_import_blockers,
    run_frozen_six_ablation_gate,
    split_parent_trajectory,
)
from src.ka_prl_memory_closure import (  # noqa: E402
    build_claim_ledger,
    classify_memory_closure_gate,
)
from src.ka_replicates import load_lammps_custom_trajectory  # noqa: E402


def _canonical(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty acquisition artifact: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("acquisition artifact rows must have one rectangular schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _canonical(row[key]) for key in fields})
    temporary.replace(path)


def _parent_paths(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        try:
            parent_id, raw_path = value.split("=", 1)
        except ValueError as error:
            raise ValueError("parent output must use PARENT_ID=/absolute/path") from error
        path = Path(raw_path)
        if not parent_id or not path.is_absolute() or parent_id in output:
            raise ValueError("parent output IDs must be unique and paths absolute")
        output[parent_id] = path
    return output


def _write_blocked_bundle(
    output: Path,
    ledger: Sequence[Mapping[str, object]],
    blocker: Mapping[str, object],
) -> None:
    gate = classify_memory_closure_gate(
        parent_summaries=[], blockers=[blocker], upper_control_parents=[]
    )
    gate.update(
        {
            "run_temperature": 0.45,
            "diagnostic_realizations": 64,
            "six_ablation_gate_executed": 0,
            "completion_gate_pass": 0,
            "heldout_observables_used_as_model_inputs": 0,
            "gate_or_claim_tuned_after_results": 0,
        }
    )
    write_rows(output / "parent_ledger.csv", ledger)
    write_rows(output / "completion_blockers.csv", [blocker])
    write_rows(output / "gate.csv", [gate])
    write_rows(output / "claim_ledger.csv", build_claim_ledger(gate))
    write_json_atomic(
        output / "run_status.json",
        {
            "schema_version": 1,
            "pipeline_state": "completion_blocked_fail_closed",
            "blocker_state": blocker["blocker_state"],
            "trajectory_files_opened": 0,
            "scientific_gate_executed": False,
            "claims_remain_closed": True,
            "credentials_persisted": False,
        },
    )


def _prepared_parents(
    *,
    manifest: Mapping[str, object],
    completion: Mapping[str, object],
    parent_paths: Mapping[str, Path],
) -> list[dict[str, object]]:
    parents = {str(row["parent_id"]): row for row in manifest["parents"]}
    jobs = {str(row["parent_id"]): row for row in completion["jobs"]}
    if set(parent_paths) != set(parents) or len(parents) != 3:
        raise ValueError("full primary import requires paths for exactly three parents")
    prepared = []
    for replicate, parent_id in enumerate(sorted(parents), start=1):
        root = parent_paths[parent_id]
        trajectory_path = root / "trajectory.lammpstrj"
        parent_manifest_path = root / "parent_manifest.json"
        trajectory_sha256 = file_sha256(trajectory_path)
        parent_manifest_sha256 = file_sha256(parent_manifest_path)
        job = jobs[parent_id]
        if (
            trajectory_sha256 != str(job["trajectory_sha256"])
            or trajectory_path.stat().st_size != int(job["trajectory_size_bytes"])
            or parent_manifest_sha256 != str(job["parent_manifest_sha256"])
        ):
            raise ValueError("local acquisition output hash disagrees with completion artifact")
        local_manifest = json.loads(parent_manifest_path.read_text())
        if str(local_manifest.get("parent_id")) != parent_id:
            raise ValueError("parent manifest ID disagrees with acquisition manifest")
        trajectory = load_lammps_custom_trajectory(trajectory_path)
        split = split_parent_trajectory(trajectory)
        prepared.append(
            {
                "temperature": 0.45,
                "replicate": replicate,
                "parent_id": parent_id,
                "velocity_seed": int(parents[parent_id]["velocity_seed"]),
                "trajectory_sha256": trajectory_sha256,
                "trajectory_size_bytes": trajectory_path.stat().st_size,
                "parent_manifest_sha256": parent_manifest_sha256,
                "calibration_blocks": split["calibration_blocks"],
                "heldout_blocks": split["heldout_blocks"],
            }
        )
    return prepared


def _write_full_bundle(output: Path, result: Mapping[str, object]) -> None:
    tables = {
        "parent_ledger.csv": "parent_ledger",
        "completion_blockers.csv": "blockers",
        "heldout_targets.csv": "heldout_target_rows",
        "stationarity.csv": "stationarity_rows",
        "environment.csv": "environment_rows",
        "spectral.csv": "spectral_rows",
        "input_lineage.csv": "input_lineage_rows",
        "realization_rows.csv": "realization_rows",
        "restart_summary.csv": "restart_summaries",
        "parent_summary.csv": "parent_summaries",
        "model_verdicts.csv": "model_verdicts",
    }
    for filename, key in tables.items():
        write_rows(output / filename, result[key])
    write_rows(output / "gate.csv", [result["gate"]])
    write_rows(output / "claim_ledger.csv", result["claim_ledger"])
    write_json_atomic(
        output / "run_status.json",
        {
            "schema_version": 1,
            "pipeline_state": str(result["gate"]["mechanism_state"]),
            "trajectory_files_opened": 3,
            "scientific_gate_executed": True,
            "diagnostic_realizations": 64,
            "six_ablation_models": sorted(
                {
                    row["model"]
                    for row in result["realization_rows"]
                    if row["model"] != "contiguous_empirical_upper_control"
                }
            ),
            "credentials_persisted": False,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--completion-artifact", type=Path, required=True)
    parser.add_argument(
        "--parent-output",
        action="append",
        default=[],
        help="repeat PARENT_ID=/absolute/path for every eligible parent",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = json.loads(args.acquisition_manifest.read_text())
    completion = json.loads(args.completion_artifact.read_text())
    manifest_sha256 = file_sha256(args.acquisition_manifest)
    completion_ledger, completion_blocker = completion_import_blockers(
        manifest,
        completion,
        acquisition_manifest_sha256=manifest_sha256,
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if completion_blocker["blocker_state"] != "eligible_for_trajectory_import":
        if args.parent_output:
            raise ValueError("parent outputs cannot override a completion blocker")
        _write_blocked_bundle(
            args.output_directory, completion_ledger, completion_blocker
        )
        return
    prepared = _prepared_parents(
        manifest=manifest,
        completion=completion,
        parent_paths=_parent_paths(args.parent_output),
    )
    result = run_frozen_six_ablation_gate(
        prepared, realizations=64, workers=args.workers, fixture_only=False
    )
    _write_full_bundle(args.output_directory, result)


if __name__ == "__main__":
    main()
