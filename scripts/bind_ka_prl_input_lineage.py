#!/usr/bin/env python3
"""Bind PRL heldout/environment/spectral tables to exact parent trajectories."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_prl_input_lineage import (  # noqa: E402
    bind_parent_lineage,
    trajectory_identities_from_ensemble,
)
from ka_prl_memory_closure import parent_identifier  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"cannot read CSV input: {path}") from error
    if not rows:
        raise ValueError(f"CSV input is empty: {path}")
    return rows


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty lineage table: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("lineage rows must use one rectangular schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _temperature_rows(
    rows: Sequence[Mapping[str, object]], temperature: float
) -> list[Mapping[str, object]]:
    output = [
        row
        for row in rows
        if abs(float(row["temperature"]) - temperature) <= 1e-12
    ]
    if not output:
        raise ValueError(f"provenance has no rows for T={temperature}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Embed verified parent ID, source hash, and complete trajectory hash "
            "in all PRL auxiliary inputs."
        )
    )
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--low-ensemble-directory", type=Path, required=True)
    parser.add_argument("--high-ensemble-directory", type=Path, required=True)
    for name in ("low-heldout", "high-heldout", "low-spectral", "high-spectral"):
        parser.add_argument(f"--{name}-input", type=Path, required=True)
        parser.add_argument(f"--{name}-output", type=Path, required=True)
    parser.add_argument("--environment-input", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--output-trajectory-identities", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    provenance = read_rows(args.provenance)
    low_provenance = _temperature_rows(provenance, 0.45)
    high_provenance = _temperature_rows(provenance, 0.58)
    low_trajectories = trajectory_identities_from_ensemble(
        low_provenance,
        ensemble_directory=args.low_ensemble_directory,
        temperature=0.45,
    )
    high_trajectories = trajectory_identities_from_ensemble(
        high_provenance,
        ensemble_directory=args.high_ensemble_directory,
        temperature=0.58,
    )
    table_specs = (
        (
            args.low_heldout_input,
            args.low_heldout_output,
            low_provenance,
            low_trajectories,
            "temperature",
        ),
        (
            args.high_heldout_input,
            args.high_heldout_output,
            high_provenance,
            high_trajectories,
            "temperature",
        ),
        (
            args.low_spectral_input,
            args.low_spectral_output,
            low_provenance,
            low_trajectories,
            "temperature",
        ),
        (
            args.high_spectral_input,
            args.high_spectral_output,
            high_provenance,
            high_trajectories,
            "temperature",
        ),
        (
            args.environment_input,
            args.environment_output,
            provenance,
            [*low_trajectories, *high_trajectories],
            "temperature_group",
        ),
    )
    for source, destination, parent_rows, trajectory_rows, table_kind in table_specs:
        write_rows(
            destination,
            bind_parent_lineage(
                read_rows(source),
                provenance_rows=parent_rows,
                trajectory_rows=trajectory_rows,
                table_kind=table_kind,
            ),
        )

    provenance_by_key = {
        (float(row["temperature"]), int(float(row["replicate"]))): row
        for row in provenance
    }
    identity_rows = []
    for trajectory in [*low_trajectories, *high_trajectories]:
        key = (float(trajectory["temperature"]), int(trajectory["replicate"]))
        source = provenance_by_key[key]
        identity_rows.append(
            {
                "temperature": key[0],
                "replicate": key[1],
                "parent_id": parent_identifier(source),
                "source_doi": source["source_doi"],
                "source_sha256": source["source_sha256"],
                "source_frame_index": int(float(source["source_frame_index"])),
                "velocity_seed": int(float(source["velocity_seed"])),
                **{
                    field: trajectory[field]
                    for field in (
                        "trajectory_sha256",
                        "trajectory_size_bytes",
                        "trajectory_hash_scope",
                    )
                },
                "positive_memory_closure_claim_allowed": 0,
                "complete_microscopic_closure_claim_allowed": 0,
                "spatial_facilitation_claim_allowed": 0,
                "thermodynamic_glass_transition_claim_allowed": 0,
            }
        )
    write_rows(args.output_trajectory_identities, identity_rows)


if __name__ == "__main__":
    main()
