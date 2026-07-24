"""Exact parent and trajectory lineage binding for PRL auxiliary inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


LINEAGE_FIELDS = (
    "parent_id",
    "source_doi",
    "source_sha256",
    "source_frame_index",
    "velocity_seed",
    "trajectory_sha256",
    "trajectory_size_bytes",
    "trajectory_hash_scope",
)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _key(row: Mapping[str, object], *, table_kind: str) -> tuple[float, int]:
    if table_kind == "temperature":
        try:
            temperature = float(row["temperature"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("lineage row lacks a valid temperature") from error
    elif table_kind == "temperature_group":
        group = str(row.get("temperature_group", ""))
        try:
            temperature = {"low": 0.45, "high": 0.58}[group]
        except KeyError as error:
            raise ValueError("lineage row has an unknown temperature group") from error
    else:
        raise ValueError("table_kind must be temperature or temperature_group")
    try:
        replicate = int(float(row["replicate"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("lineage row lacks a valid replicate") from error
    return temperature, replicate


def _unique_by_key(
    rows: Sequence[Mapping[str, object]], *, label: str
) -> dict[tuple[float, int], Mapping[str, object]]:
    output: dict[tuple[float, int], Mapping[str, object]] = {}
    for row in rows:
        key = _key(row, table_kind="temperature")
        if key in output:
            raise ValueError(f"{label} rows must be unique by temperature and replicate")
        output[key] = row
    if not output:
        raise ValueError(f"{label} rows must not be empty")
    return output


def _parent_id(row: Mapping[str, object]) -> str:
    doi = str(row.get("source_doi", "")).strip()
    source_sha256 = str(row.get("source_sha256", ""))
    if not doi or not _is_sha256(source_sha256):
        raise ValueError("parent provenance requires DOI and SHA256")
    return f"{doi}:{source_sha256}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot hash trajectory input: {path}") from error
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read lineage manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("lineage manifest root must be an object")
    return value


def trajectory_identities_from_ensemble(
    provenance_rows: Sequence[Mapping[str, object]],
    *,
    ensemble_directory: Path,
    temperature: float,
) -> list[dict[str, object]]:
    """Verify parent manifests and hash every complete child trajectory once."""

    provenance = _unique_by_key(provenance_rows, label="provenance")
    expected = {
        key: row for key, row in provenance.items() if abs(key[0] - temperature) <= 1e-12
    }
    if not expected or len(expected) != len(provenance):
        raise ValueError("provenance rows must contain exactly the requested temperature")
    ensemble_directory = Path(ensemble_directory)
    manifest = _json_object(ensemble_directory / "ensemble_manifest.json")
    first = next(iter(expected.values()))
    if (
        abs(float(manifest.get("temperature", float("nan"))) - temperature) > 1e-12
        or str(manifest.get("source_doi", "")) != str(first["source_doi"])
        or str(manifest.get("source_sha256", "")) != str(first["source_sha256"])
    ):
        raise ValueError("ensemble manifest disagrees with parent provenance")
    specs = {
        (temperature, int(float(item["replicate"]))): item
        for item in manifest.get("replicates", [])
        if isinstance(item, dict) and "replicate" in item
    }
    if set(specs) != set(expected):
        raise ValueError("ensemble manifest does not cover the provenance restart set")

    output: list[dict[str, object]] = []
    for key in sorted(expected):
        source = expected[key]
        spec = specs[key]
        if (
            int(float(spec.get("source_frame_index", -1)))
            != int(float(source["source_frame_index"]))
            or int(float(spec.get("velocity_seed", -1)))
            != int(float(source["velocity_seed"]))
        ):
            raise ValueError("ensemble child specification disagrees with provenance")
        child = ensemble_directory / str(spec.get("directory", ""))
        child_manifest = _json_object(child / "manifest.json")
        if (
            abs(float(child_manifest.get("temperature", float("nan"))) - temperature) > 1e-12
            or str(child_manifest.get("source_doi", "")) != str(source["source_doi"])
            or str(child_manifest.get("source_sha256", ""))
            != str(source["source_sha256"])
            or int(float(child_manifest.get("source_frame_index", -1)))
            != int(float(source["source_frame_index"]))
            or int(float(child_manifest.get("velocity_seed", -1)))
            != int(float(source["velocity_seed"]))
        ):
            raise ValueError("child manifest disagrees with parent provenance")
        trajectory = child / "trajectory.lammpstrj"
        try:
            size = trajectory.stat().st_size
        except OSError as error:
            raise ValueError(f"cannot stat trajectory input: {trajectory}") from error
        if size <= 0:
            raise ValueError("trajectory input must be nonempty")
        output.append(
            {
                "temperature": temperature,
                "replicate": key[1],
                "trajectory_sha256": _file_sha256(trajectory),
                "trajectory_size_bytes": size,
                "trajectory_hash_scope": "complete_file",
            }
        )
    return output


def bind_parent_lineage(
    rows: Sequence[Mapping[str, object]],
    *,
    provenance_rows: Sequence[Mapping[str, object]],
    trajectory_rows: Sequence[Mapping[str, object]],
    table_kind: str,
) -> list[dict[str, object]]:
    """Embed an exact parent/complete-trajectory join in every auxiliary row."""

    if not rows:
        raise ValueError("auxiliary table must not be empty")
    provenance = _unique_by_key(provenance_rows, label="provenance")
    trajectories = _unique_by_key(trajectory_rows, label="trajectory")
    if set(provenance) != set(trajectories):
        raise ValueError("trajectory identities do not cover the provenance restart set")

    output: list[dict[str, object]] = []
    observed_keys: set[tuple[float, int]] = set()
    for row in rows:
        key = _key(row, table_kind=table_kind)
        observed_keys.add(key)
        if key not in provenance:
            raise ValueError("auxiliary row refers to an unknown parent restart")
        source = provenance[key]
        trajectory = trajectories[key]
        digest = str(trajectory.get("trajectory_sha256", ""))
        if not _is_sha256(digest):
            raise ValueError("trajectory identity requires a complete-file SHA256")
        try:
            trajectory_size = int(float(trajectory["trajectory_size_bytes"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("trajectory identity requires a positive file size") from error
        if trajectory_size <= 0 or trajectory.get("trajectory_hash_scope") != "complete_file":
            raise ValueError("trajectory identity must cover a positive complete file")
        expected = {
            "parent_id": _parent_id(source),
            "source_doi": str(source["source_doi"]),
            "source_sha256": str(source["source_sha256"]),
            "source_frame_index": int(float(source["source_frame_index"])),
            "velocity_seed": int(float(source["velocity_seed"])),
            "trajectory_sha256": digest,
            "trajectory_size_bytes": trajectory_size,
            "trajectory_hash_scope": "complete_file",
        }
        for field, value in expected.items():
            if field in row and str(row[field]) != str(value):
                raise ValueError(f"embedded {field} conflicts with verified lineage")
        for field, value in row.items():
            if field.endswith("claim_allowed") and float(value) != 0.0:
                raise ValueError("lineage binding cannot open a scientific claim")
        bound = dict(row)
        bound.update(expected)
        output.append(bound)
    if observed_keys != set(provenance):
        raise ValueError("auxiliary table does not cover the verified parent restart set")
    return output
