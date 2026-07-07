#!/usr/bin/env python3
"""Bounded GlassBench first-NPZ particle-coordinate cache helpers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import lzma
import tarfile
import zlib
from collections.abc import Callable
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in shape)


def _cache_file_name(system_id: str, temperature: str, member_name: str) -> str:
    stem = member_name.rsplit("/", 1)[-1]
    if stem.endswith(".npz"):
        stem = stem[:-4]
    safe_temperature = str(temperature).replace(".", "_")
    return f"glassbench_{system_id.lower()}_T{safe_temperature}_{stem}_positions.npz"


def extract_first_npz_positions_cache_from_xz_bytes(
    *,
    xz_bytes: bytes,
    first_npz_member: str,
    expected_npz_md5: str,
    target_path: Path,
    system_id: str,
    temperature: str,
    source_path: str,
    compressed_probe_range_start: int,
    compressed_probe_range_end: int,
    probe_encoding: str = "xz",
    tar_read_limit_bytes: int = 8_000_000,
) -> dict[str, float | str]:
    """Extract one first-NPZ coordinate cache from a bounded xz-compressed tar prefix."""

    if not xz_bytes:
        raise ValueError("xz_bytes must be nonempty")
    if not first_npz_member:
        raise ValueError("first_npz_member must be nonempty")
    if not expected_npz_md5:
        raise ValueError("expected_npz_md5 must be nonempty")

    if probe_encoding == "xz":
        xz_stream = xz_bytes
    elif probe_encoding == "zip_deflate_xz":
        xz_stream = zlib.decompressobj(-15).decompress(xz_bytes)
    else:
        raise ValueError(f"unknown probe_encoding: {probe_encoding}")

    with lzma.LZMAFile(io.BytesIO(xz_stream)) as xz_file:
        tar_prefix = xz_file.read(tar_read_limit_bytes)
    npz_bytes = b""
    with tarfile.open(fileobj=io.BytesIO(tar_prefix), mode="r|") as archive:
        for member in archive:
            if member.name == first_npz_member:
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"missing NPZ member: {first_npz_member}")
                npz_bytes = extracted.read()
                break
    if not npz_bytes:
        raise ValueError(f"missing NPZ member: {first_npz_member}")

    npz_md5 = hashlib.md5(npz_bytes).hexdigest()
    if npz_md5 != expected_npz_md5:
        raise ValueError(f"NPZ md5 mismatch: expected {expected_npz_md5}, got {npz_md5}")

    with np.load(io.BytesIO(npz_bytes)) as source_npz:
        if "positions" not in source_npz:
            raise ValueError("positions array missing from NPZ member")
        positions = np.asarray(source_npz["positions"])
        box = np.asarray(source_npz["box"]) if "box" in source_npz else np.asarray(np.nan)
        types = np.asarray(source_npz["types"]) if "types" in source_npz else np.asarray([], dtype=int)

    if positions.ndim != 3:
        raise ValueError(f"positions must be rank-3, got shape {positions.shape}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target_path,
        positions=positions,
        box=box,
        types=types,
        system_id=np.asarray(system_id),
        temperature=np.asarray(str(temperature)),
        source_path=np.asarray(source_path),
        first_npz_member=np.asarray(first_npz_member),
        npz_member_md5=np.asarray(npz_md5),
    )

    frame_count, particle_count, spatial_dimension = positions.shape
    try:
        particle_cache_path = str(target_path.relative_to(ROOT))
    except ValueError:
        particle_cache_path = str(target_path)
    return {
        "system_id": system_id,
        "temperature": str(temperature),
        "source_path": source_path,
        "first_npz_member": first_npz_member,
        "compressed_probe_range_start": float(compressed_probe_range_start),
        "compressed_probe_range_end": float(compressed_probe_range_end),
        "compressed_probe_bytes": float(len(xz_bytes)),
        "npz_member_bytes": float(len(npz_bytes)),
        "npz_member_md5": npz_md5,
        "probe_encoding": probe_encoding,
        "particle_cache_path": particle_cache_path,
        "particle_cache_bytes": float(target_path.stat().st_size),
        "positions_shape": _shape_text(tuple(int(value) for value in positions.shape)),
        "positions_dtype": str(positions.dtype),
        "frame_count": float(frame_count),
        "particle_count": float(particle_count),
        "spatial_dimension": float(spatial_dimension),
        "particle_resolved_positions_cached": 1.0,
        "physical_time_semantics_ready": 0.0,
        "threshold_sweep_event_clock_ready": 0.0,
        "thermodynamic_claim_allowed": 0.0,
        "primary_blocker": "physical_time_semantics",
        "cache_stage": "particle_coordinate_cache_written",
    }


def build_real_multilag_particle_caches(
    *,
    target_csv: Path = DATA_DIR / "renewal_cage_sota_glassbench_multilag_particle_cache_targets.csv",
    member_index_manifest_path: Path = DATA_DIR
    / "third_party"
    / "glassbench"
    / "trajectory_npz_member_index_10118191.json",
    output_manifest_path: Path = DATA_DIR / "renewal_cage_sota_glassbench_multilag_particle_cache_manifest.csv",
    output_root: Path = ROOT,
    cache_root: str = "data/third_party/glassbench/particle_cache",
    range_fetcher: Callable[[str, int, int], bytes] | None = None,
) -> list[dict[str, float | str]]:
    """Extract visible structure-matched multi-lag target NPZ caches from bounded prefixes."""

    with target_csv.open() as f:
        target_rows = list(csv.DictReader(f))
    member_index_manifest = json.loads(member_index_manifest_path.read_text(encoding="utf-8"))
    archive_url = str(member_index_manifest["archive_url"])
    if range_fetcher is None:
        range_fetcher = fetch_range_bytes
    index_by_path = {
        str(entry.get("path", "none")): entry
        for entry in member_index_manifest.get("entries", [])
        if isinstance(entry, dict)
    }
    fetched_payloads: dict[str, tuple[bytes, int, int]] = {}
    rows: list[dict[str, float | str]] = []

    for target_row in target_rows:
        system_id = target_row["system_id"]
        temperature = target_row["temperature"]
        source_path = target_row["source_path"]
        structure_id = target_row["selected_structure_id"]
        target_members = [item for item in target_row["target_members"].split(";") if item]
        target_md5s = [item for item in target_row["target_member_md5s"].split(";") if item]
        target_lags = [item for item in target_row["target_lag_times"].split(";") if item]
        target_codes = [item for item in target_row["selected_time_codes"].split(";") if item]
        index = index_by_path.get(source_path, {})
        indexed_members = {
            str(member.get("name", "none"))
            for member in index.get("npz_members", [])
            if isinstance(member, dict)
        }
        start = int(index.get("compressed_probe_range_start", 0) or 0)
        end = int(index.get("compressed_probe_range_end", -1) or -1)

        for member_name, expected_md5, lag_time, time_code in zip(
            target_members, target_md5s, target_lags, target_codes
        ):
            member_in_prefix = member_name in indexed_members
            base = {
                "system_id": system_id,
                "temperature": temperature,
                "source_path": source_path,
                "structure_id": structure_id,
                "time_code": time_code,
                "lag_time": float(lag_time),
                "target_member": member_name,
                "target_member_md5": expected_md5,
                "member_in_bounded_prefix_index": float(member_in_prefix),
                "compressed_probe_range_start": float(start),
                "compressed_probe_range_end": float(end),
                "probe_encoding": "zip_deflate_xz",
                "thermodynamic_claim_allowed": 0.0,
            }
            if not member_in_prefix:
                rows.append(
                    {
                        **base,
                        "particle_cache_path": "none",
                        "particle_cache_bytes": 0.0,
                        "positions_shape": "none",
                        "positions_dtype": "none",
                        "frame_count": 0.0,
                        "particle_count": 0.0,
                        "spatial_dimension": 0.0,
                        "particle_resolved_positions_cached": 0.0,
                        "primary_blocker": "member_not_in_bounded_prefix_index",
                        "cache_stage": "multi_lag_target_outside_bounded_prefix",
                    }
                )
                continue

            if source_path not in fetched_payloads:
                fetched_payloads[source_path] = (range_fetcher(archive_url, start, end), start, end)
            xz_bytes, range_start, range_end = fetched_payloads[source_path]
            target_path = output_root / cache_root / _cache_file_name(system_id, temperature, member_name)
            manifest = extract_first_npz_positions_cache_from_xz_bytes(
                xz_bytes=xz_bytes,
                first_npz_member=member_name,
                expected_npz_md5=expected_md5,
                target_path=target_path,
                system_id=system_id,
                temperature=temperature,
                source_path=source_path,
                compressed_probe_range_start=range_start,
                compressed_probe_range_end=range_end,
                probe_encoding="zip_deflate_xz",
            )
            rows.append(
                {
                    **base,
                    "particle_cache_path": manifest["particle_cache_path"],
                    "particle_cache_bytes": manifest["particle_cache_bytes"],
                    "positions_shape": manifest["positions_shape"],
                    "positions_dtype": manifest["positions_dtype"],
                    "frame_count": manifest["frame_count"],
                    "particle_count": manifest["particle_count"],
                    "spatial_dimension": manifest["spatial_dimension"],
                    "particle_resolved_positions_cached": 1.0,
                    "primary_blocker": "none",
                    "cache_stage": "multi_lag_particle_coordinate_cache_written",
                }
            )

    write_cache_manifest(rows, output_manifest_path)
    return rows


def fetch_range_bytes(url: str, start: int, end: int) -> bytes:
    """Fetch inclusive HTTP byte range."""

    request = Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=120) as response:
        data = response.read()
    expected = end - start + 1
    if len(data) != expected:
        raise ValueError(f"range fetch returned {len(data)} bytes, expected {expected}")
    return data


def write_cache_manifest(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_real_first_npz_particle_caches(
    *,
    contract_csv: Path = DATA_DIR / "renewal_cage_sota_glassbench_first_npz_particle_cache_contract.csv",
    curve_manifest_path: Path = DATA_DIR
    / "third_party"
    / "glassbench"
    / "trajectory_first_npz_observable_curve_10118191.json",
    output_manifest_path: Path = DATA_DIR / "renewal_cage_sota_glassbench_first_npz_particle_cache_manifest.csv",
) -> list[dict[str, float | str]]:
    curve_manifest = json.loads(curve_manifest_path.read_text(encoding="utf-8"))
    archive_url = str(curve_manifest["archive_url"])
    rows: list[dict[str, float | str]] = []
    with contract_csv.open() as f:
        for row in csv.DictReader(f):
            start = int(float(row["compressed_probe_range_start"]))
            end = int(float(row["compressed_probe_range_end"]))
            xz_bytes = fetch_range_bytes(archive_url, start, end)
            rows.append(
                extract_first_npz_positions_cache_from_xz_bytes(
                    xz_bytes=xz_bytes,
                    first_npz_member=row["first_npz_member"],
                    expected_npz_md5=row["npz_member_md5"],
                    target_path=ROOT / row["particle_cache_target"],
                    system_id=row["system_id"],
                    temperature=row["temperature"],
                    source_path=row["source_path"],
                    compressed_probe_range_start=start,
                    compressed_probe_range_end=end,
                    probe_encoding="zip_deflate_xz",
                )
            )
    write_cache_manifest(rows, output_manifest_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DATA_DIR / "renewal_cage_sota_glassbench_first_npz_particle_cache_manifest.csv",
    )
    args = parser.parse_args()
    rows = build_real_first_npz_particle_caches(output_manifest_path=args.output_manifest)
    print(f"wrote {len(rows)} particle caches to {args.output_manifest}")


if __name__ == "__main__":
    main()
