#!/usr/bin/env python3
"""Build the pinned serial LAMMPS source with the LEPTON pair package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_directory.resolve()
    build = args.build_directory.resolve()
    manifest = args.manifest.resolve()
    if not (source / "cmake" / "CMakeLists.txt").is_file():
        raise ValueError("source directory must contain the LAMMPS CMake project")
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    build.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    configure = [
        "cmake",
        "-S",
        str(source / "cmake"),
        "-B",
        str(build),
        "-D",
        "CMAKE_BUILD_TYPE=Release",
        "-D",
        "BUILD_MPI=OFF",
        "-D",
        "PKG_LEPTON=ON",
    ]
    compile_command = ["cmake", "--build", str(build), "--target", "lmp", "-j", str(args.jobs)]
    subprocess.run(configure, check=True)
    subprocess.run(compile_command, check=True)
    binary = build / "lmp"
    if not binary.is_file():
        raise ValueError("LAMMPS build did not produce lmp")
    help_text = subprocess.run(
        [str(binary), "-h"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "lepton" not in help_text.split():
        raise ValueError("built LAMMPS binary does not list the lepton pair style")

    try:
        source_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        source_revision = "unavailable"
    payload = {
        "protocol": "serial_lammps_lepton_build",
        "source_directory": str(source),
        "source_revision": source_revision,
        "build_directory": str(build),
        "binary_path": str(binary),
        "binary_sha256": file_sha256(binary),
        "cmake_cache_sha256": file_sha256(build / "CMakeCache.txt"),
        "configure_command": configure,
        "compile_command": compile_command,
        "lepton_pair_style_available": True,
        "thermodynamic_claim_allowed": False,
    }
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest)


if __name__ == "__main__":
    main()
