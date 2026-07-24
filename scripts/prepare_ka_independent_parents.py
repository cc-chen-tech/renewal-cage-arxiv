#!/usr/bin/env python3
"""Render the frozen T=0.45 independent-parent LAMMPS inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ka_parent_acquisition import prepare_parent_acquisition  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository-commit")
    args = parser.parse_args()
    commit = args.repository_commit or subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rows = prepare_parent_acquisition(
        args.manifest,
        args.output,
        repository_commit=commit,
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
