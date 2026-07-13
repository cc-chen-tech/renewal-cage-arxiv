#!/usr/bin/env python3
"""Prepare a decorrelation-gated ensemble of KA-LJ trajectory restarts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import prepare_replicate_ensemble  # noqa: E402


def parse_integers(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--frame-indices", type=parse_integers, required=True)
    parser.add_argument("--velocity-seeds", type=parse_integers, required=True)
    parser.add_argument("--wave-number", type=float, default=7.25)
    parser.add_argument("--maximum-absolute-fs", type=float, default=0.1)
    parser.add_argument("--equilibration-time", type=float, default=100.0)
    parser.add_argument("--production-time", type=float, default=5000.0)
    parser.add_argument("--dynamics", choices=("nvt", "langevin"), default="nvt")
    parser.add_argument("--langevin-damping", type=float)
    parser.add_argument("--langevin-seeds", type=parse_integers)
    parser.add_argument("--dump-interval-time", type=float, default=1.0)
    parser.add_argument("--high-resolution-duration", type=float, default=0.0)
    parser.add_argument("--high-resolution-dump-interval", type=float)
    args = parser.parse_args()

    manifest = prepare_replicate_ensemble(
        args.source,
        args.output,
        temperature=args.temperature,
        frame_indices=args.frame_indices,
        velocity_seeds=args.velocity_seeds,
        wave_number=args.wave_number,
        maximum_absolute_fs=args.maximum_absolute_fs,
        equilibration_time=args.equilibration_time,
        production_time=args.production_time,
        dynamics=args.dynamics,
        langevin_damping=args.langevin_damping,
        langevin_seeds=args.langevin_seeds,
        dump_interval_time=args.dump_interval_time,
        high_resolution_duration=args.high_resolution_duration,
        high_resolution_dump_interval=args.high_resolution_dump_interval,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
