#!/usr/bin/env python3
"""Stop-gate the two frozen nonlinear-bath canary caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_nonlinear_bath_elimination import (  # noqa: E402
    canary_preflight,
    load_complete_cache,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--canary-cache", type=Path, required=True)
    parser.add_argument("--half-step-cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    canary = load_complete_cache(args.canary_cache, expected_mode="canary")
    half_step = load_complete_cache(
        args.half_step_cache,
        expected_mode="canary-half-step",
    )
    result = canary_preflight(canary, half_step)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_name(args.output_json.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output_json)
    print(result)
    if float(result["canary_preflight_pass"]) != 1.0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
