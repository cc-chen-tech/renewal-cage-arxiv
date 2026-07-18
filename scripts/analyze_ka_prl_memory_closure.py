#!/usr/bin/env python3
"""Recompute the preregistered, parent-first PRL memory-closure gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_segment_splice_gate import load_frozen_blocks  # noqa: E402
from ka_prl_memory_closure import (  # noqa: E402
    ABLATION_MODELS,
    FROZEN_PROTOCOLS,
    audit_parent_provenance,
    build_claim_ledger,
    classify_memory_closure_gate,
    generate_ablation_path,
    summarize_model_verdicts,
    summarize_parents,
    summarize_restarts,
)
from ka_segment_splice import cumulative_observables_many_lags  # noqa: E402


BLOCK_SIZE = 20
WAVE_NUMBERS = np.asarray((2.0, 4.0, 7.25))
BASE_SEED = 20260718
SPECTRAL_MODEL = "two_point_path_spectrum"
UPPER_CONTROL = "contiguous_empirical_upper_control"


def read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"cannot read CSV input: {path}") from error
    if not rows:
        raise ValueError(f"CSV input is empty: {path}")
    return rows


def _canonical_value(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("artifact floats must be finite")
        return str(int(number)) if number.is_integer() else repr(number)
    return str(value)


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty artifact: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("artifact table rows must have one rectangular schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _canonical_value(row[key]) for key in fields})


def _stationarity_by_temperature(
    low_path: Path, high_path: Path
) -> dict[float, list[dict[str, str]]]:
    return {0.45: read_rows(low_path), 0.58: read_rows(high_path)}


def _audit(
    *, provenance: Path, low_stationarity: Path, high_stationarity: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return audit_parent_provenance(
        provenance_rows=read_rows(provenance),
        stationarity_by_temperature=_stationarity_by_temperature(
            low_stationarity, high_stationarity
        ),
    )


def _model_seed(
    *, temperature: float, restart: int, model: str, realization: int
) -> int:
    payload = (
        f"{BASE_SEED}|{temperature:g}|{restart}|{model}|{realization}"
    ).encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (
        2**63 - 1
    )


def _frozen_lags(temperature: float) -> tuple[int, ...]:
    return tuple(
        int(value) for value in str(FROZEN_PROTOCOLS[temperature]["lag_grid"]).split(";")
    )


def _heldout_targets(
    rows: Sequence[Mapping[str, object]], *, temperature: float
) -> dict[tuple[int, int], dict[str, float]]:
    frozen_lags = set(_frozen_lags(temperature))
    targets: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        if not math.isclose(float(row["temperature"]), temperature, abs_tol=1e-12):
            continue
        restart = int(float(row["replicate"]))
        lag = int(float(row["lag"]))
        if lag not in frozen_lags:
            continue
        target = {
            "msd": float(row["observed_msd"]),
            "ngp": float(row["observed_ngp"]),
            "fs_k2": float(row["observed_fs_k2"]),
            "fs_k4": float(row["observed_fs_k4"]),
            "fs_k7p25": float(row["observed_fs_k7p25"]),
        }
        if any(not math.isfinite(value) for value in target.values()) or target["msd"] <= 0:
            raise ValueError("held-out target observables must be finite with positive MSD")
        key = (restart, lag)
        if key in targets:
            raise ValueError("held-out targets must be unique by restart and lag")
        targets[key] = target
    expected_restarts = {1, 2, 3} if temperature == 0.45 else {1, 2, 3, 4, 5}
    expected = {
        (restart, lag) for restart in expected_restarts for lag in frozen_lags
    }
    if set(targets) != expected:
        raise ValueError("held-out targets do not cover the frozen restart-lag grid")
    return targets


def _environment_times(
    rows: Sequence[Mapping[str, object]], *, temperature: float
) -> dict[int, float]:
    group = "low" if temperature == 0.45 else "high"
    result: dict[int, float] = {}
    for row in rows:
        if str(row["temperature_group"]) != group or not math.isclose(
            float(row["block_size"]), BLOCK_SIZE, abs_tol=1e-12
        ):
            continue
        restart = int(float(row["replicate"]))
        value = float(row["efold_crossing_time"])
        if not math.isfinite(value) or value <= 0.0 or restart in result:
            raise ValueError("environment crossing rows must be unique and positive")
        result[restart] = value
    expected = {1, 2, 3} if temperature == 0.45 else {1, 2, 3, 4, 5}
    if set(result) != expected:
        raise ValueError("environment crossing table misses a frozen restart")
    return result


def _prediction_rows_for_path(
    blocks: np.ndarray,
    *,
    temperature: float,
    restart: int,
    model: str,
    realization: int,
    targets: Mapping[tuple[int, int], Mapping[str, float]],
    information: Mapping[str, object],
) -> list[dict[str, object]]:
    lags = _frozen_lags(temperature)
    block_counts = tuple(lag // BLOCK_SIZE for lag in lags)
    observables = cumulative_observables_many_lags(
        blocks,
        block_counts=block_counts,
        wave_numbers=WAVE_NUMBERS,
    )
    rows: list[dict[str, object]] = []
    for lag, block_count in zip(lags, block_counts, strict=True):
        prediction = observables[block_count]
        target = targets[(restart, lag)]
        rows.append(
            {
                "temperature": temperature,
                "restart": restart,
                "model": model,
                "realization": realization,
                "lag": lag,
                "block_size": BLOCK_SIZE,
                "predicted_msd": prediction["msd"],
                "predicted_ngp": prediction["ngp"],
                "predicted_fs_k2": prediction["characteristic_k2"],
                "predicted_fs_k4": prediction["characteristic_k4"],
                "predicted_fs_k7p25": prediction["characteristic_k7p25"],
                "target_msd": target["msd"],
                "target_ngp": target["ngp"],
                "target_fs_k2": target["fs_k2"],
                "target_fs_k4": target["fs_k4"],
                "target_fs_k7p25": target["fs_k7p25"],
                "support_pass": 1.0,
                "heldout_path_used_in_prediction": 0.0,
                "heldout_observables_used_as_model_inputs": 0.0,
                "calibration_budget_equal_to_nulls": 1.0,
                "one_step_jump_law_retained": float(
                    information.get("one_step_jump_law_retained", 1.0)
                ),
                "two_point_path_spectrum_retained": float(
                    information.get("two_point_path_spectrum_retained", 0.0)
                ),
                "particle_identity_retained": float(
                    information.get("particle_identity_retained", 0.0)
                ),
                "static_particle_environment_retained": float(
                    information.get("static_particle_environment_retained", 0.0)
                ),
                "finite_exchange_environment_retained": float(
                    information.get("finite_exchange_environment_retained", 0.0)
                ),
                "ordered_path_memory_retained": float(
                    information.get("ordered_path_memory_retained", 0.0)
                ),
                "microdynamic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    return rows


def _spectral_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    temperature: float,
    targets: Mapping[tuple[int, int], Mapping[str, float]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        if str(row["model"]) != "radial_multivariate_surrogate" or not math.isclose(
            float(row["temperature"]), temperature, abs_tol=1e-12
        ):
            continue
        restart = int(float(row["replicate"]))
        lag = int(float(row["lag"]))
        if lag not in _frozen_lags(temperature):
            continue
        target = targets[(restart, lag)]
        for observable, source in (
            ("msd", "observed_msd"),
            ("ngp", "observed_ngp"),
            ("fs_k2", "observed_fs_k2"),
            ("fs_k4", "observed_fs_k4"),
            ("fs_k7p25", "observed_fs_k7p25"),
        ):
            if not math.isclose(
                float(row[source]), target[observable], rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("spectral rows and held-out targets disagree")
        output.append(
            {
                "temperature": temperature,
                "restart": restart,
                "model": SPECTRAL_MODEL,
                "realization": int(float(row["realization"])),
                "lag": lag,
                "block_size": BLOCK_SIZE,
                "predicted_msd": float(row["predicted_msd"]),
                "predicted_ngp": float(row["predicted_ngp"]),
                "predicted_fs_k2": float(row["predicted_fs_k2"]),
                "predicted_fs_k4": float(row["predicted_fs_k4"]),
                "predicted_fs_k7p25": float(row["predicted_fs_k7p25"]),
                "target_msd": target["msd"],
                "target_ngp": target["ngp"],
                "target_fs_k2": target["fs_k2"],
                "target_fs_k4": target["fs_k4"],
                "target_fs_k7p25": target["fs_k7p25"],
                "support_pass": 1.0,
                "heldout_path_used_in_prediction": 0.0,
                "heldout_observables_used_as_model_inputs": 0.0,
                "calibration_budget_equal_to_nulls": 1.0,
                "one_step_jump_law_retained": 1.0,
                "two_point_path_spectrum_retained": 1.0,
                "particle_identity_retained": 0.0,
                "static_particle_environment_retained": 0.0,
                "finite_exchange_environment_retained": 0.0,
                "ordered_path_memory_retained": 0.0,
                "microdynamic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    expected = (
        (3 if temperature == 0.45 else 5) * len(_frozen_lags(temperature)) * 8
    )
    if len(output) != expected:
        raise ValueError("spectral source does not contain the frozen eight-realization grid")
    return sorted(
        output,
        key=lambda row: (int(row["restart"]), int(row["realization"]), int(row["lag"])),
    )


def predict_correlated_parent_diagnostic(
    *,
    blocks_by_restart: Mapping[int, np.ndarray],
    target_rows: Sequence[Mapping[str, object]],
    crossing_rows: Sequence[Mapping[str, object]],
    spectral_source_rows: Sequence[Mapping[str, object]],
    temperature: float,
    realizations: int,
) -> list[dict[str, object]]:
    """Run the frozen model family as a correlated-parent diagnostic only."""

    if realizations not in {16, 64}:
        raise ValueError("realizations must be one of the frozen values 16 or 64")
    targets = _heldout_targets(target_rows, temperature=temperature)
    environment_times = _environment_times(crossing_rows, temperature=temperature)
    expected_restarts = {1, 2, 3} if temperature == 0.45 else {1, 2, 3, 4, 5}
    if set(blocks_by_restart) != expected_restarts:
        raise ValueError("calibration blocks miss a frozen restart")
    rows: list[dict[str, object]] = []
    for restart in sorted(blocks_by_restart):
        blocks = np.asarray(blocks_by_restart[restart], dtype=float)
        rows.extend(
            _prediction_rows_for_path(
                blocks,
                temperature=temperature,
                restart=restart,
                model=UPPER_CONTROL,
                realization=0,
                targets=targets,
                information={
                    "one_step_jump_law_retained": 1.0,
                    "particle_identity_retained": 1.0,
                    "static_particle_environment_retained": 1.0,
                    "ordered_path_memory_retained": 1.0,
                },
            )
        )
        for model in sorted(ABLATION_MODELS):
            for realization in range(realizations):
                generated, audit = generate_ablation_path(
                    blocks,
                    model=model,
                    environment_time=environment_times[restart],
                    block_size=BLOCK_SIZE,
                    rng=np.random.default_rng(
                        _model_seed(
                            temperature=temperature,
                            restart=restart,
                            model=model,
                            realization=realization,
                        )
                    ),
                )
                rows.extend(
                    _prediction_rows_for_path(
                        generated,
                        temperature=temperature,
                        restart=restart,
                        model=model,
                        realization=realization,
                        targets=targets,
                        information=audit,
                    )
                )
    rows.extend(
        _spectral_rows(
            spectral_source_rows,
            temperature=temperature,
            targets=targets,
        )
    )
    return sorted(
        rows,
        key=lambda row: (
            float(row["temperature"]),
            str(row["model"]),
            int(row["restart"]),
            int(row["realization"]),
            int(row["lag"]),
        ),
    )


def _annotate_verdicts(
    verdicts: Sequence[Mapping[str, object]], *, gate_state: str
) -> list[dict[str, object]]:
    return [
        {
            **row,
            "independent_parent_gate_state": gate_state,
            "correlated_restart_diagnostic_only": 1.0,
            "positive_memory_closure_claim_allowed": 0.0,
            "microdynamic_closure_claim_allowed": 0.0,
            "spatial_facilitation_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for row in verdicts
    ]


def write_svg(path: Path, verdicts: Sequence[Mapping[str, object]], gate: Mapping[str, object]) -> None:
    ordered = sorted(
        verdicts,
        key=lambda row: (str(row["model"]), str(row["parent_id"])),
    )
    width = 1120
    height = 150 + 24 * len(ordered)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:#18212b}.title{font:700 22px ui-sans-serif,system-ui}.sub{font:14px ui-sans-serif,system-ui;fill:#4d5966}.pass{fill:#16794b}.fail{fill:#b42318}</style>',
        '<text x="28" y="38" class="title">PRL memory closure — parent-first gate</text>',
        f'<text x="28" y="66" class="sub">gate: {html.escape(str(gate["mechanism_state"]))}</text>',
        '<text x="28" y="90" class="sub">Rows below are correlated-parent diagnostics; restart averages cannot open the claim.</text>',
        '<text x="28" y="122">model</text><text x="520" y="122">parent</text><text x="800" y="122">higher-order max</text><text x="1010" y="122">curve</text>',
    ]
    for index, row in enumerate(ordered):
        y = 148 + 24 * index
        passed = float(row["curve_gate_pass"]) == 1.0
        score = float(row["maximum_higher_order_score"])
        lines.extend(
            [
                f'<text x="28" y="{y}">{html.escape(str(row["model"]))}</text>',
                f'<text x="520" y="{y}">{html.escape(str(row["parent_id"]))}</text>',
                f'<text x="800" y="{y}">{score:.6g}</text>',
                f'<text x="1010" y="{y}" class="{"pass" if passed else "fail"}">{"PASS" if passed else "FAIL"}</text>',
            ]
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen independent-parent PRL memory-closure gate."
    )
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--low-stationarity", type=Path, required=True)
    parser.add_argument("--high-stationarity", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--run-temperature", type=float, choices=(0.45, 0.58))
    parser.add_argument("--ensemble-directory", type=Path)
    parser.add_argument("--heldout-targets", type=Path)
    parser.add_argument("--environment-crossings", type=Path)
    parser.add_argument("--spectral-rows", type=Path)
    parser.add_argument("--realizations", type=int, choices=(16, 64), default=16)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--output-parent-ledger", type=Path, required=True)
    parser.add_argument("--output-blockers", type=Path, required=True)
    parser.add_argument("--output-restart-rows", type=Path)
    parser.add_argument("--output-restart-summary", type=Path)
    parser.add_argument("--output-parent-summary", type=Path)
    parser.add_argument("--output-model-verdicts", type=Path)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--output-claim-ledger", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.base_seed != BASE_SEED:
        raise ValueError("the preregistered base seed is frozen at 20260718")
    parent_ledger, blockers = _audit(
        provenance=args.provenance,
        low_stationarity=args.low_stationarity,
        high_stationarity=args.high_stationarity,
    )
    write_rows(args.output_parent_ledger, parent_ledger)
    write_rows(args.output_blockers, blockers)

    if args.audit_only:
        if any(
            value is not None
            for value in (
                args.run_temperature,
                args.ensemble_directory,
                args.heldout_targets,
                args.environment_crossings,
                args.spectral_rows,
                args.output_restart_rows,
                args.output_restart_summary,
                args.output_parent_summary,
                args.output_model_verdicts,
                args.output_svg,
            )
        ):
            raise ValueError("audit-only mode cannot accept trajectory diagnostic paths")
        gate = classify_memory_closure_gate(
            parent_summaries=[], blockers=blockers, upper_control_parents=[]
        )
        write_rows(args.output_gate, [gate])
        write_rows(args.output_claim_ledger, build_claim_ledger(gate))
        return

    required = {
        "run_temperature": args.run_temperature,
        "ensemble_directory": args.ensemble_directory,
        "heldout_targets": args.heldout_targets,
        "environment_crossings": args.environment_crossings,
        "spectral_rows": args.spectral_rows,
        "output_restart_rows": args.output_restart_rows,
        "output_restart_summary": args.output_restart_summary,
        "output_parent_summary": args.output_parent_summary,
        "output_model_verdicts": args.output_model_verdicts,
        "output_svg": args.output_svg,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"full diagnostic mode is missing: {';'.join(missing)}")
    temperature = float(args.run_temperature)
    blocks_by_restart = load_frozen_blocks(
        args.ensemble_directory,
        temperature=temperature,
        block_size=BLOCK_SIZE,
    )
    realization_rows = predict_correlated_parent_diagnostic(
        blocks_by_restart=blocks_by_restart,
        target_rows=read_rows(args.heldout_targets),
        crossing_rows=read_rows(args.environment_crossings),
        spectral_source_rows=read_rows(args.spectral_rows),
        temperature=temperature,
        realizations=args.realizations,
    )
    restart_summaries = summarize_restarts(realization_rows)
    parent_summaries = summarize_parents(restart_summaries, parent_ledger)
    upper_controls = [
        row for row in parent_summaries if str(row["model"]) == UPPER_CONTROL
    ]
    model_parent_rows = [
        row for row in parent_summaries if str(row["model"]) != UPPER_CONTROL
    ]
    gate = classify_memory_closure_gate(
        parent_summaries=model_parent_rows,
        blockers=blockers,
        upper_control_parents=upper_controls,
    )
    verdicts = summarize_model_verdicts(parent_summaries)
    low_full = [
        row
        for row in verdicts
        if float(row["temperature"]) == 0.45 and str(row["model"]) == "full_candidate"
    ]
    gate.update(
        {
            "run_temperature": temperature,
            "diagnostic_realizations": args.realizations,
            "correlated_parent_diagnostic_full_candidate_pass": float(
                bool(low_full)
                and all(float(row["curve_gate_pass"]) == 1.0 for row in low_full)
            ),
            "correlated_parent_diagnostic_only": 1.0,
            "heldout_observables_used_as_model_inputs": 0.0,
            "gate_or_claim_tuned_after_results": 0.0,
        }
    )
    annotated_verdicts = _annotate_verdicts(
        verdicts, gate_state=str(gate["mechanism_state"])
    )
    write_rows(args.output_restart_rows, realization_rows)
    write_rows(args.output_restart_summary, restart_summaries)
    write_rows(args.output_parent_summary, parent_summaries)
    write_rows(args.output_model_verdicts, annotated_verdicts)
    write_rows(args.output_gate, [gate])
    write_rows(args.output_claim_ledger, build_claim_ledger(gate))
    write_svg(args.output_svg, annotated_verdicts, gate)


if __name__ == "__main__":
    main()
