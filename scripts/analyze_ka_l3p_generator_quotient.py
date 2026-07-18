#!/usr/bin/env python3
"""Test the held microscopic ``L^3p`` generator-coordinate quotient."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_l2p_conditional_diffusion import (  # noqa: E402
    load_conditional_diffusion_caches,
    output_path,
    write_rows,
)
from analyze_ka_relative_second_generator_closure import (  # noqa: E402
    load_matched_second_generator_clones,
)
from ka_l2p_conditional_diffusion import (  # noqa: E402
    conditional_covariance_diagnostic,
    fit_scaled_conditional_covariance,
)
from ka_l3p_generator import classify_l3p_numerical_canary  # noqa: E402
from ka_l3p_quotient import (  # noqa: E402
    L3P_QUOTIENT_MODELS,
    augment_l3p_quotient_clone,
    classify_l3p_quotient,
    extract_l3p_quotient_fold,
)


_NUMERICAL_ARRAYS = (
    "prefix_16_32_error",
    "position_primary_reference_error",
    "position_coarse_reference_error",
    "cage_primary_reference_error",
    "cage_coarse_reference_error",
    "acceleration_directional_error",
)


def _scalar(cache: np.lib.npyio.NpzFile, key: str) -> object:
    value = np.asarray(cache[key])
    if value.shape != ():
        raise ValueError(f"L3p metadata must be scalar: {key}")
    return value.item()


def load_l3p_caches(
    directory: Path,
    clones: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Load only complete, numerically resolved L3p caches by trajectory hash."""

    paths = sorted(Path(directory).glob("clone_*_l3p_generator.npz"))
    if not paths:
        raise ValueError("microscopic L3p caches are required")
    by_hash: dict[str, dict[str, object]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as cache:
            required = {
                "trajectory_sha256",
                "target_indices",
                "potential_protocol",
                "estimator",
                "l3p",
                "completed_frame_count",
                "requested_frame_count",
                "l3p_numerical_gate_pass",
                "numerical_state",
                "finite_l3p_gaussian_closure_supported",
                "thermodynamic_claim_allowed",
                *_NUMERICAL_ARRAYS,
            }
            if not required.issubset(cache.files):
                raise ValueError(f"incomplete L3p cache: {path}")
            source_hash = str(_scalar(cache, "trajectory_sha256"))
            if source_hash in by_hash:
                raise ValueError("duplicate microscopic L3p trajectory hash")
            targets = np.asarray(cache["target_indices"], dtype=int)
            values = np.asarray(cache["l3p"], dtype=float)
            completed = int(float(_scalar(cache, "completed_frame_count")))
            requested = int(float(_scalar(cache, "requested_frame_count")))
            protocol = str(_scalar(cache, "potential_protocol"))
            estimator = str(_scalar(cache, "estimator"))
            saved_gate = float(_scalar(cache, "l3p_numerical_gate_pass"))
            saved_state = str(_scalar(cache, "numerical_state"))
            numerical_arrays = {
                key: np.asarray(cache[key], dtype=float) for key in _NUMERICAL_ARRAYS
            }
            recomputed = classify_l3p_numerical_canary(**numerical_arrays)
            if (
                completed != requested
                or completed < 1
                or values.shape != (completed, len(targets), 3)
                or np.any(~np.isfinite(values))
                or estimator != "microscopic_l3p_generator_quotient"
                or protocol not in {"ka_lj_cut", "ka_lj_c3_switch"}
                or float(_scalar(cache, "finite_l3p_gaussian_closure_supported"))
                != 0.0
                or float(_scalar(cache, "thermodynamic_claim_allowed")) != 0.0
                or saved_gate != 1.0
                or saved_state != "l3p_generator_numerically_resolved"
                or float(recomputed["l3p_numerical_gate_pass"]) != saved_gate
                or str(recomputed["numerical_state"]) != saved_state
            ):
                raise ValueError(f"microscopic L3p numerical gate did not pass: {path}")
            by_hash[source_hash] = {
                "l3p_generator": values.copy(),
                "target_indices": targets.copy(),
                "trajectory_sha256": source_hash,
                "potential_protocol": protocol,
                "path": str(path.resolve()),
                "numerical_verdict": recomputed,
                **{key: value.copy() for key, value in numerical_arrays.items()},
                "thermodynamic_claim_allowed": 0.0,
            }
    matched = []
    for clone in clones:
        source_hash = str(clone["trajectory_sha256"])
        if source_hash not in by_hash:
            raise ValueError("L3p cache is missing a matched clone")
        loaded = by_hash[source_hash]
        if (
            not np.array_equal(loaded["target_indices"], clone["target_indices"])
            or str(loaded["potential_protocol"]) != str(clone["potential_protocol"])
            or len(np.asarray(loaded["l3p_generator"]))
            < len(np.asarray(clone["relative_position"]))
        ):
            raise ValueError("L3p cache does not align with the resolved clone")
        matched.append(loaded)
    if len(by_hash) != len(clones):
        raise ValueError("L3p cache directory contains unmatched clones")
    return matched


def _numerical_row(fold: int, cache: dict[str, object]) -> dict[str, object]:
    verdict = dict(cache["numerical_verdict"])
    row: dict[str, object] = {
        "record": "held_fold_numerical",
        "fold_index": float(fold),
        "trajectory_sha256": str(cache["trajectory_sha256"]),
        "potential_protocol": str(cache["potential_protocol"]),
        **verdict,
    }
    return row


def _model_absolute_gate(row: dict[str, object]) -> bool:
    return (
        float(row["maximum_absolute_whitened_correlation"]) <= 0.05
        and float(row["maximum_absolute_squared_whitened_correlation"]) <= 0.05
        and float(row["maximum_absolute_component_excess_kurtosis"]) <= 0.35
        and float(row["maximum_absolute_whitened_covariance_error"]) <= 0.10
        and 0.8
        <= float(row["mean_squared_mahalanobis_per_dimension"])
        <= 1.2
        and float(row["isotropic_floor_variance_fraction"]) <= 0.25
    )


def write_diagnostic_svg(path: Path, rows: list[dict[str, object]]) -> None:
    """Write an unclipped paired NLL/memory diagnostic for all four folds."""

    by_cell = {
        (int(float(row["fold_index"])), str(row["model"])): row for row in rows
    }
    expected = {
        (fold, model)
        for fold in range(1, 5)
        for model in L3P_QUOTIENT_MODELS
    }
    if set(by_cell) != expected:
        raise ValueError("L3p diagnostic grid is incomplete")
    colors = {
        "l2p_exact_q_baseline": "#50555c",
        "l3p_generator": "#166f5f",
        "l3p_time_permuted": "#b5483f",
        "l2p_backward_difference": "#7856a6",
    }
    nll = {
        cell: float(row["negative_log_likelihood"])
        for cell, row in by_cell.items()
    }
    memory = {
        cell: float(row["maximum_absolute_squared_whitened_correlation"])
        for cell, row in by_cell.items()
    }
    if any(not math.isfinite(value) for value in (*nll.values(), *memory.values())):
        raise ValueError("L3p diagnostic values must be finite")

    def bounds(values: list[float], reference: float | None) -> tuple[float, float]:
        candidates = values if reference is None else [*values, reference]
        low, high = min(candidates), max(candidates)
        span = high - low or max(abs(high), 1.0)
        return low - 0.08 * span, high + 0.08 * span

    width, height = 1200, 650
    top, bottom, panel_width = 130.0, 530.0, 470.0
    panels = (
        (75.0, nll, bounds(list(nll.values()), None), "held negative log-likelihood", None),
        (
            650.0,
            memory,
            bounds(list(memory.values()), 0.05),
            "maximum absolute squared whitened correlation",
            0.05,
        ),
    )

    def x(panel: float, fold: int) -> float:
        return panel + 55.0 + (fold - 1) * (panel_width - 110.0) / 3.0

    def y(value: float, limits: tuple[float, float]) -> float:
        return bottom - (value - limits[0]) * (bottom - top) / (limits[1] - limits[0])

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="75" y="38" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#20242a">Microscopic L3p generator-coordinate quotient</text>',
        '<text x="75" y="64" font-family="Arial, sans-serif" font-size="13" fill="#50555c">Four held clones; exact Qc whitening; unclipped axes</text>',
    ]
    for slot, model in enumerate(L3P_QUOTIENT_MODELS):
        legend_x = 80 + 275 * slot
        svg.extend(
            [
                f'<line x1="{legend_x}" y1="92" x2="{legend_x + 24}" y2="92" stroke="{colors[model]}" stroke-width="3"/>',
                f'<text x="{legend_x + 31}" y="97" font-family="Arial, sans-serif" font-size="12" fill="#30353b">{model}</text>',
            ]
        )
    for panel_x, values, limits, title, reference in panels:
        svg.append(
            f'<rect x="{panel_x}" y="{top}" width="{panel_width}" height="{bottom - top}" fill="#fbfcfd" stroke="#c9cdd2"/>'
        )
        for tick in range(5):
            value = limits[0] + tick * (limits[1] - limits[0]) / 4.0
            yy = y(value, limits)
            svg.extend(
                [
                    f'<line x1="{panel_x}" y1="{yy:.3f}" x2="{panel_x + panel_width}" y2="{yy:.3f}" stroke="#e3e6e9"/>',
                    f'<text x="{panel_x - 8}" y="{yy + 4:.3f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#555b62">{value:.3g}</text>',
                ]
            )
        if reference is not None:
            yy = y(reference, limits)
            svg.append(
                f'<line x1="{panel_x}" y1="{yy:.3f}" x2="{panel_x + panel_width}" y2="{yy:.3f}" stroke="#15191e" stroke-dasharray="7 5"/>'
            )
        for model in L3P_QUOTIENT_MODELS:
            points = " ".join(
                f'{x(panel_x, fold):.3f},{y(values[(fold, model)], limits):.3f}'
                for fold in range(1, 5)
            )
            svg.append(
                f'<polyline points="{points}" fill="none" stroke="{colors[model]}" stroke-width="2.4"/>'
            )
        svg.append(
            f'<text x="{panel_x}" y="580" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#24282e">{title}</text>'
        )
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--drift-cache-directory", type=Path, required=True)
    parser.add_argument(
        "--second-generator-cache-directories",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--conditional-diffusion-cache-directory",
        type=Path,
        required=True,
    )
    parser.add_argument("--l3p-cache-directory", type=Path, required=True)
    parser.add_argument("--memory-order", type=int, default=40, help="frozen Mori order 40")
    parser.add_argument("--bath-order", type=int, default=16, help="frozen VAR order 16")
    parser.add_argument("--maximum-lag", type=int, default=40)
    parser.add_argument("--maximum-frame-count", type=int, default=200)
    parser.add_argument("--frame-time", type=float, default=0.01)
    parser.add_argument("--ridge-regularization", type=float, default=1e-8)
    parser.add_argument("--var-ridge-regularization", type=float, default=1e-6)
    parser.add_argument(
        "--time-permutation-seed",
        type=int,
        default=20260802,
        help="frozen L3p time-null base seed 20260802",
    )
    parser.add_argument("--gaussian-reference-seed", type=int, default=20260803)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--figure-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.memory_order != 40
        or args.bath_order != 16
        or args.maximum_lag != 40
        or args.maximum_frame_count != 200
        or not math.isclose(args.frame_time, 0.01, rel_tol=0.0, abs_tol=0.0)
        or args.time_permutation_seed != 20260802
        or args.ridge_regularization < 0.0
        or args.var_ridge_regularization < 0.0
    ):
        raise ValueError("L3p quotient controls must match the frozen protocol")
    clones = load_matched_second_generator_clones(
        args.drift_cache_directory,
        args.second_generator_cache_directories,
        maximum_frame_count=args.maximum_frame_count,
    )
    if len(clones) != 4:
        raise ValueError("the frozen L3p quotient requires four clones")
    q_caches = load_conditional_diffusion_caches(
        args.conditional_diffusion_cache_directory,
        clones,
    )
    if any(
        str(cache["numerical_estimator"]) != "deterministic_velocity_jacobian"
        or float(cache["numerical_gate_pass"]) != 1.0
        for cache in q_caches
    ):
        raise ValueError("L3p quotient requires resolved exact deterministic Qc")
    l3p_caches = load_l3p_caches(args.l3p_cache_directory, clones)
    merged = [
        {**clone, "l3p_generator": np.asarray(cache["l3p_generator"])}
        for clone, cache in zip(clones, l3p_caches, strict=True)
    ]
    numerical = [
        _numerical_row(fold, cache)
        for fold, cache in enumerate(l3p_caches, start=1)
    ]
    details: list[dict[str, object]] = []
    for model in L3P_QUOTIENT_MODELS:
        model_clones = [
            augment_l3p_quotient_clone(
                clone,
                model=model,
                frame_time=args.frame_time,
                permutation_seed=args.time_permutation_seed + index,
            )
            for index, clone in enumerate(merged)
        ]
        for fold_index in range(4):
            training_indices = [index for index in range(4) if index != fold_index]
            training = [model_clones[index] for index in training_indices]
            held = model_clones[fold_index]
            extracted = extract_l3p_quotient_fold(
                training,
                held,
                memory_order=args.memory_order,
                bath_order=args.bath_order,
                ridge_regularization=args.ridge_regularization,
                var_ridge_regularization=args.var_ridge_regularization,
            )
            frame_indices = np.asarray(
                extracted["held_source_frame_indices"],
                dtype=int,
            )
            normalization_scale = float(
                np.asarray(extracted["normalization_scale"])[0, 0, 3]
            )
            training_residual = np.asarray(
                extracted["training_white_innovation"]
            )[:, :, 3].reshape(len(frame_indices), -1, 3)
            held_residual = np.asarray(extracted["held_l2p_vector_innovation"])
            training_q = np.concatenate(
                [
                    np.asarray(q_caches[index]["conditional_diffusion"])[
                        frame_indices
                    ]
                    for index in training_indices
                ],
                axis=1,
            )
            held_q = np.asarray(
                q_caches[fold_index]["conditional_diffusion"]
            )[frame_indices]
            training_q = args.frame_time * training_q / normalization_scale**2
            held_q = args.frame_time * held_q / normalization_scale**2
            if (
                training_q.shape[:2] != training_residual.shape[:2]
                or held_q.shape[:2] != held_residual.shape[:2]
            ):
                raise ValueError("L3p quotient Qc is not white-innovation aligned")
            fit = fit_scaled_conditional_covariance(training_residual, training_q)
            held_covariance = (
                fit["scale"] * held_q
                + fit["isotropic_floor"] * np.eye(3)
            )
            diagnostic = conditional_covariance_diagnostic(
                held_residual,
                held_covariance,
                maximum_lag=args.maximum_lag,
                gaussian_seed=args.gaussian_reference_seed + fold_index + 1,
            )
            details.append(
                {
                    "record": "held_fold_model",
                    "fold_index": float(fold_index + 1),
                    "model": model,
                    **diagnostic,
                    "fitted_scale": float(fit["scale"]),
                    "fitted_isotropic_floor": float(fit["isotropic_floor"]),
                    "isotropic_floor_variance_fraction": float(
                        fit["isotropic_floor_variance_fraction"]
                    ),
                    "fit_uses_held_samples": 0.0,
                    "first_source_frame_index": float(frame_indices[0]),
                    "last_source_frame_index": float(frame_indices[-1]),
                    "trajectory_sha256": str(held["trajectory_sha256"]),
                    "time_permutation_base_seed": float(
                        args.time_permutation_seed
                    ),
                    "finite_l3p_gaussian_closure_supported": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )

    summaries: list[dict[str, object]] = []
    metrics = (
        "negative_log_likelihood",
        "mean_squared_mahalanobis_per_dimension",
        "maximum_absolute_whitened_covariance_error",
        "maximum_absolute_component_excess_kurtosis",
        "maximum_absolute_whitened_correlation",
        "maximum_absolute_squared_whitened_correlation",
        "gaussian_energy_distance",
        "isotropic_floor_variance_fraction",
    )
    for model in L3P_QUOTIENT_MODELS:
        selected = [row for row in details if row["model"] == model]
        row: dict[str, object] = {
            "record": "model_aggregate",
            "model": model,
            "held_clone_count": 4.0,
            "every_fold_absolute_gate_pass": float(
                all(_model_absolute_gate(value) for value in selected)
            ),
            "finite_l3p_gaussian_closure_supported": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for metric in metrics:
            values = np.asarray([float(value[metric]) for value in selected])
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"maximum_{metric}"] = float(np.max(values))
        summaries.append(row)
    summaries.append(classify_l3p_quotient(details, numerical))
    write_rows(output_path(args.output_prefix, "_details.csv"), details)
    write_rows(output_path(args.output_prefix, "_numerical.csv"), numerical)
    write_rows(output_path(args.output_prefix, "_summary.csv"), summaries)
    if args.figure_path is not None:
        write_diagnostic_svg(args.figure_path, details)


if __name__ == "__main__":
    main()
