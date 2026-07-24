#!/usr/bin/env python3
"""Test whether microscopic ``L^2 p`` diffusion closes held Markov-bath innovations."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_relative_second_generator_closure import (  # noqa: E402
    extract_l2p_white_innovation_fold,
    load_matched_second_generator_clones,
)
from ka_l2p_conditional_diffusion import (  # noqa: E402
    conditional_covariance_diagnostic,
    fit_constant_covariance,
    fit_scaled_conditional_covariance,
    replicate_first_t_interval,
)


MODELS = (
    "constant_full",
    "constant_isotropic",
    "trace_only",
    "exact_tensor",
    "permuted_tensor",
)


def output_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def canonical_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return str(float(value))
        return format(float(value), ".12g")
    return value


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical_value(value) for key, value in row.items()})


def write_diagnostic_svg(path: Path, detail_rows: list[dict[str, object]]) -> None:
    """Plot the frozen held-fold NLL and squared-memory diagnostics."""

    plot_models = (
        "constant_full",
        "trace_only",
        "exact_tensor",
        "permuted_tensor",
    )
    expected = {(fold, model) for fold in range(1, 5) for model in plot_models}
    by_cell = {
        (int(float(row["fold_index"])), str(row["model"])): row
        for row in detail_rows
        if str(row["model"]) in plot_models
    }
    if set(by_cell) != expected:
        raise ValueError("four-fold diagnostic SVG grid is not complete")

    nll_improvement = {
        (fold, model): float(by_cell[(fold, "constant_full")]["negative_log_likelihood"])
        - float(by_cell[(fold, model)]["negative_log_likelihood"])
        for fold in range(1, 5)
        for model in plot_models
    }
    squared_memory = {
        cell: float(row["maximum_absolute_squared_whitened_correlation"])
        for cell, row in by_cell.items()
    }
    if any(
        not math.isfinite(value)
        for value in (*nll_improvement.values(), *squared_memory.values())
    ):
        raise ValueError("diagnostic SVG values must be finite")

    def padded_bounds(values: list[float], reference: float) -> tuple[float, float]:
        low = min(*values, reference)
        high = max(*values, reference)
        span = high - low
        if span == 0.0:
            span = max(abs(high), 1.0)
        return low - 0.08 * span, high + 0.08 * span

    width = 1200
    height = 650
    panel_top = 125.0
    panel_bottom = 535.0
    panel_width = 470.0
    left_x = 75.0
    right_x = 650.0
    colors = {
        "constant_full": "#50555c",
        "trace_only": "#16817a",
        "exact_tensor": "#1d5ca8",
        "permuted_tensor": "#c54b3c",
    }
    nll_bounds = padded_bounds(list(nll_improvement.values()), 0.0)
    memory_bounds = padded_bounds(list(squared_memory.values()), 0.05)

    def x_position(panel_x: float, fold: int) -> float:
        return panel_x + 55.0 + (fold - 1) * (panel_width - 110.0) / 3.0

    def y_position(value: float, bounds: tuple[float, float]) -> float:
        low, high = bounds
        return panel_bottom - (value - low) * (panel_bottom - panel_top) / (high - low)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="75" y="38" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#20242a">Deterministic microscopic L2p conditional-diffusion gate</text>',
        '<text x="75" y="65" font-family="Arial, sans-serif" font-size="13" fill="#50555c">Four held clones; unchanged covariance models and tolerances; unclipped axes</text>',
    ]
    for index, model in enumerate(plot_models):
        legend_x = 80 + 270 * index
        svg.extend(
            [
                f'<line x1="{legend_x}" y1="91" x2="{legend_x + 28}" y2="91" stroke="{colors[model]}" stroke-width="3"/>',
                f'<circle cx="{legend_x + 14}" cy="91" r="4" fill="{colors[model]}"/>',
                f'<text x="{legend_x + 36}" y="96" font-family="Arial, sans-serif" font-size="13" fill="#30353b">{model}</text>',
            ]
        )

    panels = (
        (
            left_x,
            nll_bounds,
            nll_improvement,
            "negative log-likelihood improvement",
            "constant_full NLL - model NLL",
            0.0,
            "zero improvement",
        ),
        (
            right_x,
            memory_bounds,
            squared_memory,
            "maximum absolute squared whitened correlation",
            "held residual; lower is better",
            0.05,
            "tolerance = 0.05",
        ),
    )
    for panel_x, bounds, values, title, subtitle, reference, reference_label in panels:
        svg.append(
            f'<rect x="{panel_x}" y="{panel_top}" width="{panel_width}" height="{panel_bottom - panel_top}" fill="#fbfcfd" stroke="#c9cdd2"/>'
        )
        for tick in range(5):
            fraction = tick / 4.0
            value = bounds[0] + fraction * (bounds[1] - bounds[0])
            y = y_position(value, bounds)
            svg.extend(
                [
                    f'<line x1="{panel_x}" y1="{y:.3f}" x2="{panel_x + panel_width}" y2="{y:.3f}" stroke="#e3e6e9"/>',
                    f'<text x="{panel_x - 9}" y="{y + 4:.3f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#555b62">{value:.3g}</text>',
                ]
            )
        reference_y = y_position(reference, bounds)
        svg.extend(
            [
                f'<line x1="{panel_x}" y1="{reference_y:.3f}" x2="{panel_x + panel_width}" y2="{reference_y:.3f}" stroke="#15191e" stroke-width="1.4" stroke-dasharray="7 5"/>',
                f'<text x="{panel_x + panel_width - 6}" y="{reference_y - 7:.3f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#15191e">{reference_label}</text>',
                f'<text x="{panel_x}" y="585" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#24282e">{title}</text>',
                f'<text x="{panel_x}" y="607" font-family="Arial, sans-serif" font-size="12" fill="#555b62">{subtitle}</text>',
            ]
        )
        for fold in range(1, 5):
            x = x_position(panel_x, fold)
            svg.extend(
                [
                    f'<line x1="{x:.3f}" y1="{panel_bottom}" x2="{x:.3f}" y2="{panel_bottom + 6}" stroke="#555b62"/>',
                    f'<text x="{x:.3f}" y="558" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#444a50">fold {fold}</text>',
                ]
            )
        for model in plot_models:
            points = " ".join(
                f'{x_position(panel_x, fold):.3f},{y_position(values[(fold, model)], bounds):.3f}'
                for fold in range(1, 5)
            )
            svg.append(
                f'<polyline points="{points}" fill="none" stroke="{colors[model]}" stroke-width="2.4"/>'
            )
            for fold in range(1, 5):
                x = x_position(panel_x, fold)
                y = y_position(values[(fold, model)], bounds)
                svg.append(
                    f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4.2" fill="{colors[model]}"/>'
                )
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n")


def _model_passes_absolute_gate(row: dict[str, object]) -> bool:
    return (
        float(row["maximum_absolute_whitened_correlation"]) <= 0.05
        and float(row["maximum_absolute_squared_whitened_correlation"]) <= 0.05
        and float(row["maximum_absolute_component_excess_kurtosis"]) <= 0.35
        and 0.8 <= float(row["mean_squared_mahalanobis_per_dimension"]) <= 1.2
        and float(row["maximum_absolute_whitened_covariance_error"]) <= 0.10
    )


def classify_l2p_conditional_diffusion(
    detail_rows: list[dict[str, object]],
    convergence_rows: list[dict[str, object]],
) -> dict[str, object]:
    expected = {(fold, model) for fold in range(1, 5) for model in MODELS}
    actual = {
        (int(float(row["fold_index"])), str(row["model"])) for row in detail_rows
    }
    convergence_folds = {int(float(row["fold_index"])) for row in convergence_rows}
    if actual != expected or convergence_folds != {1, 2, 3, 4}:
        raise ValueError("four-fold five-model conditional-diffusion grid is not complete")
    by_cell = {
        (int(float(row["fold_index"])), str(row["model"])): row
        for row in detail_rows
    }
    estimators = {
        str(row.get("numerical_estimator", "rademacher_32_probe"))
        for row in convergence_rows
    }
    if len(estimators) != 1:
        raise ValueError("conditional-diffusion estimators must match across folds")
    estimator = estimators.pop()
    if estimator == "rademacher_32_probe":
        probe_converged = all(
            float(row["prefix_median_relative_frobenius_error"]) <= 0.10
            and float(row["prefix_p95_relative_frobenius_error"]) <= 0.25
            and float(row["step_median_relative_frobenius_error"]) <= 0.10
            and float(row["step_p95_relative_frobenius_error"]) <= 0.25
            for row in convergence_rows
        )
        numerical_gate_pass = probe_converged
        model_name = "l2p_conditional_diffusion_32_probe_tensor"
    elif estimator == "deterministic_velocity_jacobian":
        probe_converged = False
        numerical_gate_pass = all(
            float(row["numerical_gate_pass"]) == 1.0
            for row in convergence_rows
        )
        model_name = "l2p_conditional_diffusion_deterministic_velocity_jacobian_tensor"
    else:
        raise ValueError("unknown conditional-diffusion numerical estimator")
    nll_improvement = np.asarray(
        [
            float(by_cell[(fold, "constant_full")]["negative_log_likelihood"])
            - float(by_cell[(fold, "exact_tensor")]["negative_log_likelihood"])
            for fold in range(1, 5)
        ]
    )
    orientation_improvement = np.asarray(
        [
            float(by_cell[(fold, "trace_only")]["negative_log_likelihood"])
            - float(by_cell[(fold, "exact_tensor")]["negative_log_likelihood"])
            for fold in range(1, 5)
        ]
    )
    nll_interval = replicate_first_t_interval(nll_improvement)
    orientation_interval = replicate_first_t_interval(orientation_improvement)
    exact_rows = [by_cell[(fold, "exact_tensor")] for fold in range(1, 5)]
    constant_rows = [by_cell[(fold, "constant_full")] for fold in range(1, 5)]
    permuted_rows = [by_cell[(fold, "permuted_tensor")] for fold in range(1, 5)]
    every_fold_nll_improves = bool(np.all(nll_improvement > 0.0))
    every_fold_memory_reduction = all(
        float(exact["maximum_absolute_squared_whitened_correlation"])
        <= 0.75 * float(constant["maximum_absolute_squared_whitened_correlation"])
        for exact, constant in zip(exact_rows, constant_rows, strict=True)
    )
    exact_absolute = all(_model_passes_absolute_gate(row) for row in exact_rows)
    permuted_rejected = not all(_model_passes_absolute_gate(row) for row in permuted_rows)
    floor_pass = all(
        float(row["isotropic_floor_variance_fraction"]) <= 0.25 for row in exact_rows
    )
    supported = (
        numerical_gate_pass
        and every_fold_nll_improves
        and float(nll_interval["ci95_low"]) > 0.0
        and every_fold_memory_reduction
        and exact_absolute
        and permuted_rejected
        and floor_pass
    )
    informative = (
        not supported
        and every_fold_nll_improves
        and float(nll_interval["ci95_low"]) > 0.0
    ) or (not supported and every_fold_memory_reduction)
    orientation_required = (
        supported
        and bool(np.all(orientation_improvement > 0.0))
        and float(orientation_interval["ci95_low"]) > 0.0
    )
    return {
        "record": "verdict",
        "model": model_name,
        "conditional_diffusion_estimator": estimator,
        "held_clone_count": 4.0,
        "mean_constant_to_tensor_nll_improvement": nll_interval["mean"],
        "constant_to_tensor_nll_improvement_standard_error": nll_interval[
            "standard_error"
        ],
        "constant_to_tensor_nll_improvement_ci95_low": nll_interval["ci95_low"],
        "constant_to_tensor_nll_improvement_ci95_high": nll_interval["ci95_high"],
        "mean_trace_to_tensor_nll_improvement": orientation_interval["mean"],
        "trace_to_tensor_nll_improvement_ci95_low": orientation_interval["ci95_low"],
        "trace_to_tensor_nll_improvement_ci95_high": orientation_interval["ci95_high"],
        "every_fold_nll_improves": float(every_fold_nll_improves),
        "every_fold_squared_memory_reduction_25pct": float(every_fold_memory_reduction),
        "every_fold_exact_tensor_absolute_gate_pass": float(exact_absolute),
        "permuted_tensor_null_rejected": float(permuted_rejected),
        "every_fold_floor_fraction_pass": float(floor_pass),
        "l2p_diffusion_probe_converged": float(probe_converged),
        "l2p_diffusion_numerical_gate_pass": float(numerical_gate_pass),
        "l2p_conditional_diffusion_supported": float(supported),
        "l2p_conditional_diffusion_informative_but_insufficient": float(informative),
        "l2p_tensor_orientation_required": float(orientation_required),
        "l3p_derivation_authorized": float(informative),
        "microscopic_environment_coordinate_z_allowed": 0.0,
        "continuous_gaussian_langevin_bath_allowed": 0.0,
        "autonomous_single_particle_gle_allowed": 0.0,
        "complete_event_clock_closure_allowed": 0.0,
        "kramers_escape_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def load_conditional_diffusion_caches(
    directory: Path,
    clones: list[dict[str, np.ndarray | float | str]],
) -> list[dict[str, np.ndarray | float | str]]:
    by_hash: dict[str, dict[str, np.ndarray | float | str]] = {}
    probe_paths = sorted(directory.glob("clone_*_l2p_conditional_diffusion.npz"))
    deterministic_paths = sorted(
        directory.glob("clone_*_l2p_deterministic_diffusion.npz")
    )
    if probe_paths and deterministic_paths:
        raise ValueError("conditional-diffusion cache directory mixes estimators")
    for path in probe_paths + deterministic_paths:
        with np.load(path, allow_pickle=False) as cache:
            source_hash = str(np.asarray(cache["trajectory_sha256"]).item())
            if source_hash in by_hash:
                raise ValueError("duplicate conditional-diffusion trajectory hash")
            claim = float(cache["thermodynamic_claim_allowed"])
            completed = int(float(cache["completed_frame_count"]))
            requested = int(float(cache["requested_frame_count"]))
            q = np.asarray(cache["l2p_conditional_diffusion"], dtype=float)
            targets = np.asarray(cache["target_indices"], dtype=int)
            if (
                claim != 0.0
                or completed != requested
                or q.shape != (completed, len(targets), 3, 3)
                or np.any(~np.isfinite(q))
            ):
                raise ValueError(f"incomplete conditional-diffusion cache: {path}")
            loaded: dict[str, np.ndarray | float | str] = {
                "conditional_diffusion": q,
                "target_indices": targets,
                "path": str(path.resolve()),
                "thermodynamic_claim_allowed": 0.0,
            }
            if path in deterministic_paths:
                estimator = str(np.asarray(cache["estimator"]).item())
                numerical_gate = float(cache["deterministic_numerical_gate_pass"])
                numerical_state = str(np.asarray(cache["numerical_state"]).item())
                potential_protocol = str(
                    np.asarray(cache["potential_protocol"]).item()
                )
                numerical_arrays = {
                    name: np.asarray(cache[name], dtype=float)
                    for name in (
                        "a_primary_reference_error",
                        "q_primary_reference_error",
                        "a_coarse_reference_error",
                        "q_coarse_reference_error",
                        "directional_response_error",
                        "q_minimum_eigenvalue",
                        "q_trace",
                    )
                }
                if (
                    estimator != "deterministic_velocity_jacobian"
                    or numerical_gate != 1.0
                    or numerical_state
                    != "deterministic_jacobian_numerically_resolved"
                    or any(
                        np.any(~np.isfinite(values))
                        for values in numerical_arrays.values()
                    )
                ):
                    raise ValueError(
                        f"deterministic numerical gate did not pass: {path}"
                    )
                loaded.update(numerical_arrays)
                loaded["numerical_estimator"] = estimator
                loaded["numerical_gate_pass"] = numerical_gate
                loaded["numerical_state"] = numerical_state
                loaded["potential_protocol"] = potential_protocol
            else:
                prefix_error = np.asarray(
                    cache["prefix_relative_frobenius_error"], dtype=float
                )
                step_error = np.asarray(
                    cache["step_relative_frobenius_error"], dtype=float
                )
                if (
                    prefix_error.shape[0] != completed
                    or np.any(~np.isfinite(prefix_error))
                    or np.any(~np.isfinite(step_error))
                ):
                    raise ValueError(f"incomplete conditional-diffusion cache: {path}")
                loaded.update(
                    {
                        "prefix_error": prefix_error,
                        "step_error": step_error,
                        "numerical_estimator": "rademacher_32_probe",
                        "numerical_gate_pass": 0.0,
                    }
                )
            by_hash[source_hash] = loaded
    matched = []
    for clone in clones:
        source_hash = str(clone["trajectory_sha256"])
        if source_hash not in by_hash:
            raise ValueError("conditional-diffusion cache is missing a matched clone")
        cache = by_hash[source_hash]
        if not np.array_equal(cache["target_indices"], clone["target_indices"]):
            raise ValueError("conditional-diffusion target indices do not match")
        matched.append(cache)
    if len(by_hash) != len(clones):
        raise ValueError("conditional-diffusion directory contains unmatched clones")
    protocols = {
        str(cache["potential_protocol"])
        for cache in matched
        if "potential_protocol" in cache
    }
    if len(protocols) > 1:
        raise ValueError("deterministic cache potential protocols do not match")
    return matched


def permute_covariance_time(covariance: np.ndarray, *, seed: int) -> np.ndarray:
    values = np.asarray(covariance, dtype=float)
    rng = np.random.default_rng(seed)
    permuted = np.empty_like(values)
    for target in range(values.shape[1]):
        permuted[:, target] = values[rng.permutation(len(values)), target]
    return permuted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-cache-directory", type=Path, required=True)
    parser.add_argument(
        "--second-generator-cache-directories", type=Path, nargs="+", required=True
    )
    parser.add_argument("--conditional-diffusion-cache-directory", type=Path, required=True)
    parser.add_argument("--memory-order", type=int, default=40)
    parser.add_argument("--bath-order", type=int, default=16)
    parser.add_argument("--maximum-lag", type=int, default=40)
    parser.add_argument("--maximum-frame-count", type=int, default=200)
    parser.add_argument("--frame-time", type=float, default=0.01)
    parser.add_argument("--ridge-regularization", type=float, default=1e-8)
    parser.add_argument("--var-ridge-regularization", type=float, default=1e-6)
    parser.add_argument("--permutation-seed", type=int, default=20260721)
    parser.add_argument("--gaussian-reference-seed", type=int, default=20260722)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--figure-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.memory_order < 1
        or args.bath_order < 1
        or args.maximum_lag < 1
        or args.maximum_frame_count < 2
        or args.frame_time <= 0.0
        or args.ridge_regularization < 0.0
        or args.var_ridge_regularization < 0.0
    ):
        raise ValueError("invalid L2p conditional-diffusion analysis controls")
    clones = load_matched_second_generator_clones(
        args.drift_cache_directory,
        args.second_generator_cache_directories,
        maximum_frame_count=args.maximum_frame_count,
    )
    if len(clones) != 4:
        raise ValueError("the frozen conditional-diffusion gate requires four clones")
    q_caches = load_conditional_diffusion_caches(
        args.conditional_diffusion_cache_directory, clones
    )
    details: list[dict[str, object]] = []
    convergence: list[dict[str, object]] = []
    for fold_index, (held, held_q_cache) in enumerate(zip(clones, q_caches, strict=True), start=1):
        training = [clone for index, clone in enumerate(clones) if index != fold_index - 1]
        training_q_caches = [cache for index, cache in enumerate(q_caches) if index != fold_index - 1]
        extracted = extract_l2p_white_innovation_fold(
            training,
            held,
            memory_order=args.memory_order,
            bath_order=args.bath_order,
            ridge_regularization=args.ridge_regularization,
            var_ridge_regularization=args.var_ridge_regularization,
            include_l2p=True,
        )
        frame_indices = np.asarray(extracted["held_source_frame_indices"], dtype=int)
        normalization_scale = float(np.asarray(extracted["normalization_scale"])[0, 0, 3])
        training_residual = np.asarray(extracted["training_white_innovation"])[
            :, :, 3
        ].reshape(len(frame_indices), -1, 3)
        held_residual = np.asarray(extracted["held_l2p_vector_innovation"])
        training_q = np.concatenate(
            [np.asarray(cache["conditional_diffusion"])[frame_indices] for cache in training_q_caches],
            axis=1,
        )
        held_q = np.asarray(held_q_cache["conditional_diffusion"])[frame_indices]
        training_q = args.frame_time * training_q / normalization_scale**2
        held_q = args.frame_time * held_q / normalization_scale**2
        if training_q.shape[:2] != training_residual.shape[:2] or held_q.shape[:2] != held_residual.shape[:2]:
            raise ValueError("conditional diffusion is not aligned with white innovations")

        constant_full = fit_constant_covariance(training_residual)
        constant_isotropic = np.eye(3) * float(np.mean(training_residual**2))
        training_trace = np.trace(training_q, axis1=-2, axis2=-1)[..., None, None] / 3.0 * np.eye(3)
        held_trace = np.trace(held_q, axis1=-2, axis2=-1)[..., None, None] / 3.0 * np.eye(3)
        trace_fit = fit_scaled_conditional_covariance(training_residual, training_trace)
        tensor_fit = fit_scaled_conditional_covariance(training_residual, training_q)
        permuted_training_q = permute_covariance_time(
            training_q, seed=args.permutation_seed + 100 * fold_index
        )
        permuted_held_q = permute_covariance_time(
            held_q, seed=args.permutation_seed + fold_index
        )
        permuted_fit = fit_scaled_conditional_covariance(
            training_residual, permuted_training_q
        )
        model_covariances = {
            "constant_full": np.broadcast_to(constant_full, held_q.shape),
            "constant_isotropic": np.broadcast_to(constant_isotropic, held_q.shape),
            "trace_only": trace_fit["scale"] * held_trace + trace_fit["isotropic_floor"] * np.eye(3),
            "exact_tensor": tensor_fit["scale"] * held_q + tensor_fit["isotropic_floor"] * np.eye(3),
            "permuted_tensor": permuted_fit["scale"] * permuted_held_q + permuted_fit["isotropic_floor"] * np.eye(3),
        }
        fits = {
            "constant_full": {"scale": 0.0, "isotropic_floor": 0.0, "isotropic_floor_variance_fraction": 0.0},
            "constant_isotropic": {"scale": 0.0, "isotropic_floor": float(np.mean(training_residual**2)), "isotropic_floor_variance_fraction": 1.0},
            "trace_only": trace_fit,
            "exact_tensor": tensor_fit,
            "permuted_tensor": permuted_fit,
        }
        for model in MODELS:
            diagnostic = conditional_covariance_diagnostic(
                held_residual,
                np.asarray(model_covariances[model]),
                maximum_lag=args.maximum_lag,
                gaussian_seed=args.gaussian_reference_seed + fold_index,
            )
            fit = fits[model]
            details.append(
                {
                    "record": "held_fold_model",
                    "fold_index": float(fold_index),
                    "model": model,
                    **diagnostic,
                    "fitted_scale": float(fit["scale"]),
                    "fitted_isotropic_floor": float(fit["isotropic_floor"]),
                    "isotropic_floor_variance_fraction": float(
                        fit["isotropic_floor_variance_fraction"]
                    ),
                    "fit_uses_held_samples": 0.0,
                    "trajectory_sha256": str(held["trajectory_sha256"]),
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
        estimator = str(held_q_cache["numerical_estimator"])
        convergence_row: dict[str, object] = {
            "record": "held_fold_convergence",
            "fold_index": float(fold_index),
            "numerical_estimator": estimator,
            "numerical_gate_pass": float(held_q_cache["numerical_gate_pass"]),
            "trajectory_sha256": str(held["trajectory_sha256"]),
            "thermodynamic_claim_allowed": 0.0,
        }
        if estimator == "deterministic_velocity_jacobian":
            convergence_row["numerical_state"] = str(
                held_q_cache["numerical_state"]
            )
            convergence_row["potential_protocol"] = str(
                held_q_cache["potential_protocol"]
            )
            for prefix, key in (
                ("a_primary_reference", "a_primary_reference_error"),
                ("q_primary_reference", "q_primary_reference_error"),
                ("a_coarse_reference", "a_coarse_reference_error"),
                ("q_coarse_reference", "q_coarse_reference_error"),
                ("directional", "directional_response_error"),
            ):
                values = np.asarray(held_q_cache[key], dtype=float)
                convergence_row[f"{prefix}_median_relative_error"] = float(
                    np.median(values)
                )
                convergence_row[f"{prefix}_p95_relative_error"] = float(
                    np.quantile(values, 0.95)
                )
            convergence_row["q_minimum_eigenvalue"] = float(
                np.min(np.asarray(held_q_cache["q_minimum_eigenvalue"]))
            )
        else:
            prefix_error = np.asarray(held_q_cache["prefix_error"])
            step_error = np.asarray(held_q_cache["step_error"])
            convergence_row.update(
                {
                    "prefix_median_relative_frobenius_error": float(
                        np.median(prefix_error[:, -1])
                    ),
                    "prefix_p95_relative_frobenius_error": float(
                        np.quantile(prefix_error[:, -1], 0.95)
                    ),
                    "step_median_relative_frobenius_error": float(
                        np.median(step_error)
                    ),
                    "step_p95_relative_frobenius_error": float(
                        np.quantile(step_error, 0.95)
                    ),
                }
            )
        convergence.append(convergence_row)

    summaries: list[dict[str, object]] = []
    metric_names = (
        "negative_log_likelihood",
        "mean_squared_mahalanobis_per_dimension",
        "maximum_absolute_whitened_covariance_error",
        "maximum_absolute_component_excess_kurtosis",
        "maximum_absolute_whitened_correlation",
        "maximum_absolute_squared_whitened_correlation",
        "gaussian_energy_distance",
        "isotropic_floor_variance_fraction",
    )
    for model in MODELS:
        selected = [row for row in details if row["model"] == model]
        summary: dict[str, object] = {
            "record": "model_aggregate",
            "model": model,
            "held_clone_count": 4.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected])
            summary[f"mean_{metric}"] = float(np.mean(values))
            summary[f"maximum_{metric}"] = float(np.max(values))
        summary["every_fold_absolute_gate_pass"] = float(
            all(_model_passes_absolute_gate(row) for row in selected)
        )
        summaries.append(summary)
    summaries.append(classify_l2p_conditional_diffusion(details, convergence))
    write_rows(output_path(args.output_prefix, "_details.csv"), details)
    write_rows(output_path(args.output_prefix, "_convergence.csv"), convergence)
    write_rows(output_path(args.output_prefix, "_summary.csv"), summaries)
    if args.figure_path is not None:
        write_diagnostic_svg(args.figure_path, details)


if __name__ == "__main__":
    main()
