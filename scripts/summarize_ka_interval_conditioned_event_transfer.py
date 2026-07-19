#!/usr/bin/env python3
"""Summarize lag-conditioned event transfer and its fail-closed claim boundary."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WAVE_KEYS = ("k2", "k4", "k7p25")
CHANNELS = ("count", "kernel", "residual")
MODEL_CODES = ("ccc", "hcc", "chc", "cch", "hhc", "hch", "chh", "hhh")
KERNEL_MODELS = ("lag_conditioned", "lag_pooled")
EXPECTED_GRIDS = {
    0.45: {replicate: (20, 50, 100, 200, 500, 1000, 2000, 3000, 4096) for replicate in range(1, 4)},
    0.58: {replicate: (10, 20, 50, 100, 200, 400, 600) for replicate in range(1, 6)},
}
MSD_TOLERANCE = 0.10
NGP_TOLERANCE = 0.30
FS_TOLERANCE = 0.03
COUNT_TAIL_TOLERANCE = 0.001
DIFFUSION_TOLERANCE = 0.15
ALPHA_TOLERANCE = 0.20
DIFFUSION_ALPHA_TOLERANCE = 0.25


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, object], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {key}")
    return value


def validate_transfer_rows(rows: list[dict[str, object]]) -> None:
    expected = {
        (temperature, replicate, lag, code)
        for temperature, replicate_lags in EXPECTED_GRIDS.items()
        for replicate, lags in replicate_lags.items()
        for lag in lags
        for code in MODEL_CODES
    }
    actual: set[tuple[float, int, int, str]] = set()
    cells: dict[tuple[float, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        temperature = _float(row, "temperature")
        replicate = int(_float(row, "replicate"))
        lag = int(_float(row, "lag"))
        code = str(row["model_code"])
        key = (temperature, replicate, lag, code)
        if key in actual:
            raise ValueError(f"duplicate three-channel grid cell {key}")
        actual.add(key)
        cells[(temperature, replicate, lag)].append(row)
        sources = tuple(str(row[f"{channel}_source"]) for channel in CHANNELS)
        expected_code = "".join("h" if source == "heldout" else "c" for source in sources)
        if sources.count("calibration") + sources.count("heldout") != 3 or code != expected_code:
            raise ValueError(f"invalid source code in three-channel grid cell {key}")
        tail_probability = _float(row, "count_tail_probability")
        if not -1e-12 <= tail_probability <= COUNT_TAIL_TOLERANCE:
            raise ValueError(f"count support is incomplete in three-channel grid cell {key}")
        for field in ("predicted_msd", "observed_msd", "predicted_ngp", "observed_ngp"):
            _float(row, field)
        for wave_key in WAVE_KEYS:
            _float(row, f"predicted_fs_{wave_key}")
            _float(row, f"observed_fs_{wave_key}")
    if actual != expected:
        missing = sorted(expected - actual)[:3]
        extra = sorted(actual - expected)[:3]
        raise ValueError(f"three-channel grid changed; missing={missing}, extra={extra}")
    for key, local in cells.items():
        reference = local[0]
        for row in local[1:]:
            for field in ("observed_msd", "observed_ngp", *(f"observed_fs_{wave_key}" for wave_key in WAVE_KEYS)):
                if not math.isclose(_float(row, field), _float(reference, field), abs_tol=1e-12):
                    raise ValueError(f"heldout target changed across counterfactuals at {key}")


def validate_kernel_ablation_rows(rows: list[dict[str, object]]) -> None:
    expected = {
        (temperature, replicate, lag, model)
        for temperature, replicate_lags in EXPECTED_GRIDS.items()
        for replicate, lags in replicate_lags.items()
        for lag in lags
        for model in KERNEL_MODELS
    }
    actual: set[tuple[float, int, int, str]] = set()
    cells: dict[tuple[float, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        temperature = _float(row, "temperature")
        replicate = int(_float(row, "replicate"))
        lag = int(_float(row, "lag"))
        model = str(row["kernel_model"])
        key = (temperature, replicate, lag, model)
        if key in actual:
            raise ValueError(f"duplicate kernel-ablation grid cell {key}")
        actual.add(key)
        cells[(temperature, replicate, lag)].append(row)
        tail = _float(row, "count_tail_probability")
        if not -1e-12 <= tail <= COUNT_TAIL_TOLERANCE:
            raise ValueError(f"count support is incomplete in kernel-ablation grid cell {key}")
        for field in ("predicted_msd", "observed_msd", "predicted_ngp", "observed_ngp"):
            _float(row, field)
        for wave_key in WAVE_KEYS:
            _float(row, f"predicted_fs_{wave_key}")
            _float(row, f"observed_fs_{wave_key}")
    if actual != expected:
        missing = sorted(expected - actual)[:3]
        extra = sorted(actual - expected)[:3]
        raise ValueError(f"kernel-ablation grid changed; missing={missing}, extra={extra}")
    for key, local in cells.items():
        reference = local[0]
        for row in local[1:]:
            for field in ("observed_msd", "observed_ngp", *(f"observed_fs_{wave_key}" for wave_key in WAVE_KEYS)):
                if not math.isclose(_float(row, field), _float(reference, field), abs_tol=1e-12):
                    raise ValueError(f"heldout target changed across kernel ablation at {key}")


def three_channel_shapley(values: dict[str, float]) -> dict[str, float]:
    if set(values) != set(MODEL_CODES) or any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Shapley attribution requires the complete finite three-channel cube")
    attribution: dict[str, float] = {}
    for index, channel in enumerate(CHANNELS):
        others = [local for local in range(3) if local != index]
        contribution = 0.0
        for mask in range(1 << len(others)):
            base = ["c", "c", "c"]
            subset_size = 0
            for bit, local_index in enumerate(others):
                if mask & (1 << bit):
                    base[local_index] = "h"
                    subset_size += 1
            without = "".join(base)
            base[index] = "h"
            with_channel = "".join(base)
            weight = math.factorial(subset_size) * math.factorial(2 - subset_size) / 6.0
            contribution += weight * (values[with_channel] - values[without])
        attribution[channel] = contribution
    if not math.isclose(
        math.fsum(attribution.values()), values["hhh"] - values["ccc"], abs_tol=1e-12
    ):
        raise ValueError("three-channel Shapley attribution lost closure")
    return attribution


def _pooled_ngp(msd_values: list[float], ngp_values: list[float]) -> tuple[float, float]:
    mean_msd = math.fsum(msd_values) / len(msd_values)
    mean_fourth = math.fsum(
        (5.0 / 3.0) * (1.0 + ngp) * msd**2
        for msd, ngp in zip(msd_values, ngp_values)
    ) / len(msd_values)
    ngp = 3.0 * mean_fourth / (5.0 * mean_msd**2) - 1.0 if mean_msd > 0.0 else 0.0
    return mean_fourth, ngp


def _alpha_crossing(lags: list[float], values: list[float]) -> float:
    threshold = math.exp(-1.0)
    for index in range(len(lags) - 1):
        earlier, later = values[index], values[index + 1]
        if earlier > threshold >= later and earlier > 0.0 and later > 0.0:
            fraction = (math.log(threshold) - math.log(earlier)) / (
                math.log(later) - math.log(earlier)
            )
            return math.exp(
                math.log(lags[index])
                + fraction * (math.log(lags[index + 1]) - math.log(lags[index]))
            )
    return math.nan


def summarize_transfer_rows(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    validate_transfer_rows(rows)
    curves: list[dict[str, object]] = []
    for temperature in sorted(EXPECTED_GRIDS):
        for code in MODEL_CODES:
            selected = [
                row
                for row in rows
                if math.isclose(_float(row, "temperature"), temperature)
                and row["model_code"] == code
            ]
            for lag in sorted({int(_float(row, "lag")) for row in selected}):
                local = [row for row in selected if int(_float(row, "lag")) == lag]
                predicted_msds = [_float(row, "predicted_msd") for row in local]
                observed_msds = [_float(row, "observed_msd") for row in local]
                predicted_fourth, predicted_ngp = _pooled_ngp(
                    predicted_msds, [_float(row, "predicted_ngp") for row in local]
                )
                observed_fourth, observed_ngp = _pooled_ngp(
                    observed_msds, [_float(row, "observed_ngp") for row in local]
                )
                predicted_msd = math.fsum(predicted_msds) / len(local)
                observed_msd = math.fsum(observed_msds) / len(local)
                curve: dict[str, object] = {
                    "temperature": temperature,
                    "model_code": code,
                    "lag": float(lag),
                    "replicate_count": float(len(local)),
                    "predicted_msd": predicted_msd,
                    "observed_msd": observed_msd,
                    "msd_relative_error": abs(predicted_msd / observed_msd - 1.0),
                    "predicted_fourth_moment": predicted_fourth,
                    "observed_fourth_moment": observed_fourth,
                    "predicted_ngp": predicted_ngp,
                    "observed_ngp": observed_ngp,
                    "ngp_absolute_error": abs(predicted_ngp - observed_ngp),
                    "maximum_count_tail_probability": max(
                        _float(row, "count_tail_probability") for row in local
                    ),
                }
                for wave_key in WAVE_KEYS:
                    predicted = math.fsum(
                        _float(row, f"predicted_fs_{wave_key}") for row in local
                    ) / len(local)
                    observed = math.fsum(
                        _float(row, f"observed_fs_{wave_key}") for row in local
                    ) / len(local)
                    curve[f"predicted_fs_{wave_key}"] = predicted
                    curve[f"observed_fs_{wave_key}"] = observed
                    curve[f"absolute_error_fs_{wave_key}"] = abs(predicted - observed)
                curves.append(curve)

    summaries: list[dict[str, object]] = []
    for temperature in sorted(EXPECTED_GRIDS):
        for code in MODEL_CODES:
            local_rows = [
                row
                for row in rows
                if math.isclose(_float(row, "temperature"), temperature)
                and row["model_code"] == code
            ]
            local_curves = [
                row
                for row in curves
                if math.isclose(_float(row, "temperature"), temperature)
                and row["model_code"] == code
            ]
            maximum_replicate_fs = max(
                _float(row, f"absolute_error_fs_{wave_key}")
                for row in local_rows
                for wave_key in WAVE_KEYS
            )
            maximum_ensemble_fs = max(
                _float(row, f"absolute_error_fs_{wave_key}")
                for row in local_curves
                for wave_key in WAVE_KEYS
            )
            lags = [_float(row, "lag") for row in local_curves]
            predicted_alpha = _alpha_crossing(
                lags, [_float(row, "predicted_fs_k7p25") for row in local_curves]
            )
            observed_alpha = _alpha_crossing(
                lags, [_float(row, "observed_fs_k7p25") for row in local_curves]
            )
            last = local_curves[-1]
            diffusion_lag = _float(last, "lag")
            predicted_diffusion = _float(last, "predicted_msd") / (6.0 * diffusion_lag)
            observed_diffusion = _float(last, "observed_msd") / (6.0 * diffusion_lag)
            diffusion_error = abs(predicted_diffusion / observed_diffusion - 1.0)
            alpha_ready = math.isfinite(predicted_alpha) and math.isfinite(observed_alpha)
            alpha_error = abs(predicted_alpha / observed_alpha - 1.0) if alpha_ready else math.nan
            predicted_product = predicted_diffusion * predicted_alpha
            observed_product = observed_diffusion * observed_alpha
            product_error = abs(predicted_product / observed_product - 1.0) if alpha_ready else math.nan
            maximum_ensemble_msd = max(_float(row, "msd_relative_error") for row in local_curves)
            maximum_ensemble_ngp = max(_float(row, "ngp_absolute_error") for row in local_curves)
            maximum_tail = max(_float(row, "maximum_count_tail_probability") for row in local_curves)
            ensemble_curve_pass = (
                maximum_ensemble_msd <= MSD_TOLERANCE
                and maximum_ensemble_ngp <= NGP_TOLERANCE
                and maximum_ensemble_fs <= FS_TOLERANCE
                and maximum_tail <= COUNT_TAIL_TOLERANCE
            )
            replicate_curve_pass = (
                max(_float(row, "msd_relative_error") for row in local_rows) <= MSD_TOLERANCE
                and max(_float(row, "ngp_absolute_error") for row in local_rows) <= NGP_TOLERANCE
                and maximum_replicate_fs <= FS_TOLERANCE
            )
            scalar_pass = (
                alpha_ready
                and diffusion_error <= DIFFUSION_TOLERANCE
                and alpha_error <= ALPHA_TOLERANCE
                and product_error <= DIFFUSION_ALPHA_TOLERANCE
            )
            summaries.append(
                {
                    "temperature": temperature,
                    "model_code": code,
                    "lag_count": float(len(local_curves)),
                    "replicate_count": float(len(EXPECTED_GRIDS[temperature])),
                    "maximum_replicate_msd_relative_error": max(
                        _float(row, "msd_relative_error") for row in local_rows
                    ),
                    "maximum_replicate_ngp_absolute_error": max(
                        _float(row, "ngp_absolute_error") for row in local_rows
                    ),
                    "maximum_replicate_fs_absolute_error": maximum_replicate_fs,
                    "maximum_ensemble_msd_relative_error": maximum_ensemble_msd,
                    "maximum_ensemble_ngp_absolute_error": maximum_ensemble_ngp,
                    "maximum_ensemble_fs_absolute_error": maximum_ensemble_fs,
                    "maximum_count_tail_probability": maximum_tail,
                    "diffusion_lag": diffusion_lag,
                    "predicted_diffusion": predicted_diffusion,
                    "observed_diffusion": observed_diffusion,
                    "diffusion_relative_error": diffusion_error,
                    "predicted_alpha_relaxation_time": predicted_alpha,
                    "observed_alpha_relaxation_time": observed_alpha,
                    "alpha_relaxation_relative_error": alpha_error,
                    "predicted_diffusion_alpha_product": predicted_product,
                    "observed_diffusion_alpha_product": observed_product,
                    "diffusion_alpha_product_relative_error": product_error,
                    "ensemble_curve_pass": float(ensemble_curve_pass),
                    "derived_scalar_pass": float(scalar_pass),
                    "replicate_curve_pass": float(replicate_curve_pass),
                }
            )

    attributions: list[dict[str, object]] = []
    for temperature in sorted(EXPECTED_GRIDS):
        local_lags = sorted(
            {int(_float(row, "lag")) for row in curves if math.isclose(_float(row, "temperature"), temperature)}
        )
        for lag in local_lags:
            cube = {
                str(row["model_code"]): row
                for row in curves
                if math.isclose(_float(row, "temperature"), temperature)
                and int(_float(row, "lag")) == lag
            }
            for observable in ("predicted_msd", "predicted_fourth_moment", *(f"predicted_fs_{key}" for key in WAVE_KEYS)):
                values = {code: _float(cube[code], observable) for code in MODEL_CODES}
                attribution = three_channel_shapley(values)
                observed_key = observable.replace("predicted_", "observed_")
                for channel in CHANNELS:
                    attributions.append(
                        {
                            "temperature": temperature,
                            "lag": float(lag),
                            "observable": observable.replace("predicted_", ""),
                            "channel": channel,
                            "shapley_contribution": attribution[channel],
                            "calibration_value": values["ccc"],
                            "heldout_oracle_value": values["hhh"],
                            "observed_value": _float(cube["hhh"], observed_key),
                            "calibration_to_oracle_shift": values["hhh"] - values["ccc"],
                            "oracle_representation_residual": _float(cube["hhh"], observed_key) - values["hhh"],
                        }
                    )
    return curves, summaries, attributions


def summarize_kernel_ablation_rows(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    validate_kernel_ablation_rows(rows)
    curves: list[dict[str, object]] = []
    for temperature in sorted(EXPECTED_GRIDS):
        for model in KERNEL_MODELS:
            selected = [
                row
                for row in rows
                if math.isclose(_float(row, "temperature"), temperature)
                and row["kernel_model"] == model
            ]
            for lag in sorted({int(_float(row, "lag")) for row in selected}):
                local = [row for row in selected if int(_float(row, "lag")) == lag]
                predicted_msds = [_float(row, "predicted_msd") for row in local]
                observed_msds = [_float(row, "observed_msd") for row in local]
                predicted_fourth, predicted_ngp = _pooled_ngp(
                    predicted_msds, [_float(row, "predicted_ngp") for row in local]
                )
                observed_fourth, observed_ngp = _pooled_ngp(
                    observed_msds, [_float(row, "observed_ngp") for row in local]
                )
                predicted_msd = math.fsum(predicted_msds) / len(local)
                observed_msd = math.fsum(observed_msds) / len(local)
                curve: dict[str, object] = {
                    "temperature": temperature,
                    "kernel_model": model,
                    "lag": float(lag),
                    "replicate_count": float(len(local)),
                    "predicted_msd": predicted_msd,
                    "observed_msd": observed_msd,
                    "msd_relative_error": abs(predicted_msd / observed_msd - 1.0),
                    "predicted_fourth_moment": predicted_fourth,
                    "observed_fourth_moment": observed_fourth,
                    "predicted_ngp": predicted_ngp,
                    "observed_ngp": observed_ngp,
                    "ngp_absolute_error": abs(predicted_ngp - observed_ngp),
                    "maximum_count_tail_probability": max(
                        _float(row, "count_tail_probability") for row in local
                    ),
                }
                for wave_key in WAVE_KEYS:
                    predicted = math.fsum(
                        _float(row, f"predicted_fs_{wave_key}") for row in local
                    ) / len(local)
                    observed = math.fsum(
                        _float(row, f"observed_fs_{wave_key}") for row in local
                    ) / len(local)
                    curve[f"predicted_fs_{wave_key}"] = predicted
                    curve[f"observed_fs_{wave_key}"] = observed
                    curve[f"absolute_error_fs_{wave_key}"] = abs(predicted - observed)
                curves.append(curve)

    summaries: list[dict[str, object]] = []
    for temperature in sorted(EXPECTED_GRIDS):
        for model in KERNEL_MODELS:
            local_rows = [
                row
                for row in rows
                if math.isclose(_float(row, "temperature"), temperature)
                and row["kernel_model"] == model
            ]
            local_curves = [
                row
                for row in curves
                if math.isclose(_float(row, "temperature"), temperature)
                and row["kernel_model"] == model
            ]
            maximum_ensemble_msd = max(_float(row, "msd_relative_error") for row in local_curves)
            maximum_ensemble_ngp = max(_float(row, "ngp_absolute_error") for row in local_curves)
            maximum_ensemble_fs = max(
                _float(row, f"absolute_error_fs_{wave_key}")
                for row in local_curves
                for wave_key in WAVE_KEYS
            )
            maximum_tail = max(_float(row, "maximum_count_tail_probability") for row in local_curves)
            lags = [_float(row, "lag") for row in local_curves]
            predicted_alpha = _alpha_crossing(
                lags, [_float(row, "predicted_fs_k7p25") for row in local_curves]
            )
            observed_alpha = _alpha_crossing(
                lags, [_float(row, "observed_fs_k7p25") for row in local_curves]
            )
            last = local_curves[-1]
            diffusion_lag = _float(last, "lag")
            predicted_diffusion = _float(last, "predicted_msd") / (6.0 * diffusion_lag)
            observed_diffusion = _float(last, "observed_msd") / (6.0 * diffusion_lag)
            diffusion_error = abs(predicted_diffusion / observed_diffusion - 1.0)
            alpha_ready = math.isfinite(predicted_alpha) and math.isfinite(observed_alpha)
            alpha_error = abs(predicted_alpha / observed_alpha - 1.0) if alpha_ready else math.nan
            product_error = (
                abs(
                    predicted_diffusion
                    * predicted_alpha
                    / (observed_diffusion * observed_alpha)
                    - 1.0
                )
                if alpha_ready
                else math.nan
            )
            ensemble_curve_pass = (
                maximum_ensemble_msd <= MSD_TOLERANCE
                and maximum_ensemble_ngp <= NGP_TOLERANCE
                and maximum_ensemble_fs <= FS_TOLERANCE
                and maximum_tail <= COUNT_TAIL_TOLERANCE
            )
            derived_scalar_pass = (
                alpha_ready
                and diffusion_error <= DIFFUSION_TOLERANCE
                and alpha_error <= ALPHA_TOLERANCE
                and product_error <= DIFFUSION_ALPHA_TOLERANCE
            )
            summaries.append(
                {
                    "temperature": temperature,
                    "kernel_model": model,
                    "lag_count": float(len(local_curves)),
                    "replicate_count": float(len(EXPECTED_GRIDS[temperature])),
                    "maximum_replicate_msd_relative_error": max(
                        _float(row, "msd_relative_error") for row in local_rows
                    ),
                    "maximum_replicate_ngp_absolute_error": max(
                        _float(row, "ngp_absolute_error") for row in local_rows
                    ),
                    "maximum_replicate_fs_absolute_error": max(
                        _float(row, f"absolute_error_fs_{wave_key}")
                        for row in local_rows
                        for wave_key in WAVE_KEYS
                    ),
                    "maximum_ensemble_msd_relative_error": maximum_ensemble_msd,
                    "maximum_ensemble_ngp_absolute_error": maximum_ensemble_ngp,
                    "maximum_ensemble_fs_absolute_error": maximum_ensemble_fs,
                    "maximum_count_tail_probability": maximum_tail,
                    "diffusion_relative_error": diffusion_error,
                    "predicted_alpha_relaxation_time": predicted_alpha,
                    "observed_alpha_relaxation_time": observed_alpha,
                    "alpha_relaxation_relative_error": alpha_error,
                    "diffusion_alpha_product_relative_error": product_error,
                    "ensemble_curve_pass": float(ensemble_curve_pass),
                    "derived_scalar_pass": float(derived_scalar_pass),
                }
            )
    return curves, summaries


def _summary_by_code(
    summaries: list[dict[str, object]], temperature: float, code: str
) -> dict[str, object]:
    return next(
        row
        for row in summaries
        if math.isclose(_float(row, "temperature"), temperature) and row["model_code"] == code
    )


def _summary_by_kernel(
    summaries: list[dict[str, object]], temperature: float, model: str
) -> dict[str, object]:
    return next(
        row
        for row in summaries
        if math.isclose(_float(row, "temperature"), temperature)
        and row["kernel_model"] == model
    )


def build_verdicts(
    summaries: list[dict[str, object]],
    kernel_ablation: list[dict[str, object]],
    kernel_ablation_rows: list[dict[str, object]],
    provenance: list[dict[str, object]],
    old_hybrid: list[dict[str, object]],
) -> list[dict[str, object]]:
    provenance_by_temperature = {_float(row, "temperature"): row for row in provenance}
    old_by_temperature = {_float(row, "temperature"): row for row in old_hybrid}
    verdicts: list[dict[str, object]] = []
    for temperature in sorted(EXPECTED_GRIDS):
        calibration = _summary_by_code(summaries, temperature, "ccc")
        oracle = _summary_by_code(summaries, temperature, "hhh")
        lag_conditioned = _summary_by_kernel(
            kernel_ablation, temperature, "lag_conditioned"
        )
        lag_pooled = _summary_by_kernel(kernel_ablation, temperature, "lag_pooled")
        local_provenance = provenance_by_temperature[temperature]
        parent_count = _float(local_provenance, "parent_sample_count")
        independently_prepared = bool(
            _float(local_provenance, "independently_prepared_parent_samples")
        )
        parent_sufficient = independently_prepared and parent_count >= 3.0
        previous_joint_pass = (
            _float(old_by_temperature[temperature], "joint_macro_transfer_pass")
            if temperature in old_by_temperature
            else 0.0
        )
        calibration_joint_pass = bool(_float(calibration, "ensemble_curve_pass")) and bool(
            _float(calibration, "derived_scalar_pass")
        )
        oracle_pass = bool(_float(oracle, "ensemble_curve_pass"))
        low_canary = temperature == 0.45 and calibration_joint_pass
        if any(
            not math.isclose(
                _float(calibration, field), _float(lag_conditioned, field), abs_tol=1e-12
            )
            for field in (
                "maximum_ensemble_msd_relative_error",
                "maximum_ensemble_ngp_absolute_error",
                "maximum_ensemble_fs_absolute_error",
            )
        ):
            raise ValueError("lag-conditioned ablation does not reproduce calibration cube")
        pooled_fs_error = _float(lag_pooled, "maximum_ensemble_fs_absolute_error")
        conditioned_fs_error = _float(
            lag_conditioned, "maximum_ensemble_fs_absolute_error"
        )
        lag_required_for_multik = (
            low_canary
            and not bool(_float(lag_pooled, "ensemble_curve_pass"))
            and _float(lag_pooled, "maximum_ensemble_msd_relative_error") <= MSD_TOLERANCE
            and _float(lag_pooled, "maximum_ensemble_ngp_absolute_error") <= NGP_TOLERANCE
            and pooled_fs_error > FS_TOLERANCE
            and _float(lag_pooled, "maximum_count_tail_probability")
            <= COUNT_TAIL_TOLERANCE
        )
        improved_restart_labels = 0
        for replicate in EXPECTED_GRIDS[temperature]:
            means = {}
            for model in KERNEL_MODELS:
                local = [
                    row
                    for row in kernel_ablation_rows
                    if math.isclose(_float(row, "temperature"), temperature)
                    and int(_float(row, "replicate")) == replicate
                    and row["kernel_model"] == model
                ]
                means[model] = math.fsum(
                    _float(row, f"absolute_error_fs_{wave_key}")
                    for row in local
                    for wave_key in WAVE_KEYS
                ) / (len(local) * len(WAVE_KEYS))
            improved_restart_labels += int(
                means["lag_conditioned"] < means["lag_pooled"]
            )
        uniform_restart_improvement = (
            improved_restart_labels == len(EXPECTED_GRIDS[temperature])
        )
        verdicts.append(
            {
                "temperature": temperature,
                "analysis_status": (
                    "low_temperature_retrospective_ensemble_canary_only"
                    if low_canary
                    else "high_temperature_cage_event_representation_rejected"
                ),
                "retrospective_ensemble_canary_pass": float(low_canary),
                "replicate_level_transfer_pass": _float(calibration, "replicate_curve_pass"),
                "heldout_oracle_factorization_pass": float(oracle_pass),
                "previous_global_hybrid_joint_pass": previous_joint_pass,
                "joint_empirical_interval_closure_repairs_old_hybrid_canary": float(
                    low_canary and not bool(previous_joint_pass)
                ),
                "lag_pooled_kernel_ensemble_curve_pass": _float(
                    lag_pooled, "ensemble_curve_pass"
                ),
                "lag_pooled_kernel_derived_scalar_pass": _float(
                    lag_pooled, "derived_scalar_pass"
                ),
                "lag_conditioning_required_for_frozen_multik_gate": float(
                    lag_required_for_multik
                ),
                "lag_pooled_maximum_ensemble_msd_relative_error": _float(
                    lag_pooled, "maximum_ensemble_msd_relative_error"
                ),
                "lag_pooled_maximum_ensemble_ngp_absolute_error": _float(
                    lag_pooled, "maximum_ensemble_ngp_absolute_error"
                ),
                "lag_pooled_maximum_ensemble_fs_absolute_error": pooled_fs_error,
                "lag_conditioning_maximum_fs_error_reduction_fraction": (
                    (pooled_fs_error - conditioned_fs_error) / pooled_fs_error
                    if pooled_fs_error > 0.0
                    else 0.0
                ),
                "restart_labels_with_lower_mean_fs_error": float(
                    improved_restart_labels
                ),
                "uniform_restart_mean_fs_improvement": float(
                    uniform_restart_improvement
                ),
                "lag_conditioning_generalization_claim_allowed": 0.0,
                "calibration_maximum_ensemble_msd_relative_error": _float(
                    calibration, "maximum_ensemble_msd_relative_error"
                ),
                "calibration_maximum_ensemble_ngp_absolute_error": _float(
                    calibration, "maximum_ensemble_ngp_absolute_error"
                ),
                "calibration_maximum_ensemble_fs_absolute_error": _float(
                    calibration, "maximum_ensemble_fs_absolute_error"
                ),
                "oracle_maximum_ensemble_msd_relative_error": _float(
                    oracle, "maximum_ensemble_msd_relative_error"
                ),
                "oracle_maximum_ensemble_ngp_absolute_error": _float(
                    oracle, "maximum_ensemble_ngp_absolute_error"
                ),
                "oracle_maximum_ensemble_fs_absolute_error": _float(
                    oracle, "maximum_ensemble_fs_absolute_error"
                ),
                "source_doi": local_provenance["source_doi"],
                "source_dataset": local_provenance["source_dataset"],
                "restart_replicate_count": _float(local_provenance, "replicate_count"),
                "independent_parent_sample_count": parent_count,
                "independent_parent_sufficiency_pass": float(parent_sufficient),
                "calibration_only_event_count_marginal": 1.0,
                "calibration_only_lag_conditioned_path_kernel": 1.0,
                "calibration_only_cage_residual": 1.0,
                "macro_fit_parameter_count": 0.0,
                "retrospective_analysis": 1.0,
                "preregistered_prediction_claim_allowed": 0.0,
                "finite_memory_parametric_claim_allowed": 0.0,
                "universal_cage_event_representation_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
                "next_required_test": (
                    "independent_parent_preregistered_interval_conditioned_transfer"
                    if temperature == 0.45
                    else "replace_high_temperature_cage_event_representation"
                ),
            }
        )
    return verdicts


def analyze_committed_tables(
    root: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    data = root / "data"
    rows: list[dict[str, object]] = read_rows(
        data / "renewal_cage_ka_interval_conditioned_event_transfer_rows.csv"
    )
    provenance: list[dict[str, object]] = read_rows(
        data / "renewal_cage_ka_interval_conditioned_event_transfer_provenance.csv"
    )
    ablation_rows: list[dict[str, object]] = read_rows(
        data
        / "renewal_cage_ka_interval_conditioned_event_transfer_kernel_ablation_rows.csv"
    )
    curves, summaries, attributions = summarize_transfer_rows(rows)
    ablation_curves, ablation_summaries = summarize_kernel_ablation_rows(ablation_rows)
    old_hybrid: list[dict[str, object]] = read_rows(
        data / "renewal_cage_ka_replicates_T045_hybrid_macro_verdict.csv"
    )
    verdicts = build_verdicts(
        summaries, ablation_summaries, ablation_rows, provenance, old_hybrid
    )
    return (
        rows,
        curves,
        summaries,
        verdicts,
        attributions,
        ablation_rows,
        ablation_curves,
        ablation_summaries,
    )


def write_svg(
    path: Path,
    curves: list[dict[str, object]],
    summaries: list[dict[str, object]],
    verdicts: list[dict[str, object]],
    attributions: list[dict[str, object]],
    ablation_curves: list[dict[str, object]],
    ablation_summaries: list[dict[str, object]],
) -> None:
    del attributions
    low = _summary_by_code(summaries, 0.45, "ccc")
    low_pooled = _summary_by_kernel(ablation_summaries, 0.45, "lag_pooled")
    high_oracle = _summary_by_code(summaries, 0.58, "hhh")
    low_verdict = next(row for row in verdicts if math.isclose(_float(row, "temperature"), 0.45))
    low_curves = [
        row
        for row in curves
        if math.isclose(_float(row, "temperature"), 0.45) and row["model_code"] == "ccc"
    ]
    low_pooled_curves = [
        row
        for row in ablation_curves
        if math.isclose(_float(row, "temperature"), 0.45)
        and row["kernel_model"] == "lag_pooled"
    ]
    width, height = 1200, 720
    plot_left, plot_top, plot_width, plot_height = 680, 155, 450, 340
    x_min = math.log10(min(_float(row, "lag") for row in low_curves))
    x_max = math.log10(max(_float(row, "lag") for row in low_curves))
    def x_position(lag: float) -> float:
        return plot_left + plot_width * (math.log10(lag) - x_min) / (x_max - x_min)
    def y_position(value: float) -> float:
        return plot_top + plot_height * (1.0 - min(1.0, max(0.0, value)))
    observed_points = " ".join(
        f"{x_position(_float(row, 'lag')):.1f},{y_position(_float(row, 'observed_fs_k7p25')):.1f}"
        for row in low_curves
    )
    predicted_points = " ".join(
        f"{x_position(_float(row, 'lag')):.1f},{y_position(_float(row, 'predicted_fs_k7p25')):.1f}"
        for row in low_curves
    )
    pooled_points = " ".join(
        f"{x_position(_float(row, 'lag')):.1f},{y_position(_float(row, 'predicted_fs_k7p25')):.1f}"
        for row in low_pooled_curves
    )
    metric_lines = (
        ("MSD", _float(low, "maximum_ensemble_msd_relative_error"), MSD_TOLERANCE),
        ("NGP", _float(low, "maximum_ensemble_ngp_absolute_error"), NGP_TOLERANCE),
        ("multi-k Fs", _float(low, "maximum_ensemble_fs_absolute_error"), FS_TOLERANCE),
    )
    metric_svg = []
    for index, (label, value, tolerance) in enumerate(metric_lines):
        y = 178 + 62 * index
        ratio = min(1.5, value / tolerance)
        metric_svg.append(
            f'<text x="75" y="{y}" font-family="Arial, sans-serif" font-size="16" fill="#202124">{html.escape(label)}</text>'
            f'<rect x="190" y="{y - 16}" width="300" height="18" fill="#e8eaed"/>'
            f'<rect x="190" y="{y - 16}" width="{200 * ratio:.1f}" height="18" fill="#2e7d5b"/>'
            f'<line x1="390" y1="{y - 20}" x2="390" y2="{y + 6}" stroke="#202124" stroke-width="2"/>'
            f'<text x="505" y="{y}" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">{value:.3f} / {tolerance:.2f}</text>'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="1200" height="720" fill="#ffffff"/>
<text x="60" y="52" font-family="Arial, sans-serif" font-size="28" font-weight="700" fill="#202124">Lag-conditioned event transfer</text>
<text x="60" y="82" font-family="Arial, sans-serif" font-size="15" fill="#5f6368">Retrospective Obadiya-Sussman KA diagnostic; held-out errors, no macro fit.</text>
<rect x="55" y="110" width="525" height="430" rx="6" fill="#f8f9fa" stroke="#dadce0"/>
<text x="75" y="145" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#202124">T = 0.45 ensemble canary</text>
{''.join(metric_svg)}
<text x="75" y="382" font-family="Arial, sans-serif" font-size="15" fill="#202124">replicate transfer: fail</text>
<text x="75" y="412" font-family="Arial, sans-serif" font-size="15" fill="#202124">independent parents: {int(_float(low_verdict, 'independent_parent_sample_count'))}</text>
<text x="75" y="442" font-family="Arial, sans-serif" font-size="15" fill="#202124">high-T oracle factorization: fail</text>
<text x="75" y="472" font-family="Arial, sans-serif" font-size="14" fill="#9b3a32">lag-pooled K(n): fail, max Fs error {_float(low_pooled, 'maximum_ensemble_fs_absolute_error'):.3f}</text>
<text x="75" y="498" font-family="Arial, sans-serif" font-size="14" fill="#2e7d5b">lag-conditioned Kt(n): pass, max Fs error {_float(low, 'maximum_ensemble_fs_absolute_error'):.3f}</text>
<text x="75" y="524" font-family="Arial, sans-serif" font-size="13" fill="#5f6368">T = 0.58 oracle MSD/NGP/Fs: {_float(high_oracle, 'maximum_ensemble_msd_relative_error'):.3f} / {_float(high_oracle, 'maximum_ensemble_ngp_absolute_error'):.3f} / {_float(high_oracle, 'maximum_ensemble_fs_absolute_error'):.3f}</text>
<text x="680" y="135" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#202124">Held-out F_s(k=7.25,t), T = 0.45</text>
<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#dadce0"/>
<line x1="{plot_left}" y1="{y_position(math.exp(-1)):.1f}" x2="{plot_left + plot_width}" y2="{y_position(math.exp(-1)):.1f}" stroke="#9aa0a6" stroke-dasharray="6 5"/>
<polyline points="{observed_points}" fill="none" stroke="#202124" stroke-width="3"/>
<polyline points="{pooled_points}" fill="none" stroke="#6f42c1" stroke-width="2.5" stroke-dasharray="7 5"/>
<polyline points="{predicted_points}" fill="none" stroke="#d55e00" stroke-width="3"/>
<text x="{plot_left}" y="530" font-family="Arial, sans-serif" font-size="14" fill="#202124">20</text>
<text x="{plot_left + plot_width - 42}" y="530" font-family="Arial, sans-serif" font-size="14" fill="#202124">4096</text>
<text x="{plot_left + 95}" y="555" font-family="Arial, sans-serif" font-size="14" fill="#202124">black: held-out observation</text>
<text x="{plot_left + 95}" y="578" font-family="Arial, sans-serif" font-size="14" fill="#6f42c1">purple dashed: lag-pooled K(n)</text>
<text x="{plot_left + 95}" y="601" font-family="Arial, sans-serif" font-size="14" fill="#d55e00">orange: lag-conditioned Kt(n)</text>
<rect x="55" y="575" width="525" height="110" rx="6" fill="#fff7e6" stroke="#e6b85c"/>
<text x="75" y="610" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#5b3b00">Claim boundary</text>
<text x="75" y="634" font-family="Arial, sans-serif" font-size="13" fill="#5b3b00">Positive ensemble canary; not an independent-parent forecast or finite-memory model.</text>
<text x="75" y="656" font-family="Arial, sans-serif" font-size="13" fill="#5b3b00">No universal cage-event representation, spatial-facilitation theory,</text>
<text x="75" y="678" font-family="Arial, sans-serif" font-size="13" fill="#5b3b00">or thermodynamic theory is claimed.</text>
</svg>''',
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    (
        rows,
        curves,
        summaries,
        verdicts,
        attributions,
        _,
        ablation_curves,
        ablation_summaries,
    ) = analyze_committed_tables(arguments.root)
    data = arguments.root / "data"
    write_rows(data / "renewal_cage_ka_interval_conditioned_event_transfer_ensemble_curves.csv", curves)
    write_rows(data / "renewal_cage_ka_interval_conditioned_event_transfer_summary.csv", summaries)
    write_rows(data / "renewal_cage_ka_interval_conditioned_event_transfer_verdict.csv", verdicts)
    write_rows(data / "renewal_cage_ka_interval_conditioned_event_transfer_shapley.csv", attributions)
    write_rows(
        data
        / "renewal_cage_ka_interval_conditioned_event_transfer_kernel_ablation_curves.csv",
        ablation_curves,
    )
    write_rows(
        data
        / "renewal_cage_ka_interval_conditioned_event_transfer_kernel_ablation_summary.csv",
        ablation_summaries,
    )
    write_svg(
        arguments.root / "figures" / "renewal_cage_ka_interval_conditioned_event_transfer.svg",
        curves,
        summaries,
        verdicts,
        attributions,
        ablation_curves,
        ablation_summaries,
    )
    print(f"validated {len(rows)} three-channel rows")


if __name__ == "__main__":
    main()
