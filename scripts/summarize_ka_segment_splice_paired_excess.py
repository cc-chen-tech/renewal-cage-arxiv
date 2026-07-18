#!/usr/bin/env python3
"""Center finite segment-splice losses on replicate-resolved full-path controls."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path


MODELS = (
    "within_particle_segment_shuffle",
    "cross_particle_segment_splice",
)
LENGTHS = (1, 2, 5, 10, 25, 50, 125, 250)
REPLICATES = (1, 2, 3)
T95_DF2 = 4.302652729911275
CSV_FLOAT_SIGNIFICANT_DIGITS = 15
INDEPENDENCE_CLASS = "decorrelated_parent_frames_plus_velocity_seeds"
SOURCE_CLAIM_FIELDS = (
    "microdynamic_closure_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)
SOURCE_VERDICT_CLOSED_FIELDS = (
    "independent_replicate_memory_lower_bound_claim_allowed",
    "owner_identity_sufficiency_claim_allowed",
    "static_vs_finite_exchange_resolved",
)
CLOSED_CLAIM_FIELDS = (
    "paired_excess_equivalence_claim_allowed",
    "independent_replicate_memory_lower_bound_claim_allowed",
    "finite_memory_state_addition_allowed",
    "owner_identity_sufficiency_claim_allowed",
    "replicate_first_interval_independence_claim_allowed",
    "microdynamic_closure_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)


class SourceVerdictMismatch(ValueError):
    """Raised when a stored source verdict disagrees with score recomputation."""


def _finite(row: dict[str, object], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid field: {field}") from error
    if not math.isfinite(value):
        raise ValueError(f"field must be finite: {field}")
    return value


def _exact_int(row: dict[str, object], field: str) -> int:
    value = _finite(row, field)
    if not value.is_integer():
        raise ValueError(f"field must be an exact integer: {field}")
    return int(value)


def _boolean(row: dict[str, object], field: str) -> bool:
    try:
        value = row[field]
    except KeyError as error:
        raise ValueError(f"missing or invalid field: {field}") from error
    if value in (False, 0, 0.0, "False", "false", "0", "0.0"):
        return False
    if value in (True, 1, 1.0, "True", "true", "1", "1.0"):
        return True
    raise ValueError(f"missing or invalid field: {field}")


def canonical_csv_value(value: object) -> object:
    """Serialize floats stably across platform-level last-bit drift."""

    if not isinstance(value, float):
        return value
    if not math.isfinite(value):
        raise ValueError("CSV output values must be finite")
    return format(value, f".{CSV_FLOAT_SIGNIFICANT_DIGITS}g")


def _mean_interval(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != len(REPLICATES) or any(not math.isfinite(value) for value in values):
        raise ValueError("paired intervals require three finite replicate values")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance / len(values))
    return (
        mean,
        standard_error,
        mean - T95_DF2 * standard_error,
        mean + T95_DF2 * standard_error,
    )


def _validate_claims(rows: Sequence[dict[str, object]]) -> None:
    if any(_finite(row, field) != 0.0 for row in rows for field in SOURCE_CLAIM_FIELDS):
        raise ValueError("source claim boundaries must remain closed")


def _score_lookup(rows: Sequence[dict[str, object]]) -> dict[tuple[str, int, int], float]:
    expected = {
        (model, length, replicate)
        for model in MODELS
        for length in LENGTHS
        for replicate in REPLICATES
    }
    keys = []
    values = []
    for row in rows:
        if _finite(row, "temperature") != 0.45:
            raise ValueError("paired excess accepts only the frozen T=0.45 table")
        key = (
            str(row.get("model")),
            _exact_int(row, "segment_length"),
            _exact_int(row, "replicate"),
        )
        keys.append(key)
        score = _finite(row, "higher_order_score")
        if score < 0.0:
            raise ValueError("higher-order scores must be nonnegative")
        values.append(score)
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("replicate scores do not contain the exact frozen grid")
    _validate_claims(rows)
    return dict(zip(keys, values))


def _validate_cells(rows: Sequence[dict[str, object]]) -> None:
    expected = {(model, length) for model in MODELS for length in LENGTHS}
    keys = []
    for row in rows:
        if _finite(row, "temperature") != 0.45:
            raise ValueError("paired excess accepts only T=0.45 cells")
        keys.append((str(row.get("model")), _exact_int(row, "segment_length")))
        if _finite(row, "global_source_segment_schedule_preserved") != 1.0:
            raise ValueError("global source-segment schedule boundary is not preserved")
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("cell rows do not contain the exact frozen grid")
    _validate_claims(rows)


def _validate_source_verdict(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 1:
        raise ValueError("source verdict must contain exactly one row")
    verdict = rows[0]
    if verdict.get("mechanism_state") != "mechanism_unresolved":
        raise ValueError("paired analysis requires the unresolved source mechanism state")
    if _finite(verdict, "low_mechanism_identifiable_against_full_path_control") != 0.0:
        raise ValueError("source full-path mechanism gate must remain unresolved")
    if _finite(verdict, "global_source_segment_schedule_preserved") != 1.0:
        raise ValueError("source schedule claim boundary is not preserved")
    _validate_claims(rows)
    if any(_finite(verdict, field) != 0.0 for field in SOURCE_VERDICT_CLOSED_FIELDS):
        raise ValueError("source mechanism claim boundaries must remain closed")
    return verdict


def _validate_provenance(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    selected = [row for row in rows if _finite(row, "temperature") == 0.45]
    keys = [_exact_int(row, "replicate") for row in selected]
    if len(keys) != len(REPLICATES) or len(set(keys)) != len(keys):
        raise ValueError("provenance does not contain exactly three T=0.45 restarts")
    if set(keys) != set(REPLICATES):
        raise ValueError("provenance replicate labels do not match the frozen grid")
    source_hashes = {str(row.get("source_sha256", "")) for row in selected}
    if len(source_hashes) != 1 or not next(iter(source_hashes)):
        raise ValueError("T=0.45 restarts must share one identified parent sample")
    frames = [_exact_int(row, "source_frame_index") for row in selected]
    seeds = [_exact_int(row, "velocity_seed") for row in selected]
    if len(set(frames)) != len(REPLICATES) or len(set(seeds)) != len(REPLICATES):
        raise ValueError("restart frames and velocity seeds must be distinct")
    if any(str(row.get("independence_class")) != INDEPENDENCE_CLASS for row in selected):
        raise ValueError("restart independence class does not match provenance")
    if any(_boolean(row, "independently_prepared_parent_samples") for row in selected):
        raise ValueError("paired excess requires the recorded correlated-parent provenance")
    if any(_boolean(row, "thermodynamic_claim_allowed") for row in selected):
        raise ValueError("provenance thermodynamic claim boundary must remain closed")
    return {
        "replicate_provenance_validation_pass": 1.0,
        "replicate_count": float(len(REPLICATES)),
        "parent_sample_count": 1.0,
        "independent_replicate_count": 0.0,
        "independently_prepared_parent_samples": 0.0,
        "independence_class": INDEPENDENCE_CLASS,
    }


def _full_path_baselines(
    scores: dict[tuple[str, int, int], float],
) -> tuple[dict[int, float], float]:
    baselines: dict[int, float] = {}
    differences: list[float] = []
    for replicate in REPLICATES:
        within = scores[(MODELS[0], LENGTHS[-1], replicate)]
        cross = scores[(MODELS[1], LENGTHS[-1], replicate)]
        difference = abs(within - cross)
        differences.append(difference)
        if not math.isclose(within, cross, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("within and cross full-path controls disagree")
        baselines[replicate] = 0.5 * (within + cross)
    return baselines, max(differences)


def _full_path_replicate_summary(
    baselines: dict[int, float],
) -> tuple[float, float]:
    failed_count = sum(baselines[replicate] > 1.0 for replicate in REPLICATES)
    return float(failed_count == 0), float(failed_count)


def _validate_source_full_path_summary(
    source: dict[str, object],
    all_replicates_pass: float,
    failed_replicate_count: float,
) -> None:
    stored_pass = _finite(source, "low_full_path_control_all_replicates_pass")
    stored_failed = _finite(source, "low_full_path_control_failed_replicate_count")
    if stored_pass != all_replicates_pass or stored_failed != failed_replicate_count:
        raise SourceVerdictMismatch(
            "source full-path replicate verdict disagrees with score recomputation"
        )


def _contiguous_prefix(lengths: Sequence[int]) -> bool:
    selected = tuple(sorted(lengths))
    return bool(selected) and selected == LENGTHS[: len(selected)]


def _classify_paired_excess_gate_strict(
    replicate_scores: Sequence[dict[str, object]],
    cells: Sequence[dict[str, object]],
    source_verdict: Sequence[dict[str, object]],
    provenance: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return finite-length paired contrasts and a fail-closed exploratory gate."""

    scores = _score_lookup(replicate_scores)
    _validate_cells(cells)
    source = _validate_source_verdict(source_verdict)
    provenance_summary = _validate_provenance(provenance)
    baselines, full_path_max_difference = _full_path_baselines(scores)
    full_path_all_pass, full_path_failed_count = _full_path_replicate_summary(
        baselines
    )
    _validate_source_full_path_summary(
        source,
        full_path_all_pass,
        full_path_failed_count,
    )

    rows: list[dict[str, object]] = []
    identified: dict[str, list[int]] = {model: [] for model in MODELS}
    model_replicate_grid_means: dict[str, list[float]] = {model: [] for model in MODELS}
    for model in MODELS:
        by_replicate = {replicate: [] for replicate in REPLICATES}
        for length in LENGTHS[:-1]:
            excess = [
                scores[(model, length, replicate)] - baselines[replicate]
                for replicate in REPLICATES
            ]
            for replicate, value in zip(REPLICATES, excess):
                by_replicate[replicate].append(value)
            mean, standard_error, ci_low, ci_high = _mean_interval(excess)
            degradation = float(ci_low > 0.0)
            if degradation:
                identified[model].append(length)
            rows.append(
                {
                    "temperature": 0.45,
                    "model": model,
                    "segment_length": float(length),
                    "tau_L": float(20 * length),
                    "replicate_1_full_path_baseline": baselines[1],
                    "replicate_2_full_path_baseline": baselines[2],
                    "replicate_3_full_path_baseline": baselines[3],
                    "replicate_1_paired_excess": excess[0],
                    "replicate_2_paired_excess": excess[1],
                    "replicate_3_paired_excess": excess[2],
                    "replicate_1_excess": excess[0],
                    "replicate_2_excess": excess[1],
                    "replicate_3_excess": excess[2],
                    "replicate_first_mean_paired_excess": mean,
                    "mean_paired_excess_score": mean,
                    "replicate_first_standard_error": standard_error,
                    "replicate_first_t95_ci_low": ci_low,
                    "replicate_first_t95_ci_high": ci_high,
                    "paired_excess_t95_ci_low": ci_low,
                    "paired_excess_t95_ci_high": ci_high,
                    "paired_degradation_identified": degradation,
                    "independent_replicate_count": 0.0,
                    "replicate_count": 3.0,
                    "parent_sample_count": 1.0,
                    "independently_prepared_parent_samples": 0.0,
                    "independence_class": INDEPENDENCE_CLASS,
                    "replicate_provenance_validation_pass": 1.0,
                    "replicate_first_interval_independence_claim_allowed": 0.0,
                    "full_path_baseline_max_model_difference": full_path_max_difference,
                    "ci95_method": "student_t_replicate_first_correlated_parent_exploratory_df2",
                    "post_run_exploratory": 1.0,
                    "global_source_segment_schedule_preserved": 1.0,
                    "paired_excess_equivalence_claim_allowed": 0.0,
                    "microdynamic_closure_claim_allowed": 0.0,
                    "spatial_facilitation_claim_allowed": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
        model_replicate_grid_means[model] = [
            sum(by_replicate[replicate]) / len(LENGTHS[:-1])
            for replicate in REPLICATES
        ]

    prefix_pass = all(_contiguous_prefix(identified[model]) for model in MODELS)
    common_prefix_max = min(max(identified[model]) for model in MODELS) if prefix_pass else 0
    within_grid = _mean_interval(model_replicate_grid_means[MODELS[0]])
    cross_grid = _mean_interval(model_replicate_grid_means[MODELS[1]])

    owner_values: dict[int, list[float]] = {replicate: [] for replicate in REPLICATES}
    ordering_count = 0
    for replicate in REPLICATES:
        for length in LENGTHS[:-1]:
            contrast = scores[(MODELS[1], length, replicate)] - scores[(MODELS[0], length, replicate)]
            owner_values[replicate].append(contrast)
            ordering_count += int(contrast > 0.0)
    owner_replicate_means = [
        sum(owner_values[replicate]) / len(LENGTHS[:-1]) for replicate in REPLICATES
    ]
    owner_mean, owner_se, owner_ci_low, owner_ci_high = _mean_interval(owner_replicate_means)
    owner_supported = ordering_count == 21 and owner_ci_low > 0.0

    gate: dict[str, object] = {
        "mechanism_state": "mechanism_unresolved",
        "analysis_status": "post_run_exploratory",
        "substantive_interpretation_condition": "conditional_on_preserved_global_source_segment_schedule_and_correlated_parent_restart_labels",
        "input_completeness_pass": 1.0,
        "paired_input_exactness_pass": 1.0,
        "source_verdict_fail_closed": 0.0,
        "full_path_model_agreement_pass": 1.0,
        "full_path_baseline_max_model_difference": full_path_max_difference,
        "low_full_path_control_all_replicates_pass": full_path_all_pass,
        "full_path_replicate_baseline_pass": full_path_all_pass,
        "low_full_path_control_failed_replicate_count": full_path_failed_count,
        **provenance_summary,
        "within_identified_prefix_max_segment_length": float(
            max(identified[MODELS[0]]) if _contiguous_prefix(identified[MODELS[0]]) else 0
        ),
        "cross_identified_prefix_max_segment_length": float(
            max(identified[MODELS[1]]) if _contiguous_prefix(identified[MODELS[1]]) else 0
        ),
        "identified_prefix_max_segment_length": float(common_prefix_max),
        "identified_prefix_max_tau": float(20 * common_prefix_max),
        "short_horizon_information_loss_supported_exploratory": float(prefix_pass),
        "within_frozen_grid_mean_paired_excess": within_grid[0],
        "within_frozen_grid_mean_standard_error": within_grid[1],
        "within_frozen_grid_mean_t95_ci_low": within_grid[2],
        "within_frozen_grid_mean_t95_ci_high": within_grid[3],
        "cross_frozen_grid_mean_paired_excess": cross_grid[0],
        "cross_frozen_grid_mean_standard_error": cross_grid[1],
        "cross_frozen_grid_mean_t95_ci_low": cross_grid[2],
        "cross_frozen_grid_mean_t95_ci_high": cross_grid[3],
        "owner_identity_paired_ordering_count": float(ordering_count),
        "owner_identity_paired_ordering_total": 21.0,
        "owner_identity_replicate_first_mean_score_difference": owner_mean,
        "owner_identity_replicate_first_standard_error": owner_se,
        "owner_identity_replicate_first_t95_ci_low": owner_ci_low,
        "owner_identity_replicate_first_t95_ci_high": owner_ci_high,
        "owner_identity_information_supported_exploratory": float(owner_supported),
        "next_required_action": "replicate_resolved_full_path_baseline_or_new_trajectory_validation",
    }
    gate.update({field: 0.0 for field in CLOSED_CLAIM_FIELDS})
    return rows, gate


def _failed_gate(
    *,
    source_verdict_fail_closed: bool,
    input_completeness_pass: bool,
) -> dict[str, object]:
    gate: dict[str, object] = {
        "mechanism_state": "mechanism_unresolved",
        "analysis_status": "post_run_exploratory",
        "substantive_interpretation_condition": "conditional_on_preserved_global_source_segment_schedule_and_correlated_parent_restart_labels",
        "input_completeness_pass": float(input_completeness_pass),
        "paired_input_exactness_pass": 0.0,
        "source_verdict_fail_closed": float(source_verdict_fail_closed),
        "full_path_model_agreement_pass": 0.0,
        "full_path_baseline_max_model_difference": 0.0,
        "low_full_path_control_all_replicates_pass": 0.0,
        "full_path_replicate_baseline_pass": 0.0,
        "low_full_path_control_failed_replicate_count": 0.0,
        "replicate_provenance_validation_pass": 0.0,
        "replicate_count": 0.0,
        "parent_sample_count": 0.0,
        "independent_replicate_count": 0.0,
        "independently_prepared_parent_samples": 0.0,
        "independence_class": "unresolved",
        "within_identified_prefix_max_segment_length": 0.0,
        "cross_identified_prefix_max_segment_length": 0.0,
        "identified_prefix_max_segment_length": 0.0,
        "identified_prefix_max_tau": 0.0,
        "short_horizon_information_loss_supported_exploratory": 0.0,
        "within_frozen_grid_mean_paired_excess": 0.0,
        "within_frozen_grid_mean_standard_error": 0.0,
        "within_frozen_grid_mean_t95_ci_low": 0.0,
        "within_frozen_grid_mean_t95_ci_high": 0.0,
        "cross_frozen_grid_mean_paired_excess": 0.0,
        "cross_frozen_grid_mean_standard_error": 0.0,
        "cross_frozen_grid_mean_t95_ci_low": 0.0,
        "cross_frozen_grid_mean_t95_ci_high": 0.0,
        "owner_identity_paired_ordering_count": 0.0,
        "owner_identity_paired_ordering_total": 21.0,
        "owner_identity_replicate_first_mean_score_difference": 0.0,
        "owner_identity_replicate_first_standard_error": 0.0,
        "owner_identity_replicate_first_t95_ci_low": 0.0,
        "owner_identity_replicate_first_t95_ci_high": 0.0,
        "owner_identity_information_supported_exploratory": 0.0,
        "next_required_action": "replicate_resolved_full_path_baseline_or_new_trajectory_validation",
    }
    gate.update({field: 0.0 for field in CLOSED_CLAIM_FIELDS})
    return gate


def classify_paired_excess_gate(
    replicate_scores: Sequence[dict[str, object]],
    cells: Sequence[dict[str, object]],
    source_verdict: Sequence[dict[str, object]],
    provenance: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Fail closed when paired support or the unresolved source verdict is invalid."""

    try:
        _validate_source_verdict(source_verdict)
        source_failed = False
    except (KeyError, TypeError, ValueError):
        source_failed = True
        return [], _failed_gate(
            source_verdict_fail_closed=True,
            input_completeness_pass=False,
        )
    try:
        _validate_provenance(provenance)
    except (KeyError, TypeError, ValueError):
        return [], _failed_gate(
            source_verdict_fail_closed=False,
            input_completeness_pass=False,
        )
    input_complete = False
    try:
        _score_lookup(replicate_scores)
        _validate_cells(cells)
        input_complete = True
        return _classify_paired_excess_gate_strict(
            replicate_scores,
            cells,
            source_verdict,
            provenance,
        )
    except SourceVerdictMismatch:
        return [], _failed_gate(
            source_verdict_fail_closed=True,
            input_completeness_pass=input_complete,
        )
    except (KeyError, TypeError, ValueError):
        return [], _failed_gate(
            source_verdict_fail_closed=source_failed,
            input_completeness_pass=input_complete,
        )


def compute_paired_excess_rows(
    replicate_scores: Sequence[dict[str, object]],
    provenance: Sequence[dict[str, object]],
    *,
    block_size: int = 20,
) -> list[dict[str, object]]:
    """Compute the 14 exact paired rows, raising on malformed score support."""

    if isinstance(block_size, bool) or block_size != 20:
        raise ValueError("the frozen paired-excess block size is 20")
    cells = [
        {
            "temperature": 0.45,
            "model": model,
            "segment_length": float(length),
            "global_source_segment_schedule_preserved": 1.0,
            **{field: 0.0 for field in SOURCE_CLAIM_FIELDS},
            **{field: 0.0 for field in SOURCE_VERDICT_CLOSED_FIELDS},
        }
        for model in MODELS
        for length in LENGTHS
    ]
    scores = _score_lookup(replicate_scores)
    baselines, _ = _full_path_baselines(scores)
    all_pass, failed_count = _full_path_replicate_summary(baselines)
    source = [
        {
            "mechanism_state": "mechanism_unresolved",
            "low_mechanism_identifiable_against_full_path_control": 0.0,
            "low_full_path_control_all_replicates_pass": all_pass,
            "low_full_path_control_failed_replicate_count": failed_count,
            "global_source_segment_schedule_preserved": 1.0,
            **{field: 0.0 for field in SOURCE_CLAIM_FIELDS},
            **{field: 0.0 for field in SOURCE_VERDICT_CLOSED_FIELDS},
        }
    ]
    rows, _ = _classify_paired_excess_gate_strict(
        replicate_scores,
        cells,
        source,
        provenance,
    )
    return rows


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input table is empty: {path}")
    return rows


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty paired-excess output")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {
                field: canonical_csv_value(value)
                for field, value in row.items()
            }
            for row in rows
        )


def write_paired_excess_svg(
    path: Path,
    rows: Sequence[dict[str, object]],
    gate: dict[str, object],
) -> None:
    """Write a deterministic paired-excess plot with replicate-level t intervals."""

    expected = {(model, length) for model in MODELS for length in LENGTHS[:-1]}
    keyed = {
        (str(row.get("model")), int(_finite(row, "segment_length"))): row for row in rows
    }
    if len(rows) != len(expected) or set(keyed) != expected:
        raise ValueError("SVG requires the complete finite paired-excess grid")
    width, height = 1120, 650
    lefts = {MODELS[0]: 90.0, MODELS[1]: 620.0}
    panel_width, panel_height, top = 410.0, 405.0, 105.0
    y_min, y_max = -8.0, 30.0
    colors = {MODELS[0]: "#176b87", MODELS[1]: "#c64b2c"}
    labels = {MODELS[0]: "within-particle segment shuffle", MODELS[1]: "cross-particle segment splice"}
    log_min, log_max = math.log(20.0), math.log(2500.0)

    def x_position(left: float, tau: float) -> float:
        return left + (math.log(tau) - log_min) / (log_max - log_min) * panel_width

    def y_position(value: float) -> float:
        clipped = min(max(value, y_min), y_max)
        return top + (y_max - clipped) / (y_max - y_min) * panel_height

    elements: list[str] = []
    for model in MODELS:
        left = lefts[model]
        zero_y = y_position(0.0)
        elements.extend(
            (
                f'<rect x="{left:.1f}" y="{top:.1f}" width="{panel_width:.1f}" height="{panel_height:.1f}" fill="#ffffff" stroke="#c9c9c3"/>',
                f'<line x1="{left:.1f}" y1="{zero_y:.1f}" x2="{left + panel_width:.1f}" y2="{zero_y:.1f}" stroke="#333333" stroke-width="1.5" stroke-dasharray="7 5"/>',
                f'<text x="{left + 7:.1f}" y="{zero_y - 7:.1f}" font-size="10">zero excess</text>',
                f'<text x="{left + panel_width / 2:.1f}" y="82" text-anchor="middle" font-size="18" font-weight="bold">{labels[model]}</text>',
            )
        )
        points = []
        for length in LENGTHS[:-1]:
            row = keyed[(model, length)]
            tau = _finite(row, "tau_L")
            mean = _finite(row, "replicate_first_mean_paired_excess")
            low = _finite(row, "replicate_first_t95_ci_low")
            high = _finite(row, "replicate_first_t95_ci_high")
            identified = _finite(row, "paired_degradation_identified") == 1.0
            x = x_position(left, tau)
            y = y_position(mean)
            low_y, high_y = y_position(low), y_position(high)
            points.append(f"{x:.2f},{y:.2f}")
            elements.extend(
                (
                    f'<line x1="{x:.2f}" y1="{high_y:.2f}" x2="{x:.2f}" y2="{low_y:.2f}" stroke="{colors[model]}" stroke-width="1.6"/>',
                    f'<line x1="{x - 4:.2f}" y1="{high_y:.2f}" x2="{x + 4:.2f}" y2="{high_y:.2f}" stroke="{colors[model]}"/>',
                    f'<line x1="{x - 4:.2f}" y1="{low_y:.2f}" x2="{x + 4:.2f}" y2="{low_y:.2f}" stroke="{colors[model]}"/>',
                    f'<circle data-identified="{str(identified).lower()}" cx="{x:.2f}" cy="{y:.2f}" r="{5 if identified else 4}" fill="{colors[model] if identified else "#ffffff"}" stroke="{colors[model]}" stroke-width="2"/>',
                )
            )
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[model]}" stroke-width="2"/>'
        )
        for value in (-5, 0, 10, 20, 30):
            y = y_position(float(value))
            elements.append(
                f'<line x1="{left - 5:.1f}" y1="{y:.1f}" x2="{left:.1f}" y2="{y:.1f}" stroke="#555555"/>'
                f'<text x="{left - 9:.1f}" y="{y + 4:.1f}" text-anchor="end" font-size="10">{value}</text>'
            )
        for length in LENGTHS[:-1]:
            x = x_position(left, float(20 * length))
            elements.append(
                f'<text x="{x:.2f}" y="{top + panel_height + 22:.1f}" text-anchor="middle" font-size="10">{20 * length}</text>'
            )
        elements.append(
            f'<text x="{left + panel_width / 2:.1f}" y="{top + panel_height + 50:.1f}" text-anchor="middle">retained horizon tau_L</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="#f5f5f1"/>
<style>text{{font-family:Arial,sans-serif;fill:#202020;font-size:14px;letter-spacing:0}}</style>
<text x="560" y="35" text-anchor="middle" font-size="23" font-weight="bold">Segment-splice paired-excess baseline</text>
<text x="560" y="58" text-anchor="middle" font-size="13">Paired excess over replicate full-path baseline</text>
<text x="28" y="308" text-anchor="middle" font-size="12" transform="rotate(-90 28 308)">higher-order score excess (tolerance units; descriptive t95)</text>
{''.join(elements)}
<circle cx="90" cy="590" r="5" fill="#176b87"/><text x="104" y="595">t95 interval above zero: identified short-horizon degradation</text>
<circle cx="510" cy="590" r="4" fill="#ffffff" stroke="#176b87" stroke-width="2"/><text x="524" y="595">t95 interval crosses zero: unresolved</text>
<text x="560" y="622" text-anchor="middle" font-size="13" font-weight="bold">post-run exploratory; mechanism_unresolved (mechanism unresolved)</text>
<text x="560" y="642" text-anchor="middle" font-size="11">Correlated-parent restart labels; no independent-sample CI, sufficiency, microscopic, spatial, or thermodynamic claim.</text>
</svg>'''
    if "nan" in svg.lower() or "inf" in svg.lower():
        raise ValueError("paired-excess SVG contains nonfinite coordinates")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute post-run paired excess over replicate full-path controls."
    )
    parser.add_argument("--replicate-scores", "--scores", dest="replicate_scores", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--source-verdict", "--source-gate", dest="source_verdict", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows, gate = classify_paired_excess_gate(
        _read_rows(args.replicate_scores),
        _read_rows(args.cells),
        _read_rows(args.source_verdict),
        _read_rows(args.provenance),
    )
    _write_rows(args.output_rows, rows)
    _write_rows(args.output_gate, [gate])
    write_paired_excess_svg(args.output_svg, rows, gate)


if __name__ == "__main__":
    main()
