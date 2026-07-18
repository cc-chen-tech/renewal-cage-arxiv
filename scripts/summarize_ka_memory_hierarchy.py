#!/usr/bin/env python3
"""Select the minimal memory hierarchy supported by frozen KA diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_replicates import summarize_waiting_diagnostic_rows  # noqa: E402
from analyze_ka_waiting_threshold_sensitivity import (  # noqa: E402
    classify_waiting_threshold_sensitivity,
)
from summarize_ka_anchor_semi_markov_gate import (  # noqa: E402
    classify_anchor_semi_markov_gate,
)
from summarize_ka_debye_waller_environment_crossover import (  # noqa: E402
    classify_environment_crossover,
)
from summarize_ka_empirical_path_transfer import (  # noqa: E402
    classify_empirical_path_crossover,
)


def _finite(row: dict[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid field: {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"field must be finite: {key}")
    return value


def _explicit_zero(row: dict[str, object], key: str) -> bool:
    return _finite(row, key) == 0.0


def _claim_boundaries_hold(
    path: dict[str, object],
    anchor: dict[str, object],
    waiting: dict[str, object],
    environment: dict[str, object],
    spatial_rows: Sequence[dict[str, object]],
    spatial_fit: dict[str, object],
) -> bool:
    return (
        all(
            _explicit_zero(row, key)
            for row in (path, anchor)
            for key in (
                "microdynamic_closure_claim_allowed",
                "spatial_facilitation_claim_allowed",
                "thermodynamic_claim_allowed",
            )
        )
        and _explicit_zero(waiting, "temporal_waiting_memory_parameter_claim_allowed")
        and _explicit_zero(waiting, "spatial_cooperation_proven")
        and _explicit_zero(waiting, "thermodynamic_claim_allowed")
        and _explicit_zero(environment, "spatial_facilitation_claim_allowed")
        and _explicit_zero(environment, "thermodynamic_claim_allowed")
        and all(
            _finite(row, "spatial_measurement_claim_allowed") == 1.0
            and _explicit_zero(row, "spatial_model_claim_allowed")
            and _explicit_zero(row, "thermodynamic_claim_allowed")
            for row in tuple(spatial_rows) + (spatial_fit,)
        )
    )


def classify_memory_hierarchy(
    *,
    path: dict[str, object],
    anchor: dict[str, object],
    waiting: dict[str, object],
    environment: dict[str, object],
    spatial_rows: Sequence[dict[str, object]],
    spatial_fit: dict[str, object],
) -> dict[str, object]:
    """Combine ablations without promoting spatial correlations to a cause."""

    result: dict[str, object] = {
        "mechanism_state": "mechanism_hierarchy_unresolved",
        "evidence_completeness_pass": 0.0,
        "selection_scope": "within_frozen_single_particle_alternatives",
        "static_environment_family_excluded": 0.0,
        "cross_particle_mechanism_excluded": 0.0,
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    try:
        boundaries_hold = _claim_boundaries_hold(
            path, anchor, waiting, environment, spatial_rows, spatial_fit
        )
        anchor_provenance = (
            anchor.get("mechanism_state") == "anchor_aware_model_rejected"
            and _finite(anchor, "provenance_and_competitor_completeness_pass") == 1.0
            and _finite(anchor, "low_anchor_aware_semi_markov_curve_transfer_pass")
            == 0.0
            and _finite(anchor, "high_anchor_aware_semi_markov_curve_transfer_pass")
            == 0.0
        )
        distances = sorted(_finite(row, "distance_midpoint") for row in spatial_rows)
        spatial_complete = (
            len(spatial_rows) >= 2
            and len(set(distances)) == len(distances)
            and any(distance <= 3.0 for distance in distances)
            and any(distance >= 4.0 for distance in distances)
            and min(
                _finite(row, "independent_replicate_count") for row in spatial_rows
            )
            >= 3.0
            and _finite(spatial_fit, "independent_replicate_count") >= 3.0
            and spatial_fit.get("fit_status") == "between_replicate_uncertainty"
        )
        pooled_one_step_rejected = (
            _finite(path, "low_temperature_markov_failure") == 1.0
            and _finite(path, "shared_low_temperature_higher_order_failure") == 1.0
        )
        ordered_upper_bound = (
            _finite(path, "low_temperature_contiguous_closure") == 1.0
            and _finite(path, "high_temperature_contiguous_closure") == 1.0
            and _finite(path, "replicate_consensus_pass") == 1.0
        )
        shuffle_precision = _finite(path, "shuffle_precision_pass") == 1.0
        shuffle_closes = (
            _finite(path, "within_particle_time_shuffle_curve_transfer_pass") == 1.0
        )
        if not shuffle_precision:
            identity_state = "unresolved"
        elif shuffle_closes:
            identity_state = "sufficient"
        else:
            identity_state = "rejected"
        ordered_required = (
            _finite(path, "single_particle_multiblock_path_memory_required") == 1.0
            and _finite(path, "ordered_recoil_path_required") == 1.0
            and _finite(path, "amplitude_persistence_alone_sufficient") == 0.0
            and identity_state == "rejected"
        )
        waiting_environment = (
            _finite(waiting, "threshold_robust_dominant_mechanism") == 1.0
            and waiting.get("dominant_mechanism")
            == "mixed_particle_environment_and_event_memory"
            and _finite(
                waiting, "median_window_particle_conditioned_shuffle_sufficient"
            )
            == 1.0
            and _finite(waiting, "persistent_particle_environment_supported") == 1.0
            and _finite(waiting, "finite_exchange_supported_by_prior_identity_decay")
            == 1.0
            and _finite(waiting, "spatial_cooperation_proven") == 0.0
        )
        finite_exchange = (
            waiting_environment
            and _finite(environment, "waiting_mechanism_crossover_detected") == 1.0
            and _finite(environment, "exchange_time_growth_detected_all_block_sizes")
            == 1.0
            and _finite(environment, "minimum_exchange_time_growth_ci95_low") > 1.0
            and _finite(
                environment, "cross_half_identity_correlation_growth_ci95_low"
            )
            > 1.0
            and _finite(environment, "pure_static_particle_rate_disorder_rejected")
            == 1.0
            and _finite(environment, "finite_exchange_environment_claim_allowed")
            == 1.0
            and _finite(environment, "finite_waiting_sequence_memory_required") == 0.0
        )
        complete = (
            boundaries_hold
            and anchor_provenance
            and shuffle_precision
            and spatial_complete
        )
        positive_spatial = any(
            _finite(row, "distance_midpoint") <= 3.0
            and _finite(row, "ci95_low") > 0.0
            and _finite(row, "spatial_measurement_claim_allowed") == 1.0
            for row in spatial_rows
        )
        correlation_length = _finite(spatial_fit, "mean_correlation_length")
        spatial_length_resolved = (
            correlation_length > 0.0
            and _finite(spatial_fit, "ci95_low_correlation_length") > 0.0
        )
    except (ValueError, KeyError, TypeError):
        return result

    result.update(
        {
            "evidence_completeness_pass": float(complete),
            "pooled_one_step_rejected": float(pooled_one_step_rejected),
            "pooled_anchor_semi_markov_rejected": float(anchor_provenance),
            "anchor_model_improves_all_low_replicates": _finite(
                anchor, "all_low_anchor_replicates_improve_over_recoil"
            ),
            "particle_identity_without_order_state": identity_state,
            "particle_identity_without_order_sufficient": float(
                identity_state == "sufficient"
            ),
            "particle_identity_without_order_rejected": float(
                identity_state == "rejected"
            ),
            "ordered_particle_path_upper_bound_closes": float(ordered_upper_bound),
            "ordered_recoil_path_required": float(ordered_required),
            "persistent_particle_environment_supported": float(waiting_environment),
            "finite_exchange_environment_supported": float(finite_exchange),
            "pure_static_particle_rate_disorder_rejected": _finite(
                environment, "pure_static_particle_rate_disorder_rejected"
            ),
            "minimum_temporal_ordering_contribution_fraction": _finite(
                waiting, "minimum_temporal_ordering_contribution_fraction"
            ),
            "minimum_particle_identity_contribution_fraction": _finite(
                waiting, "minimum_particle_identity_contribution_fraction"
            ),
            "minimum_exchange_time_growth_ratio": _finite(
                environment, "minimum_exchange_time_growth_ratio"
            ),
            "minimum_exchange_time_growth_ci95_low": _finite(
                environment, "minimum_exchange_time_growth_ci95_low"
            ),
            "cross_half_identity_correlation_growth_ratio": _finite(
                environment, "cross_half_identity_correlation_growth_ratio"
            ),
            "cross_half_identity_correlation_growth_ci95_low": _finite(
                environment, "cross_half_identity_correlation_growth_ci95_low"
            ),
            "positive_short_range_spatial_covariance_measured": float(
                positive_spatial
            ),
            "spatial_correlation_length": correlation_length,
            "spatial_correlation_length_ci95_low": _finite(
                spatial_fit, "ci95_low_correlation_length"
            ),
            "spatial_correlation_length_ci95_high": _finite(
                spatial_fit, "ci95_high_correlation_length"
            ),
            "spatial_correlation_length_resolved": float(spatial_length_resolved),
            "spatial_facilitation_required": 0.0,
            "next_minimal_model_candidate": "no_model_candidate_selected",
        }
    )
    if not complete:
        return result
    if not (pooled_one_step_rejected and ordered_upper_bound and ordered_required):
        result["mechanism_state"] = "ordered_path_hierarchy_rejected"
        return result
    if not finite_exchange:
        result["mechanism_state"] = "ordered_path_required_environment_unresolved"
        result["next_minimal_model_candidate"] = (
            "particle_conditioned_ordered_path_kernel_environment_unresolved"
        )
        return result
    result["mechanism_state"] = (
        "ordered_particle_path_required_finite_exchange_supported"
    )
    result["next_minimal_model_candidate"] = (
        "particle_conditioned_finite_exchange_ordered_path_kernel"
    )
    return result


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input table is empty: {path}")
    return rows


def _one(path: Path) -> dict[str, str]:
    rows = _read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"expected one row: {path}")
    return rows[0]


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _stored_row_matches(
    stored: dict[str, object], recomputed: dict[str, object]
) -> bool:
    for key, expected in recomputed.items():
        if key not in stored:
            return False
        if isinstance(expected, str):
            if str(stored[key]) != expected:
                return False
        elif not math.isclose(
            _finite(stored, key), float(expected), rel_tol=1e-12, abs_tol=1e-12
        ):
            return False
    return True


def build_evidence_rows(verdict: dict[str, object]) -> list[dict[str, object]]:
    """Expose the ablations in causal order and keep spatial evidence descriptive."""

    one_step_rejected = float(verdict.get("pooled_one_step_rejected", 0.0)) == 1.0
    anchor_rejected = (
        float(verdict.get("pooled_anchor_semi_markov_rejected", 0.0)) == 1.0
    )
    identity_state = str(
        verdict.get("particle_identity_without_order_state", "unresolved")
    )
    ordered_closes = (
        float(verdict.get("ordered_particle_path_upper_bound_closes", 0.0)) == 1.0
    )
    finite_exchange = (
        float(verdict.get("finite_exchange_environment_supported", 0.0)) == 1.0
    )
    positive_spatial = (
        float(verdict.get("positive_short_range_spatial_covariance_measured", 0.0))
        == 1.0
    )
    one_step_interpretation = (
        "one-step recoil statistics are insufficient"
        if one_step_rejected
        else "one-step recoil rejection is not established"
    )
    anchor_interpretation = (
        "two-state anchor geometry is insufficient"
        if anchor_rejected
        else "two-state anchor rejection is not established"
    )
    identity_interpretation = {
        "rejected": "identity alone does not replace path order",
        "sufficient": "identity-preserving shuffled paths are sufficient",
        "unresolved": "identity-without-order test is unresolved",
    }.get(identity_state, "identity-without-order test is unresolved")
    ordered_interpretation = (
        "nonparametric upper bound selects ordered path memory"
        if ordered_closes
        else "ordered-path closure is not established"
    )
    return [
        {
            "evidence_stage": "pooled_one_step_recoil",
            "preserved_information": "one-step recoil conditional",
            "heldout_result": "rejected" if one_step_rejected else "not rejected",
            "evidence_status": float(one_step_rejected),
            "mechanism_claim_allowed": 0.0,
            "interpretation": one_step_interpretation,
        },
        {
            "evidence_stage": "pooled_anchor_semi_markov",
            "preserved_information": "observed anchor state and holding time",
            "heldout_result": "rejected" if anchor_rejected else "not rejected",
            "evidence_status": float(anchor_rejected),
            "mechanism_claim_allowed": 0.0,
            "interpretation": anchor_interpretation,
        },
        {
            "evidence_stage": "particle_identity_without_order",
            "preserved_information": "particle identity and path marginals",
            "heldout_result": identity_state,
            "evidence_status": float(identity_state != "unresolved"),
            "mechanism_claim_allowed": 0.0,
            "interpretation": identity_interpretation,
        },
        {
            "evidence_stage": "ordered_particle_path",
            "preserved_information": "particle identity and ordered multiblock path",
            "heldout_result": (
                "closes both temperatures" if ordered_closes else "does not close"
            ),
            "evidence_status": float(ordered_closes),
            "mechanism_claim_allowed": 0.0,
            "interpretation": ordered_interpretation,
        },
        {
            "evidence_stage": "finite_exchange_environment",
            "preserved_information": "particle mobility identity across finite blocks",
            "heldout_result": "supported" if finite_exchange else "not supported",
            "evidence_status": float(finite_exchange),
            "mechanism_claim_allowed": 0.0,
            "interpretation": (
                "finite exchange is supported within tested rate alternatives; "
                "the static-environment family remains open"
                if finite_exchange
                else "environment mechanism remains unresolved"
            ),
        },
        {
            "evidence_stage": "spatial_covariance_measurement",
            "preserved_information": "cross-particle event-count covariance",
            "heldout_result": (
                "measurement only" if positive_spatial else "no resolved short-range signal"
            ),
            "evidence_status": float(positive_spatial),
            "mechanism_claim_allowed": 0.0,
            "interpretation": "spatial facilitation necessity is not identified",
        },
    ]


def write_svg(path: Path, verdict: dict[str, object]) -> None:
    width, height = 980, 520
    evidence = build_evidence_rows(verdict)
    labels = (
        "One-step recoil",
        "Anchor semi-Markov",
        "Identity, order shuffled",
        "Ordered path (particle)",
    )
    color_by_outcome = {
        "rejected": "#b3261e",
        "closes both temperatures": "#137333",
        "sufficient": "#1967d2",
    }
    display_outcome = {"closes both temperatures": "CLOSES BOTH T"}
    stages = tuple(
        (
            str(index + 1),
            label,
            display_outcome.get(
                str(evidence[index]["heldout_result"]),
                str(evidence[index]["heldout_result"]).upper(),
            ),
            color_by_outcome.get(str(evidence[index]["heldout_result"]), "#5f6368"),
        )
        for index, label in enumerate(labels)
    )
    blocks: list[str] = []
    for index, (number, label, outcome, color) in enumerate(stages):
        x = 35 + 235 * index
        blocks.extend(
            [
                f'<rect x="{x}" y="105" width="205" height="112" rx="6" fill="#ffffff" stroke="#3c4043"/>',
                f'<circle cx="{x + 28}" cy="133" r="16" fill="{color}"/>',
                f'<text x="{x + 28}" y="138" text-anchor="middle" fill="#ffffff" font-weight="700">{number}</text>',
                f'<text x="{x + 102.5}" y="171" text-anchor="middle" font-size="14" font-weight="700">{label}</text>',
                f'<text x="{x + 102.5}" y="199" text-anchor="middle" font-size="13" fill="{color}" font-weight="700">{outcome}</text>',
            ]
        )
        if index < len(stages) - 1:
            blocks.append(
                f'<path d="M {x + 207} 161 L {x + 228} 161" stroke="#5f6368" stroke-width="2" marker-end="url(#arrow)"/>'
            )
    finite_exchange = (
        float(verdict.get("finite_exchange_environment_supported", 0.0)) == 1.0
    )
    positive_spatial = (
        float(verdict.get("positive_short_range_spatial_covariance_measured", 0.0))
        == 1.0
    )
    environment_text = (
        "finite exchange supported; tested pure static rate null rejected"
        if finite_exchange
        else "finite-exchange support not established"
    )
    spatial_text = (
        "positive short-range covariance; spatial cause not identified"
        if positive_spatial
        else "no resolved short-range covariance; spatial cause not identified"
    )
    state = str(verdict["mechanism_state"])
    selection_text = {
        "ordered_particle_path_required_finite_exchange_supported": (
            "Selected next candidate: particle-conditioned finite-exchange ordered-path kernel"
        ),
        "ordered_path_required_environment_unresolved": (
            "Ordered path required; environment mechanism unresolved"
        ),
        "ordered_path_hierarchy_rejected": "Ordered-path hierarchy not selected",
        "mechanism_hierarchy_unresolved": "Mechanism hierarchy unresolved",
    }.get(state, "Mechanism hierarchy unresolved")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f8f9fa"/>
<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="#5f6368"/></marker></defs>
<style>text{{font-family:Arial,sans-serif;letter-spacing:0;fill:#202124}}</style>
<text x="490" y="40" text-anchor="middle" font-size="22" font-weight="700">Information-ablation hierarchy</text>
<text x="490" y="67" text-anchor="middle" font-size="13">Held-out KA dynamics: add information only when the prior level fails</text>
{''.join(blocks)}
<rect x="55" y="270" width="420" height="118" rx="6" fill="#e6f4ea" stroke="#137333"/>
<text x="265" y="300" text-anchor="middle" font-size="16" font-weight="700">Independent environment evidence</text>
<text x="265" y="329" text-anchor="middle" font-size="14">{environment_text}</text>
<text x="265" y="356" text-anchor="middle" font-size="13">exchange growth CI low = {float(verdict['minimum_exchange_time_growth_ci95_low']):.2f}</text>
<text x="265" y="377" text-anchor="middle" font-size="12">static-environment family remains unexcluded</text>
<rect x="505" y="270" width="420" height="118" rx="6" fill="#fef7e0" stroke="#b06000"/>
<text x="715" y="300" text-anchor="middle" font-size="16" font-weight="700">Spatial covariance: measurement only</text>
<text x="715" y="329" text-anchor="middle" font-size="14">{spatial_text}</text>
<text x="715" y="356" text-anchor="middle" font-size="13">decay length = {float(verdict['spatial_correlation_length']):.3f}</text>
<text x="715" y="377" text-anchor="middle" font-size="12">spatial facilitation claim allowed = 0</text>
<text x="490" y="438" text-anchor="middle" font-size="15" font-weight="700">{selection_text}</text>
<text x="490" y="468" text-anchor="middle" font-size="13">Effective dynamical diagnostic; no microdynamic closure, spatial-causation, or thermodynamic-glass claim.</text>
<text x="490" y="496" text-anchor="middle" font-size="11">state: {verdict['mechanism_state']}</text>
</svg>'''
    if ">nan<" in svg.lower() or ">inf<" in svg.lower():
        raise ValueError("hierarchy SVG contains a nonfinite value")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg + "\n")


def recompute_hierarchy(data: Path) -> dict[str, object]:
    low_empirical = _read_rows(
        data / "renewal_cage_ka_replicates_T045_empirical_path_verdict.csv"
    )
    high_empirical = _read_rows(
        data / "renewal_cage_ka_replicates_T058_empirical_path_verdict.csv"
    )
    path = classify_empirical_path_crossover(
        low_empirical,
        high_empirical,
        _one(data / "renewal_cage_ka_replicates_T045_recoil_markov_verdict.csv"),
    )
    prefixes = {
        "low": data / "renewal_cage_ka_replicates_T045_anchor_semi_markov",
        "high": data / "renewal_cage_ka_replicates_T058_anchor_semi_markov",
    }
    anchor = classify_anchor_semi_markov_gate(
        low_quality_rows=_read_rows(
            prefixes["low"].with_name(prefixes["low"].name + "_quality.csv")
        ),
        low_summary_rows=_read_rows(
            prefixes["low"].with_name(prefixes["low"].name + "_summary.csv")
        ),
        low_replicate_rows=_read_rows(
            prefixes["low"].with_name(prefixes["low"].name + "_rows.csv")
        ),
        high_quality_rows=_read_rows(
            prefixes["high"].with_name(prefixes["high"].name + "_quality.csv")
        ),
        high_summary_rows=_read_rows(
            prefixes["high"].with_name(prefixes["high"].name + "_summary.csv")
        ),
        high_replicate_rows=_read_rows(
            prefixes["high"].with_name(prefixes["high"].name + "_rows.csv")
        ),
        low_recoil_rows=_read_rows(
            data / "renewal_cage_ka_replicates_T045_recoil_markov_rows.csv"
        ),
        recoil_verdict_rows=[
            _one(
                data
                / f"renewal_cage_ka_replicates_{label}_recoil_markov_verdict.csv"
            )
            for label in ("T045", "T058")
        ],
        empirical_verdict_rows=[
            row
            for rows in (low_empirical, high_empirical)
            for row in rows
            if row["model"] == "contiguous_empirical_path"
        ],
    )
    _, high_environment_waiting = summarize_waiting_diagnostic_rows(
        _read_rows(
            data / "renewal_cage_ka_replicates_T058_debye_waller_waiting_windows.csv"
        )
    )
    _, low_environment_waiting = summarize_waiting_diagnostic_rows(
        _read_rows(
            data / "renewal_cage_ka_replicates_T045_debye_waller_waiting_windows.csv"
        )
    )
    stored_high_environment_waiting = _one(
        data / "renewal_cage_ka_replicates_T058_debye_waller_waiting_verdict.csv"
    )
    stored_low_environment_waiting = _one(
        data / "renewal_cage_ka_replicates_T045_debye_waller_waiting_verdict.csv"
    )
    high_environment_waiting_consistent = _stored_row_matches(
        stored_high_environment_waiting, high_environment_waiting
    )
    low_environment_waiting_consistent = _stored_row_matches(
        stored_low_environment_waiting, low_environment_waiting
    )
    environment_waiting_consistent = (
        high_environment_waiting_consistent and low_environment_waiting_consistent
    )
    _, _, environment = classify_environment_crossover(
        high_curve_rows=_read_rows(
            data / "renewal_cage_ka_replicates_T058_debye_waller_environment_curve.csv"
        ),
        low_curve_rows=_read_rows(
            data / "renewal_cage_ka_replicates_T045_debye_waller_environment_curve.csv"
        ),
        high_cross_half_rows=_read_rows(
            data
            / "renewal_cage_ka_replicates_T058_debye_waller_environment_cross_half.csv"
        ),
        low_cross_half_rows=_read_rows(
            data
            / "renewal_cage_ka_replicates_T045_debye_waller_environment_cross_half.csv"
        ),
        high_waiting=high_environment_waiting,
        low_waiting=low_environment_waiting,
    )
    stored_environment = _one(
        data / "renewal_cage_ka_debye_waller_environment_crossover_verdict.csv"
    )
    _, waiting = classify_waiting_threshold_sensitivity(
        _read_rows(
            data
            / "renewal_cage_ka_replicates_T045_waiting_threshold_sensitivity_thresholds.csv"
        ),
        finite_exchange_supported=(
            _finite(environment, "finite_exchange_environment_claim_allowed") == 1.0
        ),
    )
    stored_waiting = _one(
        data / "renewal_cage_ka_replicates_T045_waiting_threshold_sensitivity_verdict.csv"
    )
    environment_consistent = _stored_row_matches(stored_environment, environment)
    waiting_consistent = _stored_row_matches(stored_waiting, waiting)
    verdict = classify_memory_hierarchy(
        path=path,
        anchor=anchor,
        waiting=waiting,
        environment=environment,
        spatial_rows=_read_rows(
            data
            / "renewal_cage_ka_replicates_T045_spatial_covariance_ensemble_summary.csv"
        ),
        spatial_fit=_one(
            data
            / "renewal_cage_ka_replicates_T045_spatial_covariance_ensemble_fit_summary.csv"
        ),
    )
    verdict.update(
        {
            "environment_recomputed_from_raw": 1.0,
            "environment_waiting_consensus_recomputed_from_windows": 1.0,
            "waiting_recomputed_from_thresholds": 1.0,
            "stored_high_environment_waiting_consistency_pass": float(
                high_environment_waiting_consistent
            ),
            "stored_low_environment_waiting_consistency_pass": float(
                low_environment_waiting_consistent
            ),
            "stored_environment_waiting_consistency_pass": float(
                environment_waiting_consistent
            ),
            "stored_environment_consistency_pass": float(environment_consistent),
            "stored_waiting_consistency_pass": float(waiting_consistent),
            "stored_subgate_consistency_pass": float(
                environment_waiting_consistent
                and environment_consistent
                and waiting_consistent
            ),
        }
    )
    if not (
        environment_waiting_consistent and environment_consistent and waiting_consistent
    ):
        verdict["mechanism_state"] = "mechanism_hierarchy_unresolved"
        verdict["evidence_completeness_pass"] = 0.0
        verdict["next_minimal_model_candidate"] = "no_model_candidate_selected"
    return verdict


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-evidence-csv", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args(argv)
    verdict = recompute_hierarchy(args.data_dir)
    _write_rows(args.output_csv, [verdict])
    _write_rows(args.output_evidence_csv, build_evidence_rows(verdict))
    write_svg(args.output_svg, verdict)


if __name__ == "__main__":
    main()
