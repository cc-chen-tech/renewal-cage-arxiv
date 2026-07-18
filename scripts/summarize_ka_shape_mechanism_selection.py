#!/usr/bin/env python3
"""Compare marginal-shape and factorized event-path closures on one KA grid."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


WAVE_NUMBERS = (2.0, 4.0, 7.25)
FS_TOLERANCE = 0.03
TAIL_TOLERANCE = 0.01
EVENT_MODELS = (
    "independent_jump",
    "disjoint_pair",
    "pair_eigenmode",
    "empirical_path",
)
VARIANCE_MODELS = (
    "gamma_variance_mixture",
    "inverse_gaussian_variance_mixture",
)
MODELS = EVENT_MODELS + VARIANCE_MODELS
CLAIM_FLAGS = (
    "blind_prediction_claim_allowed",
    "unique_variance_mixture_family_selected",
    "static_environment_resolved",
    "finite_exchange_resolved",
    "cage_jump_coupling_identified",
    "factorized_event_path_family_excluded_beyond_tested_closures",
    "microdynamic_closure_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)
EXPECTED_GRIDS = {
    0.45: frozenset(
        {
            (1, 20),
            (1, 100),
            (1, 200),
            (1, 500),
            (1, 1000),
            (1, 2000),
            (2, 100),
            (2, 200),
            (2, 500),
            (2, 1000),
            (2, 2000),
            (2, 3000),
            (3, 20),
            (3, 100),
            (3, 200),
            (3, 500),
            (3, 1000),
            (3, 2000),
        }
    ),
    0.58: frozenset(
        (replicate, lag)
        for replicate in range(1, 6)
        for lag in (20, 100, 200, 400)
    ),
}


def wave_key(wave_number: float) -> str:
    return f"k{wave_number:g}".replace(".", "p")


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


def count_pgf(*, mean_count: float, fano_factor: float, argument: float) -> float:
    values = (mean_count, fano_factor, argument)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("count PGF inputs must be finite")
    if mean_count < 0.0 or fano_factor < 1.0 or not -1.0 <= argument <= 1.0:
        raise ValueError("count PGF inputs are outside their physical domain")
    if math.isclose(fano_factor, 1.0, abs_tol=1e-12):
        return math.exp(mean_count * (argument - 1.0))
    excess = fano_factor - 1.0
    return (1.0 + excess * (1.0 - argument)) ** (-mean_count / excess)


def count_pmf(
    *,
    mean_count: float,
    fano_factor: float,
    maximum_count: int,
) -> tuple[list[float], float]:
    if (
        not math.isfinite(mean_count)
        or not math.isfinite(fano_factor)
        or mean_count < 0.0
        or fano_factor < 1.0
        or isinstance(maximum_count, bool)
        or not isinstance(maximum_count, int)
        or maximum_count < 0
    ):
        raise ValueError("count PMF inputs are outside their physical domain")
    probabilities = [count_pgf(mean_count=mean_count, fano_factor=fano_factor, argument=0.0)]
    if math.isclose(fano_factor, 1.0, abs_tol=1e-12):
        for count in range(1, maximum_count + 1):
            probabilities.append(probabilities[-1] * mean_count / count)
    else:
        excess = fano_factor - 1.0
        shape = mean_count / excess
        continuation = excess / fano_factor
        for count in range(1, maximum_count + 1):
            probabilities.append(
                probabilities[-1]
                * (count - 1.0 + shape)
                / count
                * continuation
            )
    tail = 1.0 - math.fsum(probabilities)
    if tail < -1e-12 or tail > 1.0 + 1e-12:
        raise ValueError("count PMF recurrence lost normalization")
    return probabilities, min(1.0, max(0.0, tail))


def event_characteristic(
    *,
    model: str,
    mean_count: float,
    fano_factor: float,
    single_characteristic: float,
    pair_characteristic: float,
    empirical_characteristics: list[float] | None,
) -> tuple[float, float]:
    values = (single_characteristic, pair_characteristic)
    if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values):
        raise ValueError("jump characteristics must lie in [-1, 1]")
    if model == "independent_jump":
        return (
            count_pgf(
                mean_count=mean_count,
                fano_factor=fano_factor,
                argument=single_characteristic,
            ),
            0.0,
        )
    if model == "disjoint_pair":
        if pair_characteristic < 0.0:
            raise ValueError("disjoint-pair closure requires a nonnegative pair characteristic")
        root = math.sqrt(pair_characteristic)
        if root <= 1e-15:
            probabilities, _ = count_pmf(
                mean_count=mean_count,
                fano_factor=fano_factor,
                maximum_count=1,
            )
            return probabilities[0] + probabilities[1] * single_characteristic, 0.0
        positive = count_pgf(
            mean_count=mean_count,
            fano_factor=fano_factor,
            argument=root,
        )
        negative = count_pgf(
            mean_count=mean_count,
            fano_factor=fano_factor,
            argument=-root,
        )
        ratio = single_characteristic / root
        return 0.5 * ((1.0 + ratio) * positive + (1.0 - ratio) * negative), 0.0
    if model == "pair_eigenmode":
        if abs(single_characteristic) <= 1e-15:
            raise ValueError("pair-eigenmode closure requires a nonzero single characteristic")
        eigenvalue = pair_characteristic / single_characteristic
        if not -1.0 <= eigenvalue <= 1.0 or abs(eigenvalue) <= 1e-15:
            raise ValueError("pair-eigenmode eigenvalue lies outside its physical domain")
        zero = count_pgf(
            mean_count=mean_count,
            fano_factor=fano_factor,
            argument=0.0,
        )
        transformed = count_pgf(
            mean_count=mean_count,
            fano_factor=fano_factor,
            argument=eigenvalue,
        )
        return zero + single_characteristic / eigenvalue * (transformed - zero), 0.0
    if model == "empirical_path":
        if empirical_characteristics is None or len(empirical_characteristics) < 3:
            raise ValueError("empirical path closure requires count-zero through count-two kernels")
        if any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0
            for value in empirical_characteristics
        ):
            raise ValueError("empirical path characteristics must lie in [-1, 1]")
        if not math.isclose(empirical_characteristics[0], 1.0, abs_tol=1e-12):
            raise ValueError("zero-jump empirical characteristic must equal one")
        if not math.isclose(
            empirical_characteristics[1], single_characteristic, abs_tol=1e-12
        ) or not math.isclose(
            empirical_characteristics[2], pair_characteristic, abs_tol=1e-12
        ):
            raise ValueError("empirical path does not share the supplied one- and two-jump kernels")
        probabilities, tail = count_pmf(
            mean_count=mean_count,
            fano_factor=fano_factor,
            maximum_count=len(empirical_characteristics) - 1,
        )
        return math.fsum(
            probability * characteristic
            for probability, characteristic in zip(
                probabilities, empirical_characteristics, strict=True
            )
        ), tail
    raise ValueError(f"unknown event model: {model}")


def _variance_cells(root: Path) -> tuple[
    dict[tuple[float, int, int], dict[float, dict[str, str]]],
    dict[float, dict[str, str]],
]:
    data = root / "data"
    rows = read_rows(data / "renewal_cage_ka_variance_mixture_shape_quotient_rows.csv")
    cells: dict[tuple[float, int, int], dict[float, dict[str, str]]] = {}
    for row in rows:
        key = (
            float(row["temperature"]),
            int(float(row["replicate"])),
            int(float(row["lag"])),
        )
        wave_number = float(row["wave_number"])
        if wave_number in cells.setdefault(key, {}):
            raise ValueError(f"duplicate variance-mixture cell {key + (wave_number,)}")
        cells[key][wave_number] = row
    for temperature, expected in EXPECTED_GRIDS.items():
        actual = {(replicate, lag) for local_temperature, replicate, lag in cells if local_temperature == temperature}
        if actual != expected:
            raise ValueError(f"variance-mixture common grid changed at T={temperature}")
    if set(cells) != {
        (temperature, replicate, lag)
        for temperature, grid in EXPECTED_GRIDS.items()
        for replicate, lag in grid
    }:
        raise ValueError("variance-mixture table contains an unexpected temperature grid")
    for key, wave_rows in cells.items():
        if set(wave_rows) != set(WAVE_NUMBERS):
            raise ValueError(f"variance-mixture cell {key} lacks the frozen wave-number grid")
    gates = {
        float(row["temperature"]): row
        for row in read_rows(
            data / "renewal_cage_ka_variance_mixture_shape_quotient_gate.csv"
        )
    }
    if set(gates) != set(EXPECTED_GRIDS):
        raise ValueError("variance-mixture source gate temperature grid changed")
    return cells, gates


def analyze_committed_tables(
    root: Path,
) -> tuple[
    list[dict[str, float | str]],
    list[dict[str, float | str]],
    list[dict[str, float | str]],
]:
    data = root / "data"
    variance_cells, source_gates = _variance_cells(root)
    count_rows = {
        (
            float(row["temperature"]),
            int(float(row["replicate"])),
            int(float(row["lag"])),
        ): row
        for row in read_rows(data / "renewal_cage_ka_count_overdispersed_geometry_rows.csv")
    }
    propagators: dict[tuple[float, int], dict[int, dict[str, str]]] = {}
    for temperature, label in ((0.45, "045"), (0.58, "058")):
        rows = read_rows(
            data / f"renewal_cage_ka_replicates_T{label}_hybrid_macro_propagator.csv"
        )
        for row in rows:
            key = (temperature, int(float(row["replicate"])))
            count = int(float(row["jump_count"]))
            if count in propagators.setdefault(key, {}):
                raise ValueError(f"duplicate propagator count {key + (count,)}")
            propagators[key][count] = row
    output: list[dict[str, float | str]] = []
    for key in sorted(variance_cells):
        temperature, replicate, lag = key
        if key not in count_rows or float(count_rows[key]["supported"]) != 1.0:
            raise ValueError(f"common shape cell lacks physical count support: {key}")
        count_row = count_rows[key]
        path = propagators.get((temperature, replicate))
        if path is None or set(path) != set(range(max(path) + 1)) or max(path) < 2:
            raise ValueError(f"propagator count grid is not contiguous for {(temperature, replicate)}")
        base: dict[str, float | str] = {
            "temperature": temperature,
            "replicate": float(replicate),
            "lag": float(lag),
            "heldout_msd": float(count_row["heldout_msd"]),
            "heldout_ngp": float(count_row["heldout_ngp"]),
            "inferred_mean_event_count": float(count_row["inferred_mean_event_count"]),
            "count_fano_factor": float(count_row["count_fano_factor"]),
            "inferred_cage_variance": float(count_row["inferred_cage_variance"]),
        }
        for model in MODELS:
            local: dict[str, float | str] = {
                **base,
                "model": model,
                "model_class": (
                    "factorized_event_path" if model in EVENT_MODELS else "marginal_variance_mixture"
                ),
                "maximum_omitted_count_probability": 0.0,
                "calibration_event_path_statistics_used": float(model in EVENT_MODELS),
                "heldout_msd_used_as_diagnostic_input": 1.0,
                "heldout_ngp_used_as_diagnostic_input": 1.0,
                "macro_fit_parameter_count": 0.0,
            }
            errors: list[float] = []
            for wave_number in WAVE_NUMBERS:
                wave_row = variance_cells[key][wave_number]
                wave = wave_key(wave_number)
                observed = float(wave_row["heldout_fs"])
                if model in VARIANCE_MODELS:
                    field = (
                        "gamma_fs"
                        if model == "gamma_variance_mixture"
                        else "inverse_gaussian_fs"
                    )
                    predicted = float(wave_row[field])
                    tail = 0.0
                else:
                    characteristic_field = f"conditional_characteristic_{wave}"
                    empirical = [
                        float(path[count][characteristic_field])
                        for count in range(max(path) + 1)
                    ]
                    event_value, tail = event_characteristic(
                        model=model,
                        mean_count=float(base["inferred_mean_event_count"]),
                        fano_factor=float(base["count_fano_factor"]),
                        single_characteristic=empirical[1],
                        pair_characteristic=empirical[2],
                        empirical_characteristics=empirical,
                    )
                    predicted = math.exp(
                        -wave_number**2 * float(base["inferred_cage_variance"])
                    ) * event_value
                error = abs(predicted - observed)
                errors.append(error)
                local[f"observed_fs_{wave}"] = observed
                local[f"predicted_fs_{wave}"] = predicted
                local[f"fs_{wave}_absolute_error"] = error
                local["maximum_omitted_count_probability"] = max(
                    float(local["maximum_omitted_count_probability"]), tail
                )
            local["maximum_absolute_error"] = max(errors)
            output.append(local)
    models = summarize_models(output, source_gates=source_gates)
    verdicts = summarize_verdicts(models, source_gates=source_gates)
    return output, models, verdicts


def summarize_models(
    rows: list[dict[str, float | str]],
    *,
    source_gates: dict[float, dict[str, str]] | None = None,
) -> list[dict[str, float | str]]:
    if not rows:
        raise ValueError("model summary requires rows")
    source_gates = source_gates or {
        temperature: {"source_stationarity_pass": "1", "parent_sample_count": "1"}
        for temperature in EXPECTED_GRIDS
    }
    summaries: list[dict[str, float | str]] = []
    for temperature, expected_grid in EXPECTED_GRIDS.items():
        temperature_rows = [
            row for row in rows if math.isclose(float(row["temperature"]), temperature)
        ]
        actual = {
            (int(float(row["replicate"])), int(float(row["lag"])), str(row["model"]))
            for row in temperature_rows
        }
        expected = {
            (replicate, lag, model)
            for replicate, lag in expected_grid
            for model in MODELS
        }
        if actual != expected or len(temperature_rows) != len(expected):
            raise ValueError(f"common model grid is incomplete at T={temperature}")
        stationarity = bool(float(source_gates[temperature]["source_stationarity_pass"]))
        for model in MODELS:
            local = [row for row in temperature_rows if row["model"] == model]
            wave_errors = {
                wave_key(wave_number): max(
                    float(row[f"fs_{wave_key(wave_number)}_absolute_error"])
                    for row in local
                )
                for wave_number in WAVE_NUMBERS
            }
            maximum_error = max(wave_errors.values())
            maximum_tail = max(
                float(row["maximum_omitted_count_probability"]) for row in local
            )
            passed = (
                stationarity
                and maximum_error <= FS_TOLERANCE
                and maximum_tail <= TAIL_TOLERANCE
            )
            summaries.append(
                {
                    "temperature": temperature,
                    "model": model,
                    "model_class": local[0]["model_class"],
                    "replicate_lag_cell_count": float(len(local)),
                    "replicate_count": float(len({int(float(row["replicate"])) for row in local})),
                    "wave_number_count": float(len(WAVE_NUMBERS)),
                    "source_stationarity_pass": float(stationarity),
                    "fs_k2_max_absolute_error": wave_errors["k2"],
                    "fs_k4_max_absolute_error": wave_errors["k4"],
                    "fs_k7p25_max_absolute_error": wave_errors["k7p25"],
                    "maximum_absolute_error": maximum_error,
                    "fs_absolute_error_tolerance": FS_TOLERANCE,
                    "maximum_omitted_count_probability": maximum_tail,
                    "count_tail_probability_tolerance": TAIL_TOLERANCE,
                    "all_k_pass": float(passed),
                    "calibration_event_path_statistics_used": float(model in EVENT_MODELS),
                    "heldout_msd_used_as_diagnostic_input": 1.0,
                    "heldout_ngp_used_as_diagnostic_input": 1.0,
                    "macro_fit_parameter_count": 0.0,
                }
            )
    return summaries


def summarize_verdicts(
    models: list[dict[str, float | str]],
    *,
    source_gates: dict[float, dict[str, str]],
) -> list[dict[str, float | str]]:
    verdicts: list[dict[str, float | str]] = []
    for temperature in sorted(EXPECTED_GRIDS):
        local = {
            str(row["model"]): row
            for row in models
            if math.isclose(float(row["temperature"]), temperature)
        }
        if set(local) != set(MODELS):
            raise ValueError(f"model verdict grid is incomplete at T={temperature}")
        stationarity = bool(float(source_gates[temperature]["source_stationarity_pass"]))
        event_fail = all(float(local[model]["all_k_pass"]) == 0.0 for model in EVENT_MODELS)
        variance_pass = all(
            float(local[model]["all_k_pass"]) == 1.0 for model in VARIANCE_MODELS
        )
        selected = stationarity and event_fail and variance_pass
        result: dict[str, float | str] = {
            "temperature": temperature,
            "analysis_status": (
                "variance_mixture_shape_survives_factorized_event_path_closures_fail"
                if selected
                else (
                    "high_temperature_canary_only"
                    if not stationarity
                    else "shape_mechanism_unresolved"
                )
            ),
            "common_replicate_lag_cell_count": float(len(EXPECTED_GRIDS[temperature])),
            "wave_number_count": float(len(WAVE_NUMBERS)),
            "tested_factorized_event_path_model_count": float(len(EVENT_MODELS)),
            "tested_variance_mixture_model_count": float(len(VARIANCE_MODELS)),
            "source_stationarity_pass": float(stationarity),
            "parent_sample_count": float(source_gates[temperature]["parent_sample_count"]),
            "tested_factorized_event_path_closures_excluded_on_common_grid": float(
                selected
            ),
            "positive_variance_mixture_shape_class_survives": float(selected),
            "diagnostic_shape_selection_only": 1.0,
            "heldout_msd_used_as_diagnostic_input": 1.0,
            "heldout_ngp_used_as_diagnostic_input": 1.0,
            "macro_fit_parameter_count": 0.0,
            "confirmatory_independent_parent_replication_required": 1.0,
            "next_required_action": "test_nonfactorized_cage_jump_or_environment_coupling",
        }
        result.update({flag: 0.0 for flag in CLAIM_FLAGS})
        verdicts.append(result)
    return verdicts


def write_svg(path: Path, models: list[dict[str, float | str]]) -> None:
    low = {
        str(row["model"]): row
        for row in models
        if math.isclose(float(row["temperature"]), 0.45)
    }
    if set(low) != set(MODELS):
        raise ValueError("SVG requires the complete T=0.45 model grid")
    labels = {
        "independent_jump": ("Independent", "jump"),
        "disjoint_pair": ("Disjoint", "pairs"),
        "pair_eigenmode": ("Pair", "eigenmode"),
        "empirical_path": ("Empirical", "path"),
        "gamma_variance_mixture": ("Gamma", "mixture"),
        "inverse_gaussian_variance_mixture": ("Inverse-Gaussian", "mixture"),
    }
    colors = {2.0: "#147d92", 4.0: "#d97706", 7.25: "#b83280"}
    width = 960
    height = 560
    left = 88.0
    right = 36.0
    bottom = 150.0
    top = 102.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum_normalized_error = 4.25

    def y_coordinate(value: float) -> float:
        clipped = min(maximum_normalized_error, max(0.0, value))
        return top + plot_height * (1.0 - clipped / maximum_normalized_error)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="560" viewBox="0 0 960 560">',
        "<title>Shape-class mechanism selection</title>",
        '<rect width="960" height="560" fill="#ffffff"/>',
        '<g font-family="Arial,Helvetica,sans-serif" letter-spacing="0">',
        '<text x="48" y="42" font-size="22" font-weight="700" fill="#17202a">Shape-class mechanism selection</text>',
        '<text x="48" y="70" font-size="13" fill="#44515c">T=0.45, identical 18 replicate-lag cells; held-out MSD and NGP are diagnostic inputs</text>',
    ]
    spacing = plot_width / len(MODELS)
    for index, model in enumerate(MODELS):
        if model not in VARIANCE_MODELS:
            continue
        x = left + spacing * (index + 0.5)
        lines.append(
            f'<rect x="{x-spacing*0.44:.1f}" y="{top:.1f}" width="{spacing*0.88:.1f}" height="{plot_height:.1f}" fill="#edf7f0"/>'
        )
    for tick in range(5):
        y = y_coordinate(float(tick))
        lines.extend(
            [
                f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{width-right:.1f}" y2="{y:.1f}" stroke="#d9dee3" stroke-width="1"/>',
                f'<text x="{left-12:.1f}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#56616b">{tick}</text>',
            ]
        )
    tolerance_y = y_coordinate(1.0)
    lines.extend(
        [
            f'<line x1="{left:.1f}" y1="{tolerance_y:.1f}" x2="{width-right:.1f}" y2="{tolerance_y:.1f}" stroke="#20262c" stroke-width="2" stroke-dasharray="7 5"/>',
            f'<text x="{width-right-4:.1f}" y="{tolerance_y-8:.1f}" text-anchor="end" font-size="11" font-weight="700" fill="#20262c">frozen tolerance</text>',
            f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{top+plot_height:.1f}" stroke="#20262c" stroke-width="1.5"/>',
            f'<line x1="{left:.1f}" y1="{top+plot_height:.1f}" x2="{width-right:.1f}" y2="{top+plot_height:.1f}" stroke="#20262c" stroke-width="1.5"/>',
            '<text x="20" y="300" transform="rotate(-90 20 300)" text-anchor="middle" font-size="12" fill="#26323b">maximum absolute Fs error / 0.03</text>',
        ]
    )
    for index, model in enumerate(MODELS):
        x = left + spacing * (index + 0.5)
        row = low[model]
        for offset, wave_number in zip((-14.0, 0.0, 14.0), WAVE_NUMBERS, strict=True):
            key = wave_key(wave_number)
            normalized = float(row[f"fs_{key}_max_absolute_error"]) / FS_TOLERANCE
            y = y_coordinate(normalized)
            lines.append(
                f'<circle cx="{x+offset:.1f}" cy="{y:.1f}" r="6.5" fill="{colors[wave_number]}" stroke="#ffffff" stroke-width="1.5"><title>{model}, k={wave_number:g}: {normalized:.3f}</title></circle>'
            )
        first, second = labels[model]
        lines.extend(
            [
                f'<text x="{x:.1f}" y="{top+plot_height+28:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="#26323b">{first}</text>',
                f'<text x="{x:.1f}" y="{top+plot_height+44:.1f}" text-anchor="middle" font-size="11" fill="#26323b">{second}</text>',
            ]
        )
    legend_x = 610.0
    for index, wave_number in enumerate(WAVE_NUMBERS):
        x = legend_x + index * 92.0
        lines.extend(
            [
                f'<circle cx="{x:.1f}" cy="42" r="5.5" fill="{colors[wave_number]}"/>',
                f'<text x="{x+10:.1f}" y="46" font-size="11" fill="#34414b">k={wave_number:g}</text>',
            ]
        )
    lines.extend(
        [
            '<text x="48" y="520" font-size="12" font-weight="700" fill="#26323b">Result:</text>',
            '<text x="98" y="520" font-size="12" fill="#26323b">all tested factorized event-path closures fail; both positive variance-mixture resummations pass.</text>',
            '<text x="48" y="542" font-size="11" fill="#59656f">Diagnostic shape-class selection only; no unique family, static/finite exchange, cage-jump coupling, or microscopic mechanism is identified.</text>',
            "</g>",
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--output-models", type=Path, required=True)
    parser.add_argument("--output-verdict", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    rows, models, verdicts = analyze_committed_tables(args.root)
    write_rows(args.output_rows, rows)
    write_rows(args.output_models, models)
    write_rows(args.output_verdict, verdicts)
    write_svg(args.output_svg, models)


if __name__ == "__main__":
    main()
