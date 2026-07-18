import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_summary():
    path = ROOT / "scripts" / "summarize_ka_transport_clock_shape_quotient.py"
    spec = importlib.util.spec_from_file_location("transport_clock_shape_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODELS = (
    "within_particle_segment_shuffle",
    "cross_particle_segment_splice",
)


def source_rows(
    *,
    temperature=0.45,
    full_length=3,
    replicate_count=2,
    clock_shift=0.2,
    ngp_shape_offset=0.02,
    high_k_shape_offset=0.01,
):
    lags = (1, 2, 3)
    rows = []
    for replicate in range(1, replicate_count + 1):
        calibration_msd = (1.0, 2.0, 3.0)
        heldout_msd = (
            tuple(value + clock_shift for value in calibration_msd)
            if replicate == 1
            else tuple(value - clock_shift for value in calibration_msd)
        )
        for model in MODELS:
            for lag, predicted_msd, observed_msd in zip(
                lags, calibration_msd, heldout_msd
            ):
                perturbation = 5e-15 if model == MODELS[1] else 0.0
                rows.append(
                    {
                        "model": model,
                        "temperature": temperature,
                        "segment_length": float(full_length),
                        "replicate": float(replicate),
                        "lag": float(lag),
                        "predicted_msd": predicted_msd + perturbation,
                        "observed_msd": observed_msd,
                        "predicted_ngp": 0.3 * predicted_msd + perturbation,
                        "observed_ngp": 0.3 * observed_msd + ngp_shape_offset,
                        "predicted_fs_k2": 1.0 - 0.1 * predicted_msd + perturbation,
                        "observed_fs_k2": 1.0 - 0.1 * observed_msd,
                        "predicted_fs_k4": 1.0 - 0.2 * predicted_msd + perturbation,
                        "observed_fs_k4": 1.0 - 0.2 * observed_msd,
                        "predicted_fs_k7p25": 1.0 - 0.25 * predicted_msd + perturbation,
                        "observed_fs_k7p25": (
                            1.0 - 0.25 * observed_msd + high_k_shape_offset
                        ),
                        "heldout_path_used_in_prediction": 0.0,
                        "macro_fit_parameter_count": 0.0,
                        "microdynamic_closure_claim_allowed": 0.0,
                        "spatial_facilitation_claim_allowed": 0.0,
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )
    return rows


def manifest(*, temperature=0.45, replicate_count=2, expected_lags=(1, 2, 3)):
    return {
        "temperature": temperature,
        "replicate_count": replicate_count,
        "expected_lags": list(expected_lags),
        "independently_prepared_parent_samples": False,
        "independence_class": "decorrelated_parent_frames_plus_velocity_seeds",
        "replicate_provenance_validation_pass": 1.0,
        "parent_sample_count": 1.0,
        "independent_replicate_count": 0.0,
        "replicates": [
            {"replicate": replicate, "directory": f"replicate_{replicate:02d}"}
            for replicate in range(1, replicate_count + 1)
        ],
        "thermodynamic_claim_allowed": False,
    }


def provenance_rows():
    rows = []
    for temperature, count, source_hash, seed_base in (
        (0.45, 3, "low-parent", 45000),
        (0.58, 5, "high-parent", 58000),
    ):
        for replicate in range(1, count + 1):
            rows.append(
                {
                    "temperature": temperature,
                    "replicate": replicate,
                    "source_sha256": source_hash,
                    "source_frame_index": replicate * 1000,
                    "velocity_seed": seed_base + replicate,
                    "independence_class": (
                        "decorrelated_parent_frames_plus_velocity_seeds"
                    ),
                    "independently_prepared_parent_samples": False,
                    "thermodynamic_claim_allowed": False,
                }
            )
    return rows


def stationarity_rows(*, temperature, all_pass):
    return [
        {
            "temperature": temperature,
            "comparison": comparison,
            "curve_transfer_pass": float(all_pass),
            "microdynamic_closure_claim_allowed": 0.0,
            "spatial_facilitation_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for comparison in ("early_late", "early_heldout", "late_heldout")
    ]


class TransportClockShapeKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_summary()

    def test_piecewise_linear_interpolation_prohibits_extrapolation(self):
        self.assertAlmostEqual(
            self.summary.interpolate_no_extrapolation(
                [1.0, 2.0, 4.0], [10.0, 20.0, 40.0], 3.0
            ),
            30.0,
        )
        self.assertIsNone(
            self.summary.interpolate_no_extrapolation(
                [1.0, 2.0, 4.0], [10.0, 20.0, 40.0], 0.9
            )
        )
        self.assertIsNone(
            self.summary.interpolate_no_extrapolation(
                [1.0, 2.0, 4.0], [10.0, 20.0, 40.0], 4.1
            )
        )
        with self.assertRaises(ValueError):
            self.summary.interpolate_no_extrapolation(
                [1.0, 1.0, 4.0], [10.0, 20.0, 40.0], 2.0
            )

    def test_computes_replicate_rows_and_marks_heldout_msd_as_diagnostic(self):
        rows = self.summary.compute_transport_clock_shape_rows(
            source_rows(),
            manifest(),
            temperature=0.45,
            full_length=3,
            expected_lags=(1, 2, 3),
            minimum_support_fraction=2.0 / 3.0,
        )

        self.assertEqual(len(rows), 2 * 3 * 4)
        self.assertEqual(
            {(int(row["replicate"]), int(row["lag"])) for row in rows},
            {(replicate, lag) for replicate in (1, 2) for lag in (1, 2, 3)},
        )
        supported = [row for row in rows if row["in_calibration_msd_support"] == 1.0]
        self.assertEqual(len(supported), 2 * 2 * 4)
        fs2 = next(
            row
            for row in supported
            if row["replicate"] == 1.0
            and row["lag"] == 1.0
            and row["observable"] == "fs_k2"
        )
        self.assertAlmostEqual(fs2["msd_matched_value"], fs2["heldout_value"])
        self.assertEqual(fs2["heldout_msd_used_as_diagnostic_input"], 1.0)
        self.assertEqual(fs2["blind_prediction_claim_allowed"], 0.0)
        self.assertEqual(fs2["microdynamic_closure_claim_allowed"], 0.0)
        self.assertTrue(math.isfinite(fs2["msd_matched_absolute_error"]))

    def test_rejects_nonfinite_duplicate_leaking_or_open_claim_sources(self):
        mutations = []
        missing = source_rows()
        missing[0].pop("predicted_msd")
        mutations.append(missing)
        duplicate = source_rows()
        duplicate.append(dict(duplicate[0]))
        mutations.append(duplicate)
        nonfinite = source_rows()
        nonfinite[0]["predicted_ngp"] = math.nan
        mutations.append(nonfinite)
        leaking = source_rows()
        leaking[0]["heldout_path_used_in_prediction"] = 1.0
        mutations.append(leaking)
        open_claim = source_rows()
        open_claim[0]["spatial_facilitation_claim_allowed"] = 1.0
        mutations.append(open_claim)
        model_disagreement = source_rows()
        for row in model_disagreement:
            if row["model"] == MODELS[1]:
                row["predicted_fs_k4"] += 1e-4
                break
        mutations.append(model_disagreement)

        for rows in mutations:
            with self.subTest():
                with self.assertRaises(ValueError):
                    self.summary.compute_transport_clock_shape_rows(
                        rows,
                        manifest(),
                        temperature=0.45,
                        full_length=3,
                        expected_lags=(1, 2, 3),
                        minimum_support_fraction=2.0 / 3.0,
                    )

    def test_rejects_fractional_manifest_counts_and_replicates(self):
        fractional_count = manifest()
        fractional_count["replicate_count"] = 2.5
        fractional_replicate = manifest()
        fractional_replicate["replicates"][0]["replicate"] = 1.5

        for invalid_manifest in (fractional_count, fractional_replicate):
            with self.subTest(invalid_manifest=invalid_manifest):
                with self.assertRaises(ValueError):
                    self.summary.compute_transport_clock_shape_rows(
                        source_rows(),
                        invalid_manifest,
                        temperature=0.45,
                        full_length=3,
                        expected_lags=(1, 2, 3),
                        minimum_support_fraction=2.0 / 3.0,
                    )

    def test_csv_float_serialization_uses_cross_platform_precision(self):
        self.assertEqual(self.summary.CSV_FLOAT_SIGNIFICANT_DIGITS, 15)
        self.assertEqual(
            self.summary.canonical_csv_value(2.0953229763876298),
            "2.09532297638763",
        )

    def test_provenance_requires_exact_correlated_parent_restart_contract(self):
        manifests = self.summary.manifests_from_provenance(provenance_rows())
        self.assertEqual(manifests[0.45]["replicate_count"], 3)
        self.assertEqual(manifests[0.58]["replicate_count"], 5)
        for manifest_row in manifests.values():
            self.assertEqual(
                manifest_row["replicate_provenance_validation_pass"], 1.0
            )
            self.assertEqual(manifest_row["parent_sample_count"], 1.0)
            self.assertEqual(manifest_row["independent_replicate_count"], 0.0)

        malformed = []
        malformed.append(provenance_rows()[:-1])
        duplicate_frame = provenance_rows()
        duplicate_frame[1]["source_frame_index"] = duplicate_frame[0][
            "source_frame_index"
        ]
        malformed.append(duplicate_frame)
        duplicate_seed = provenance_rows()
        duplicate_seed[1]["velocity_seed"] = duplicate_seed[0]["velocity_seed"]
        malformed.append(duplicate_seed)
        mixed_parent = provenance_rows()
        mixed_parent[0]["source_sha256"] = "other-parent"
        malformed.append(mixed_parent)

        for rows in malformed:
            with self.subTest():
                with self.assertRaises(ValueError):
                    self.summary.manifests_from_provenance(rows)


class TransportClockShapeGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_summary()

    def quotient_rows(self):
        low = self.summary.compute_transport_clock_shape_rows(
            source_rows(clock_shift=0.8, ngp_shape_offset=0.35),
            manifest(),
            temperature=0.45,
            full_length=3,
            expected_lags=(1, 2, 3),
            minimum_support_fraction=2.0 / 3.0,
        )
        high = self.summary.compute_transport_clock_shape_rows(
            source_rows(temperature=0.58, clock_shift=0.4),
            manifest(temperature=0.58),
            temperature=0.58,
            full_length=3,
            expected_lags=(1, 2, 3),
            minimum_support_fraction=2.0 / 3.0,
        )
        return low + high

    def test_classifies_t045_separation_and_keeps_t058_as_canary(self):
        gates = self.summary.classify_transport_clock_shape_gate(
            self.quotient_rows(),
            {
                0.45: stationarity_rows(temperature=0.45, all_pass=True),
                0.58: stationarity_rows(temperature=0.58, all_pass=False),
            },
            {
                0.45: manifest(),
                0.58: manifest(temperature=0.58),
            },
            minimum_support_fraction=2.0 / 3.0,
        )

        self.assertEqual(len(gates), 2)
        low = next(row for row in gates if row["temperature"] == 0.45)
        high = next(row for row in gates if row["temperature"] == 0.58)
        self.assertGreater(low["fs_k2_max_normalized_support_same_time_error"], 1.0)
        self.assertGreater(low["fs_k4_max_normalized_support_same_time_error"], 1.0)
        self.assertLessEqual(low["fs_k2_max_normalized_msd_matched_error"], 1.0)
        self.assertLessEqual(low["fs_k4_max_normalized_msd_matched_error"], 1.0)
        self.assertGreater(low["ngp_max_normalized_msd_matched_error"], 1.0)
        self.assertEqual(low["replicate_clock_drift_opposite_signs"], 1.0)
        self.assertEqual(low["clock_shape_separation_supported_exploratory"], 1.0)
        self.assertEqual(low["clock_only_closure_allowed"], 0.0)
        self.assertEqual(high["source_ensemble_stationarity_all_comparisons_pass"], 0.0)
        self.assertEqual(high["high_temperature_canary_only"], 1.0)
        self.assertEqual(high["high_temperature_control_resolved"], 0.0)
        self.assertEqual(high["cooling_enhanced_shape_memory_claim_allowed"], 0.0)

        for gate in gates:
            self.assertEqual(gate["replicate_provenance_validation_pass"], 1.0)
            self.assertEqual(gate["parent_sample_count"], 1.0)
            self.assertEqual(gate["independent_replicate_count"], 0.0)
            self.assertEqual(
                gate["independence_class"],
                "decorrelated_parent_frames_plus_velocity_seeds",
            )
            for field in self.summary.CLOSED_GATE_FIELDS:
                self.assertEqual(gate[field], 0.0, (gate["temperature"], field))

    def test_rejects_incomplete_stationarity_or_quotient_support(self):
        bad_stationarity = {
            0.45: stationarity_rows(temperature=0.45, all_pass=True)[:-1],
            0.58: stationarity_rows(temperature=0.58, all_pass=False),
        }
        with self.assertRaises(ValueError):
            self.summary.classify_transport_clock_shape_gate(
                self.quotient_rows(),
                bad_stationarity,
                {0.45: manifest(), 0.58: manifest(temperature=0.58)},
                minimum_support_fraction=2.0 / 3.0,
            )

        missing_complete_lag = [
            row
            for row in self.quotient_rows()
            if not (row["temperature"] == 0.45 and row["lag"] == 3.0)
        ]
        with self.assertRaises(ValueError):
            self.summary.classify_transport_clock_shape_gate(
                missing_complete_lag,
                {
                    0.45: stationarity_rows(temperature=0.45, all_pass=True),
                    0.58: stationarity_rows(temperature=0.58, all_pass=False),
                },
                {0.45: manifest(), 0.58: manifest(temperature=0.58)},
                minimum_support_fraction=2.0 / 3.0,
            )

        incomplete = self.quotient_rows()[:-1]
        with self.assertRaises(ValueError):
            self.summary.classify_transport_clock_shape_gate(
                incomplete,
                {
                    0.45: stationarity_rows(temperature=0.45, all_pass=True),
                    0.58: stationarity_rows(temperature=0.58, all_pass=False),
                },
                {0.45: manifest(), 0.58: manifest(temperature=0.58)},
                minimum_support_fraction=2.0 / 3.0,
            )


class TransportClockShapeRealArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_summary()
        cls.data = ROOT / "data"

    @staticmethod
    def read_rows(path):
        with path.open() as handle:
            return list(csv.DictReader(handle))

    def compute_real(self):
        provenance = self.read_rows(
            self.data / "renewal_cage_ka_replicates_T058_T045_provenance.csv"
        )
        manifests = self.summary.manifests_from_provenance(provenance)
        low = self.summary.compute_transport_clock_shape_rows(
            self.read_rows(
                self.data / "renewal_cage_ka_replicates_T045_segment_splice_rows.csv"
            ),
            manifests[0.45],
            temperature=0.45,
            full_length=250,
            expected_lags=(20, 100, 200, 500, 1000, 2000, 3000),
        )
        high = self.summary.compute_transport_clock_shape_rows(
            self.read_rows(
                self.data / "renewal_cage_ka_replicates_T058_segment_splice_rows.csv"
            ),
            manifests[0.58],
            temperature=0.58,
            full_length=37,
            expected_lags=(20, 100, 200, 400, 600),
        )
        gates = self.summary.classify_transport_clock_shape_gate(
            low + high,
            {
                0.45: self.read_rows(
                    self.data
                    / "renewal_cage_ka_replicates_T045_nonlinear_path_stationarity.csv"
                ),
                0.58: self.read_rows(
                    self.data
                    / "renewal_cage_ka_replicates_T058_block20_nonlinear_path_stationarity.csv"
                ),
            },
            manifests,
        )
        return low + high, gates

    def test_real_quotient_reproduces_clock_and_shape_residuals(self):
        rows, gates = self.compute_real()
        self.assertEqual(len(rows), 184)
        low = next(row for row in gates if row["temperature"] == 0.45)
        high = next(row for row in gates if row["temperature"] == 0.58)
        expected_low = {
            "ngp": 1.0686063508252424,
            "fs_k2": 0.12358652917920931,
            "fs_k4": 0.508749096733759,
            "fs_k7p25": 1.133591809836528,
        }
        expected_high = {
            "ngp": 0.22733683834210927,
            "fs_k2": 0.5664872639006984,
            "fs_k4": 0.824308979649456,
            "fs_k7p25": 0.5981788494163145,
        }
        for observable, expected in expected_low.items():
            self.assertAlmostEqual(
                low[f"{observable}_max_normalized_msd_matched_error"],
                expected,
                delta=5e-12,
            )
        for observable, expected in expected_high.items():
            self.assertAlmostEqual(
                high[f"{observable}_max_normalized_msd_matched_error"],
                expected,
                delta=5e-12,
            )
        self.assertEqual(low["clock_shape_separation_supported_exploratory"], 1.0)
        self.assertEqual(low["replicate_clock_drift_opposite_signs"], 1.0)
        self.assertEqual(high["all_replicates_calibration_slower_than_heldout"], 1.0)
        self.assertEqual(high["source_ensemble_stationarity_all_comparisons_pass"], 0.0)
        self.assertEqual(high["high_temperature_canary_only"], 1.0)

    def test_cli_outputs_are_byte_deterministic_and_claim_limited(self):
        arguments = [
            "--low-rows",
            str(self.data / "renewal_cage_ka_replicates_T045_segment_splice_rows.csv"),
            "--high-rows",
            str(self.data / "renewal_cage_ka_replicates_T058_segment_splice_rows.csv"),
            "--low-stationarity",
            str(
                self.data
                / "renewal_cage_ka_replicates_T045_nonlinear_path_stationarity.csv"
            ),
            "--high-stationarity",
            str(
                self.data
                / "renewal_cage_ka_replicates_T058_block20_nonlinear_path_stationarity.csv"
            ),
            "--provenance",
            str(self.data / "renewal_cage_ka_replicates_T058_T045_provenance.csv"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            outputs = [
                temporary / "rows.csv",
                temporary / "gate.csv",
                temporary / "figure.svg",
            ]
            output_arguments = [
                "--output-rows",
                str(outputs[0]),
                "--output-gate",
                str(outputs[1]),
                "--output-svg",
                str(outputs[2]),
            ]
            self.summary.main(arguments + output_arguments)
            first = [path.read_bytes() for path in outputs]
            self.summary.main(arguments + output_arguments)
            self.assertEqual(first, [path.read_bytes() for path in outputs])

        self.assertEqual(len(list(csv.DictReader(first[0].decode().splitlines()))), 184)
        self.assertEqual(len(list(csv.DictReader(first[1].decode().splitlines()))), 2)
        svg = first[2].decode()
        self.assertIn("Transport clock / shape quotient", svg)
        self.assertIn("heldout MSD is a diagnostic input", svg)
        self.assertIn("T=0.58 canary only", svg)
        self.assertIn("clock-only closure rejected", svg)
        self.assertNotIn("nan", svg.lower())
        self.assertNotIn("inf", svg.lower())

if __name__ == "__main__":
    unittest.main()
