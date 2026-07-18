import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_summary():
    path = ROOT / "scripts" / "summarize_ka_variance_mixture_shape_quotient.py"
    spec = importlib.util.spec_from_file_location(
        "variance_mixture_shape_quotient_summary",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gamma_log_scattering(alpha, x):
    if alpha == 0.0:
        return -x
    return -math.log1p(alpha * x) / alpha


def quotient_rows(*, supported=True):
    temperature = 0.45
    replicate = 1.0
    lag = 20.0 if supported else 40.0
    msd = 0.12
    calibration_alpha = 0.4
    heldout_alpha = 0.6
    common = {
        "temperature": temperature,
        "replicate": replicate,
        "lag": lag,
        "tolerance": 0.03,
        "calibration_msd": msd,
        "heldout_msd": msd,
        "same_time_value": 0.0,
        "same_time_absolute_error": 0.0,
        "msd_matched_absolute_error": 0.0,
        "same_time_normalized_error": 0.0,
        "msd_matched_normalized_error": 0.0,
        "in_calibration_msd_support": float(supported),
        "replicate_support_fraction": 1.0,
        "replicate_anchor_alpha_window_point_count": 1.0,
        "replicate_clock_drift_sign": -1.0,
        "heldout_msd_used_as_diagnostic_input": 1.0,
        "blind_prediction_claim_allowed": 0.0,
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    rows = [
        {
            **common,
            "observable": "ngp",
            "tolerance": 0.3,
            "heldout_value": heldout_alpha,
            "msd_matched_value": calibration_alpha if supported else "",
        }
    ]
    for observable, wave_number, baseline in (
        ("fs_k2", 2.0, 0.91),
        ("fs_k4", 4.0, 0.72),
        ("fs_k7p25", 7.25, 0.43),
    ):
        x = wave_number**2 * msd / 6.0
        correction = math.exp(
            _gamma_log_scattering(heldout_alpha, x)
            - _gamma_log_scattering(calibration_alpha, x)
        )
        rows.append(
            {
                **common,
                "observable": observable,
                "heldout_value": baseline * correction,
                "msd_matched_value": baseline if supported else "",
            }
        )
    return rows


class VarianceMixtureKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_summary()

    def test_analytic_families_share_gaussian_and_fourth_order_limits(self):
        for family in ("gamma", "inverse_gaussian"):
            with self.subTest(family=family):
                self.assertEqual(
                    self.summary.mixture_log_scattering(0.0, 0.7, family),
                    -0.7,
                )
                alpha = 0.4
                x = 1e-5
                correction = (
                    self.summary.mixture_log_scattering(alpha, x, family) + x
                )
                self.assertAlmostEqual(
                    correction / (alpha * x**2 / 2.0),
                    1.0,
                    delta=2e-5,
                )

    def test_analytic_families_reject_nonphysical_domains(self):
        for arguments in (
            (-0.1, 0.2, "gamma"),
            (0.1, -0.2, "gamma"),
            (math.nan, 0.2, "gamma"),
            (0.1, math.inf, "gamma"),
            (0.1, 0.2, "unknown"),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    self.summary.mixture_log_scattering(*arguments)

    def test_csv_serialization_absorbs_platform_transcendental_drift(self):
        self.assertEqual(self.summary.CSV_FLOAT_SIGNIFICANT_DIGITS, 8)
        macos_value = 4.21459089287e-05
        linux_value = 4.21459089148e-05
        self.assertEqual(
            self.summary.canonical_csv_value(macos_value),
            self.summary.canonical_csv_value(linux_value),
        )
        self.assertNotEqual(
            self.summary.canonical_csv_value(1.0),
            self.summary.canonical_csv_value(1.0001),
        )

    def test_scores_only_supported_complete_cells_and_keeps_claims_closed(self):
        rows = self.summary.compute_shape_quotient_rows(
            quotient_rows() + quotient_rows(supported=False)
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual({row["wave_number"] for row in rows}, {2.0, 4.0, 7.25})
        for row in rows:
            self.assertEqual(row["heldout_msd_used_as_diagnostic_input"], 1.0)
            self.assertEqual(row["heldout_ngp_used_as_diagnostic_input"], 1.0)
            self.assertEqual(row["macro_fit_parameter_count"], 0.0)
            self.assertEqual(row["blind_prediction_claim_allowed"], 0.0)
            self.assertEqual(row["static_environment_resolved"], 0.0)
            self.assertEqual(row["finite_exchange_resolved"], 0.0)
            self.assertEqual(row["microdynamic_closure_claim_allowed"], 0.0)
            self.assertEqual(row["spatial_facilitation_claim_allowed"], 0.0)
            self.assertEqual(row["thermodynamic_claim_allowed"], 0.0)
            self.assertAlmostEqual(row["gamma_absolute_error"], 0.0, delta=1e-14)
            self.assertTrue(math.isfinite(row["inverse_gaussian_absolute_error"]))

    def test_rejects_incomplete_duplicate_or_open_claim_cells(self):
        incomplete = quotient_rows()[:-1]
        duplicate = quotient_rows() + [dict(quotient_rows()[0])]
        open_claim = quotient_rows()
        open_claim[0]["thermodynamic_claim_allowed"] = 1.0
        bad_ngp_tolerance = quotient_rows()
        bad_ngp_tolerance[0]["tolerance"] = 0.03

        for rows in (incomplete, duplicate, open_claim, bad_ngp_tolerance):
            with self.subTest():
                with self.assertRaises(ValueError):
                    self.summary.compute_shape_quotient_rows(rows)


class VarianceMixtureGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_summary()
        data = ROOT / "data"
        with (
            data / "renewal_cage_ka_transport_clock_shape_quotient_rows.csv"
        ).open() as handle:
            cls.source_rows = list(csv.DictReader(handle))
        with (
            data / "renewal_cage_ka_transport_clock_shape_quotient_gate.csv"
        ).open() as handle:
            cls.source_gate = list(csv.DictReader(handle))

    def test_real_rows_support_low_shape_closure_and_keep_high_as_canary(self):
        rows = self.summary.compute_shape_quotient_rows(self.source_rows)
        gates = self.summary.classify_shape_quotient_gate(rows, self.source_gate)

        self.assertEqual(len(rows), (18 + 20) * 3)
        self.assertEqual(len(gates), 2)
        low = next(row for row in gates if row["temperature"] == 0.45)
        high = next(row for row in gates if row["temperature"] == 0.58)
        self.assertGreater(low["maximum_fourth_normalized_error"], 1.0)
        self.assertLess(low["maximum_gamma_normalized_error"], 0.47)
        self.assertLess(low["maximum_inverse_gaussian_normalized_error"], 0.43)
        self.assertEqual(low["gamma_all_k_pass"], 1.0)
        self.assertEqual(low["inverse_gaussian_all_k_pass"], 1.0)
        self.assertEqual(low["family_robust_resummation_pass"], 1.0)
        self.assertEqual(
            low["marginal_variance_mixture_shape_closure_supported_exploratory"],
            1.0,
        )
        self.assertEqual(low["variance_mixture_family_selected"], 0.0)
        self.assertEqual(high["high_temperature_canary_only"], 1.0)
        self.assertEqual(high["source_stationarity_pass"], 0.0)
        self.assertEqual(
            high["marginal_variance_mixture_shape_closure_supported_exploratory"],
            0.0,
        )
        for gate in gates:
            self.assertEqual(gate["heldout_msd_used_as_diagnostic_input"], 1.0)
            self.assertEqual(gate["heldout_ngp_used_as_diagnostic_input"], 1.0)
            self.assertEqual(gate["macro_fit_parameter_count"], 0.0)
            for field in self.summary.OUTPUT_CLOSED_CLAIMS:
                self.assertEqual(gate[field], 0.0, (gate["temperature"], field))

    def test_gate_rejects_incomplete_or_open_source_provenance(self):
        rows = self.summary.compute_shape_quotient_rows(self.source_rows)
        incomplete = self.source_gate[:-1]
        open_claim = [dict(row) for row in self.source_gate]
        open_claim[0]["static_environment_resolved"] = 1.0
        wrong_provenance = [dict(row) for row in self.source_gate]
        wrong_provenance[0]["parent_sample_count"] = 2.0

        for source_gate in (incomplete, open_claim, wrong_provenance):
            with self.subTest():
                with self.assertRaises(ValueError):
                    self.summary.classify_shape_quotient_gate(rows, source_gate)

    def test_cli_outputs_are_byte_deterministic_and_claim_limited(self):
        data = ROOT / "data"
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            outputs = [
                temporary / "rows.csv",
                temporary / "gate.csv",
                temporary / "figure.svg",
            ]
            arguments = [
                "--quotient-rows",
                str(data / "renewal_cage_ka_transport_clock_shape_quotient_rows.csv"),
                "--source-gate",
                str(data / "renewal_cage_ka_transport_clock_shape_quotient_gate.csv"),
                "--output-rows",
                str(outputs[0]),
                "--output-gate",
                str(outputs[1]),
                "--output-svg",
                str(outputs[2]),
            ]
            self.summary.main(arguments)
            first = [path.read_bytes() for path in outputs]
            self.summary.main(arguments)
            self.assertEqual(first, [path.read_bytes() for path in outputs])

        rows = list(csv.DictReader(first[0].decode().splitlines()))
        gates = list(csv.DictReader(first[1].decode().splitlines()))
        svg = first[2].decode()
        self.assertEqual(len(rows), (18 + 20) * 3)
        self.assertEqual(len(gates), 2)
        self.assertIn("Variance-mixture shape quotient", svg)
        self.assertIn("held-out MSD and NGP are diagnostic inputs", svg)
        self.assertIn("family unresolved", svg)
        self.assertIn("T=0.58 canary only", svg)
        self.assertIn('width="1280" height="720"', svg)
        self.assertEqual(svg.count('transform="rotate(-90'), 2)
        self.assertIn("stationarity unresolved", svg)
        self.assertIn("Blind and mechanism claims remain 0", svg)
        self.assertNotIn("nan", svg.lower())
        self.assertNotIn("inf", svg.lower())


if __name__ == "__main__":
    unittest.main()
