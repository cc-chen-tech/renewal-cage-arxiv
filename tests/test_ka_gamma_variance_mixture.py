import csv
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_summary():
    path = SCRIPTS / "summarize_ka_gamma_variance_mixture.py"
    spec = importlib.util.spec_from_file_location("gamma_variance_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GammaVarianceFormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_summary()

    def test_gamma_formula_and_gaussian_limit(self):
        self.assertAlmostEqual(
            self.summary.gamma_variance_fs(msd=6.0, alpha2=0.5, wave_number=1.0),
            4.0 / 9.0,
        )
        self.assertAlmostEqual(
            self.summary.gamma_variance_fs(msd=6.0, alpha2=0.0, wave_number=1.0),
            math.exp(-1.0),
        )
        self.assertAlmostEqual(
            self.summary.shifted_gamma_variance_fs(
                msd=6.0,
                alpha2=0.5,
                cage_variance=0.0,
                wave_number=1.0,
            ),
            4.0 / 9.0,
        )

    def test_cage_variance_inversion_is_bracketed_and_has_no_extrapolation(self):
        expected = 0.2
        target = self.summary.shifted_gamma_variance_fs(
            msd=3.0,
            alpha2=0.4,
            cage_variance=expected,
            wave_number=2.0,
        )
        inferred = self.summary.infer_cage_variance(
            msd=3.0,
            alpha2=0.4,
            fs_k2=target,
        )
        self.assertIsNotNone(inferred)
        self.assertAlmostEqual(inferred, expected, places=11)

        gamma_endpoint = self.summary.shifted_gamma_variance_fs(
            msd=3.0,
            alpha2=0.4,
            cage_variance=0.0,
            wave_number=2.0,
        )
        self.assertIsNone(
            self.summary.infer_cage_variance(
                msd=3.0,
                alpha2=0.4,
                fs_k2=gamma_endpoint + 1e-3,
            )
        )

    def test_formula_rejects_nonphysical_inputs(self):
        invalid = (
            {"msd": 0.0, "alpha2": 0.2, "wave_number": 1.0},
            {"msd": 1.0, "alpha2": -0.1, "wave_number": 1.0},
            {"msd": 1.0, "alpha2": 0.1, "wave_number": 0.0},
            {"msd": math.nan, "alpha2": 0.1, "wave_number": 1.0},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    self.summary.gamma_variance_fs(**arguments)

    def test_csv_serialization_discards_platform_last_bit_drift(self):
        self.assertEqual(self.summary.CSV_FLOAT_SIGNIFICANT_DIGITS, 12)
        self.assertEqual(
            self.summary.canonical_csv_value(0.314802545826245),
            self.summary.canonical_csv_value(0.314802545826246),
        )
        self.assertNotEqual(
            self.summary.canonical_csv_value(1.0),
            self.summary.canonical_csv_value(1.0001),
        )


class GammaVarianceRealDataTests(unittest.TestCase):
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
        rows = []
        for temperature, prefix, full_length, lags in (
            (0.45, "T045", 250, (20, 100, 200, 500, 1000, 2000, 3000)),
            (0.58, "T058", 37, (20, 100, 200, 400, 600)),
        ):
            rows.extend(
                self.summary.compute_gamma_variance_mixture_rows(
                    self.read_rows(
                        self.data
                        / f"renewal_cage_ka_replicates_{prefix}_segment_splice_rows.csv"
                    ),
                    manifests[temperature],
                    temperature=temperature,
                    full_length=full_length,
                    expected_lags=lags,
                )
            )
        gates = self.summary.classify_gamma_variance_mixture_gate(
            rows,
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
        return rows, gates

    def test_real_scalar_mobility_passes_low_k_but_fails_t045_cage_scale(self):
        rows, gates = self.compute_real()
        self.assertEqual(len(rows), 46)
        low = next(row for row in gates if row["temperature"] == 0.45)
        high = next(row for row in gates if row["temperature"] == 0.58)
        expected_low = {
            "fs_k2": 0.07739448487366445,
            "fs_k4": 0.6167447851817813,
            "fs_k7p25": 1.3854210524976782,
        }
        expected_high = {
            "fs_k2": 0.11304979791994056,
            "fs_k4": 0.18960310152781404,
            "fs_k7p25": 0.36735177768728405,
        }
        for observable, expected in expected_low.items():
            self.assertAlmostEqual(
                low[f"{observable}_gamma_max_normalized_error"],
                expected,
                delta=5e-12,
            )
        for observable, expected in expected_high.items():
            self.assertAlmostEqual(
                high[f"{observable}_gamma_max_normalized_error"],
                expected,
                delta=5e-12,
            )
        self.assertEqual(low["scalar_mobility_shape_closure_supported_exploratory"], 0.0)
        self.assertEqual(low["cage_plus_mobility_support_coverage_pass"], 0.0)
        self.assertEqual(high["high_temperature_canary_only"], 1.0)
        self.assertEqual(high["high_temperature_control_resolved"], 0.0)

    def test_real_gate_keeps_provenance_and_claim_boundaries_closed(self):
        _, gates = self.compute_real()
        for gate in gates:
            self.assertEqual(gate["replicate_provenance_validation_pass"], 1.0)
            self.assertEqual(gate["parent_sample_count"], 1.0)
            self.assertEqual(gate["independent_replicate_count"], 0.0)
            for field in self.summary.CLOSED_GATE_FIELDS:
                self.assertEqual(gate[field], 0.0, (gate["temperature"], field))

    def test_cli_outputs_are_deterministic_and_claim_limited(self):
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
            "--simulation-particles",
            "2000",
            "--simulation-steps",
            "20",
        ]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            outputs = [
                temporary / "rows.csv",
                temporary / "gate.csv",
                temporary / "simulation.csv",
                temporary / "figure.svg",
            ]
            output_arguments = [
                "--output-rows",
                str(outputs[0]),
                "--output-gate",
                str(outputs[1]),
                "--output-simulation",
                str(outputs[2]),
                "--output-svg",
                str(outputs[3]),
            ]
            self.summary.main(arguments + output_arguments)
            first = [path.read_bytes() for path in outputs]
            self.summary.main(arguments + output_arguments)
            self.assertEqual(first, [path.read_bytes() for path in outputs])

        self.assertEqual(len(list(csv.DictReader(first[0].decode().splitlines()))), 46)
        self.assertEqual(len(list(csv.DictReader(first[1].decode().splitlines()))), 2)
        self.assertEqual(len(list(csv.DictReader(first[2].decode().splitlines()))), 3)
        svg = first[3].decode()
        self.assertIn("Gamma variance-mixture Langevin diagnostic", svg)
        self.assertIn("maximum normalized Fs error (tolerance units)", svg)
        self.assertIn("heldout MSD and NGP are diagnostic inputs", svg)
        self.assertIn("scalar mobility fails at cage scale", svg)
        self.assertIn("T=0.58 canary only", svg)
        self.assertNotIn("nan", svg.lower())
        self.assertNotIn("inf", svg.lower())


class SquaredOuLangevinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_summary()

    def test_slow_squared_ou_mobility_converges_to_gamma_closure(self):
        rows = self.summary.simulate_squared_ou_mobility(
            tau_ratios=(1.0, 10.0, 100.0),
            particle_count=60_000,
            step_count=100,
            seed=20260718,
        )
        self.assertEqual([row["tau_D_over_t"] for row in rows], [1.0, 10.0, 100.0])
        slow = rows[-1]
        self.assertEqual(slow["slow_environment_limit_validation_pass"], 1.0)
        self.assertEqual(rows[0]["slow_environment_limit_validation_pass"], 0.0)
        self.assertLess(abs(slow["empirical_msd"] / slow["analytic_msd"] - 1.0), 0.015)
        self.assertLess(abs(slow["empirical_ngp"] - slow["analytic_ngp"]), 0.035)
        for wave_number in (0.5, 1.0, 2.0):
            self.assertLess(
                abs(
                    slow[f"empirical_fs_k{str(wave_number).replace('.', 'p')}"]
                    - slow[f"analytic_fs_k{str(wave_number).replace('.', 'p')}"]
                ),
                0.012,
            )
        self.assertGreater(rows[0]["empirical_ngp"], 0.0)
        self.assertLess(rows[0]["empirical_ngp"], rows[-1]["empirical_ngp"])

    def test_simulation_is_seed_deterministic(self):
        arguments = {
            "tau_ratios": (100.0,),
            "particle_count": 2_000,
            "step_count": 20,
            "seed": 17,
        }
        self.assertEqual(
            self.summary.simulate_squared_ou_mobility(**arguments),
            self.summary.simulate_squared_ou_mobility(**arguments),
        )


if __name__ == "__main__":
    unittest.main()
