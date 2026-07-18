import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_script(filename: str, module_name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CountOverdispersedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_script(
            "summarize_ka_count_overdispersed_geometry.py",
            "summarize_ka_count_overdispersed_geometry",
        )

    def test_row_uses_gamma_poisson_count_cumulant_and_pgf(self):
        geometry = {
            "temperature": "0.45",
            "replicate": "1",
            "lag": "20",
            "heldout_msd": "0.6",
            "heldout_ngp": "0.5",
            "jump_msd": "0.2",
            "jump_component_fourth_moment": "0.04",
            "jump_characteristic_k2": "0.8",
            "jump_characteristic_k4": "0.6",
            "jump_characteristic_k7p25": "0.4",
            "observed_fs_k2": "0.7",
            "observed_fs_k4": "0.5",
            "observed_fs_k7p25": "0.3",
        }
        macro = {
            "replicate": "1",
            "lag": "20",
            "predicted_mean_count": "1.0",
            "predicted_count_variance": "2.0",
        }

        row = self.analysis.count_overdispersed_geometry_row(geometry, macro)

        denominator = 0.04 + 3.0 * (0.2 / 3.0) ** 2
        mean_count = 0.5 * 0.6**2 / (3.0 * denominator)
        cage_variance = (0.6 - mean_count * 0.2) / 6.0
        expected_fs = math.exp(-4.0 * cage_variance) * 1.2 ** (-mean_count)
        self.assertEqual(row["supported"], 1.0)
        self.assertAlmostEqual(row["count_fano_factor"], 2.0)
        self.assertAlmostEqual(row["inferred_mean_event_count"], mean_count)
        self.assertAlmostEqual(row["inferred_cage_variance"], cage_variance)
        self.assertAlmostEqual(row["predicted_fs_k2"], expected_fs)

    def test_poisson_limit_is_continuous(self):
        value = self.analysis.gamma_poisson_count_pgf(
            mean_count=1.7,
            fano_factor=1.0,
            argument=0.4,
        )
        self.assertAlmostEqual(value, math.exp(1.7 * (0.4 - 1.0)))

    def test_negative_cage_variance_is_unsupported_without_clipping(self):
        geometry = {
            "temperature": "0.45",
            "replicate": "1",
            "lag": "20",
            "heldout_msd": "0.1",
            "heldout_ngp": "10.0",
            "jump_msd": "1.0",
            "jump_component_fourth_moment": "0.01",
            "jump_characteristic_k2": "0.8",
            "jump_characteristic_k4": "0.6",
            "jump_characteristic_k7p25": "0.4",
            "observed_fs_k2": "0.7",
            "observed_fs_k4": "0.5",
            "observed_fs_k7p25": "0.3",
        }
        macro = {
            "replicate": "1",
            "lag": "20",
            "predicted_mean_count": "1.0",
            "predicted_count_variance": "1.0",
        }

        row = self.analysis.count_overdispersed_geometry_row(geometry, macro)

        self.assertEqual(row["supported"], 0.0)
        self.assertLess(row["inferred_cage_variance"], 0.0)
        self.assertTrue(math.isnan(row["predicted_fs_k2"]))

    def test_committed_tables_fix_moment_support_but_not_high_k_shape(self):
        rows, gates = self.analysis.analyze_committed_tables(ROOT)
        by_temperature = {float(row["temperature"]): row for row in gates}
        low = by_temperature[0.45]
        high = by_temperature[0.58]

        self.assertEqual(len(rows), 46)
        self.assertEqual(float(low["supported_row_count"]), 20.0)
        self.assertEqual(float(low["row_count"]), 21.0)
        self.assertAlmostEqual(
            float(low["fs_k7p25_max_absolute_error"]),
            0.04108845937739283,
            places=12,
        )
        self.assertEqual(float(low["curve_transfer_pass"]), 0.0)
        self.assertEqual(float(high["supported_row_count"]), 25.0)
        self.assertEqual(float(high["row_count"]), 25.0)
        self.assertAlmostEqual(
            float(high["fs_k7p25_max_absolute_error"]),
            0.04664285739962798,
            places=12,
        )
        self.assertEqual(float(high["curve_transfer_pass"]), 0.0)
        self.assertEqual(float(high["high_temperature_canary_only"]), 1.0)

        for gate in gates:
            for key in self.analysis.CLAIM_FLAGS:
                self.assertEqual(float(gate[key]), 0.0)


if __name__ == "__main__":
    unittest.main()
