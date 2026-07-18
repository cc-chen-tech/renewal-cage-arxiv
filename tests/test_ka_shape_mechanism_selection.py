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


def load_script(filename: str, module_name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShapeMechanismSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_script(
            "summarize_ka_shape_mechanism_selection.py",
            "summarize_ka_shape_mechanism_selection",
        )

    def test_pair_closures_reduce_to_iid_when_pair_factorizes(self):
        mean_count = 2.3
        fano_factor = 1.8
        single = 0.7
        independent = self.analysis.count_pgf(
            mean_count=mean_count,
            fano_factor=fano_factor,
            argument=single,
        )

        disjoint, disjoint_tail = self.analysis.event_characteristic(
            model="disjoint_pair",
            mean_count=mean_count,
            fano_factor=fano_factor,
            single_characteristic=single,
            pair_characteristic=single**2,
            empirical_characteristics=None,
        )
        eigenmode, eigenmode_tail = self.analysis.event_characteristic(
            model="pair_eigenmode",
            mean_count=mean_count,
            fano_factor=fano_factor,
            single_characteristic=single,
            pair_characteristic=single**2,
            empirical_characteristics=None,
        )

        self.assertAlmostEqual(disjoint, independent)
        self.assertAlmostEqual(eigenmode, independent)
        self.assertEqual(disjoint_tail, 0.0)
        self.assertEqual(eigenmode_tail, 0.0)

    def test_empirical_path_uses_count_pmf_and_reports_omitted_tail(self):
        value, tail = self.analysis.event_characteristic(
            model="empirical_path",
            mean_count=1.0,
            fano_factor=1.0,
            single_characteristic=0.5,
            pair_characteristic=0.25,
            empirical_characteristics=[1.0, 0.5, 0.25],
        )

        expected = math.exp(-1.0) * (1.0 + 0.5 + 0.25 / 2.0)
        self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(tail, 1.0 - 2.5 * math.exp(-1.0))

    def test_committed_common_grid_selects_shape_class_not_mechanism(self):
        rows, models, verdicts = self.analysis.analyze_committed_tables(ROOT)
        low_models = {
            row["model"]: row
            for row in models
            if math.isclose(float(row["temperature"]), 0.45)
        }
        low_verdict = next(
            row
            for row in verdicts
            if math.isclose(float(row["temperature"]), 0.45)
        )

        self.assertEqual(len(rows), 228)
        self.assertEqual(set(low_models), set(self.analysis.MODELS))
        for model in self.analysis.EVENT_MODELS:
            self.assertEqual(float(low_models[model]["all_k_pass"]), 0.0)
        self.assertEqual(float(low_models["gamma_variance_mixture"]["all_k_pass"]), 1.0)
        self.assertEqual(
            float(low_models["inverse_gaussian_variance_mixture"]["all_k_pass"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(low_models["independent_jump"]["maximum_absolute_error"]),
            0.0410884594,
            places=8,
        )
        self.assertAlmostEqual(
            float(low_models["disjoint_pair"]["maximum_absolute_error"]),
            0.0452145499,
            places=8,
        )
        self.assertAlmostEqual(
            float(low_models["empirical_path"]["maximum_absolute_error"]),
            0.1206499477,
            places=8,
        )
        self.assertEqual(
            low_verdict["analysis_status"],
            "variance_mixture_shape_survives_factorized_event_path_closures_fail",
        )
        self.assertEqual(
            float(
                low_verdict[
                    "tested_factorized_event_path_closures_excluded_on_common_grid"
                ]
            ),
            1.0,
        )
        self.assertEqual(
            float(low_verdict["positive_variance_mixture_shape_class_survives"]),
            1.0,
        )
        for flag in self.analysis.CLAIM_FLAGS:
            self.assertEqual(float(low_verdict[flag]), 0.0)

    def test_analysis_fails_closed_when_common_grid_is_incomplete(self):
        rows, _, _ = self.analysis.analyze_committed_tables(ROOT)
        truncated = [
            row
            for row in rows
            if not (
                math.isclose(float(row["temperature"]), 0.45)
                and int(float(row["replicate"])) == 1
                and int(float(row["lag"])) == 20
                and row["model"] == "empirical_path"
            )
        ]

        with self.assertRaisesRegex(ValueError, "common model grid"):
            self.analysis.summarize_models(truncated)

    def test_svg_reports_common_grid_errors_and_frozen_tolerance(self):
        _, models, _ = self.analysis.analyze_committed_tables(ROOT)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.svg"
            self.analysis.write_svg(path, models)
            svg = path.read_text()

        self.assertIn("Shape-class mechanism selection", svg)
        self.assertIn("frozen tolerance", svg)
        self.assertIn(">Empirical</text>", svg)
        self.assertIn(">path</text>", svg)
        self.assertIn(">Inverse-Gaussian</text>", svg)
        self.assertLess(svg.index('fill="#edf7f0"'), svg.index("frozen tolerance"))


if __name__ == "__main__":
    unittest.main()
