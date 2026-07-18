import importlib.util
import csv
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for directory in (SCRIPTS, SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def load_script(filename: str, module_name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class JumpGeometryExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_script(
            "extract_ka_calibration_jump_geometry.py",
            "extract_ka_calibration_jump_geometry",
        )

    def test_jump_geometry_statistics_uses_component_characteristic(self):
        jumps = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

        row = self.analysis.jump_geometry_statistics(jumps, np.array([2.0]))

        self.assertAlmostEqual(row["event_count"], 2.0)
        self.assertAlmostEqual(row["jump_msd"], 1.0)
        self.assertAlmostEqual(row["jump_radial_fourth_moment"], 1.0)
        self.assertAlmostEqual(row["jump_component_fourth_moment"], 1.0 / 3.0)
        self.assertAlmostEqual(
            row["jump_characteristic_k2"],
            (math.cos(2.0) + 2.0) / 3.0,
        )

    def test_statistics_reject_invalid_or_empty_jump_arrays(self):
        invalid = (
            np.empty((0, 3)),
            np.ones((3, 2)),
            np.array([[math.nan, 0.0, 0.0]]),
        )
        for jumps in invalid:
            with self.subTest(shape=jumps.shape):
                with self.assertRaises(ValueError):
                    self.analysis.jump_geometry_statistics(jumps, np.array([2.0]))
        with self.assertRaises(ValueError):
            self.analysis.jump_geometry_statistics(
                np.ones((3, 3)),
                np.array([0.0]),
            )

    def test_extractor_reads_only_the_calibration_prefix(self):
        positions = np.zeros((11, 2, 3), dtype=float)
        events = {"jump_vector": np.array([[0.2, -0.1, 0.3]])}
        times = np.arange(3)
        activity = np.zeros((3, 2), dtype=float)

        with mock.patch.object(
            self.analysis,
            "position_fluctuation_values",
            return_value=(times, activity),
        ) as fluctuation, mock.patch.object(
            self.analysis,
            "extract_debye_waller_cage_jumps",
            return_value=events,
        ) as extract:
            row = self.analysis.extract_calibration_jump_geometry(
                positions,
                calibration_time=5,
                debye_waller_factor=0.03,
                half_window=1,
                wave_numbers=np.array([2.0, 4.0, 7.25]),
            )

        self.assertEqual(fluctuation.call_args.args[0].shape, (6, 2, 3))
        self.assertEqual(extract.call_args.args[0].shape, (6, 2, 3))
        self.assertEqual(row["calibration_frame_count"], 6.0)
        self.assertEqual(row["calibration_events_only"], 1.0)
        self.assertEqual(row["heldout_events_used"], 0.0)

    def test_trajectory_loader_stops_after_the_calibration_prefix(self):
        trajectory = {
            "unwrapped_positions": np.zeros((6, 3, 3), dtype=float),
            "particle_types": np.array([0, 1, 0]),
        }
        with mock.patch.object(
            self.analysis,
            "load_lammps_custom_trajectory",
            return_value=trajectory,
        ) as loader:
            positions = self.analysis.load_calibration_type_a_positions(
                Path("trajectory.lammpstrj"),
                calibration_time=5,
            )

        loader.assert_called_once_with(
            Path("trajectory.lammpstrj"),
            maximum_frame_count=6,
        )
        self.assertEqual(positions.shape, (6, 2, 3))


class GeometryQuotientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_script(
            "summarize_ka_activated_cage_geometry.py",
            "summarize_ka_activated_cage_geometry",
        )

    def test_empirical_geometry_quotient_matches_compound_poisson_formula(self):
        result = self.summary.empirical_geometry_quotient(
            msd=0.6,
            ngp=0.5,
            jump_msd=0.2,
            jump_component_fourth_moment=0.04,
            jump_characteristic={2.0: 0.8},
        )

        self.assertEqual(result["supported"], 1.0)
        self.assertAlmostEqual(result["mean_event_count"], 1.5)
        self.assertAlmostEqual(result["cage_variance"], 0.05)
        self.assertAlmostEqual(result["predicted_fs"][2.0], math.exp(-0.5))

    def test_empirical_geometry_quotient_rejects_negative_cage_variance(self):
        result = self.summary.empirical_geometry_quotient(
            msd=0.1,
            ngp=10.0,
            jump_msd=1.0,
            jump_component_fourth_moment=0.01,
            jump_characteristic={2.0: 0.5},
        )
        self.assertEqual(result["supported"], 0.0)
        self.assertGreater(result["mean_event_count"], 0.0)
        self.assertLess(result["cage_variance"], 0.0)
        self.assertEqual(result["predicted_fs"], {})

    def test_fixed_length_null_has_only_three_supported_t045_rows(self):
        data = ROOT / "data"
        with (data / "renewal_cage_ka_replicates_T045_debye_waller_heldout_replicates.csv").open() as handle:
            jump_msd = {
                int(float(row["replicate"])): float(row["dw_mean_jump_squared"])
                for row in csv.DictReader(handle)
            }
        with (data / "renewal_cage_ka_gamma_variance_mixture_rows.csv").open() as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if float(row["temperature"]) == 0.45
            ]

        supported = sum(
            self.summary.fixed_length_geometry_quotient(
                msd=float(row["heldout_msd"]),
                ngp=float(row["heldout_ngp"]),
                jump_msd=jump_msd[int(row["replicate"])],
                wave_numbers=(2.0, 4.0, 7.25),
            )["supported"]
            for row in rows
        )

        self.assertEqual(supported, 3.0)
        self.assertEqual(len(rows), 21)

    def test_gate_requires_primary_support_curves_and_all_replicates(self):
        rows = []
        for temperature, replicate_count in ((0.45, 3), (0.58, 5)):
            for replicate in range(1, replicate_count + 1):
                rows.append(
                    {
                        "temperature": temperature,
                        "replicate": replicate,
                        "lag": 20,
                        "empirical_geometry_supported": 1.0,
                        "fixed_length_geometry_supported": 0.0,
                        "empirical_fs_k2_absolute_error": 0.01,
                        "empirical_fs_k4_absolute_error": 0.02,
                        "empirical_fs_k7p25_absolute_error": 0.029,
                    }
                )

        gates = self.summary.classify_geometry_gate(
            rows,
            stationarity_pass={0.45: True, 0.58: False},
            provenance_pass={0.45: True, 0.58: True},
        )
        low = next(row for row in gates if row["temperature"] == 0.45)
        high = next(row for row in gates if row["temperature"] == 0.58)

        self.assertEqual(low["empirical_support_coverage_pass"], 1.0)
        self.assertEqual(low["all_primary_replicates_supported"], 1.0)
        self.assertEqual(low["curve_transfer_pass"], 1.0)
        self.assertEqual(
            low["empirical_activated_jump_geometry_supported_exploratory"],
            1.0,
        )
        self.assertEqual(high["high_temperature_canary_only"], 1.0)
        self.assertEqual(high["source_stationarity_pass"], 0.0)
        self.assertEqual(
            high["empirical_activated_jump_geometry_supported_exploratory"],
            0.0,
        )
        for gate in gates:
            for flag in self.summary.STRONG_ZERO_FLAGS:
                self.assertEqual(gate[flag], 0.0)

    def test_gate_fails_closed_when_one_primary_replicate_has_no_support(self):
        rows = [
            {
                "temperature": 0.45,
                "replicate": replicate,
                "lag": lag,
                "empirical_geometry_supported": float(replicate != 3),
                "fixed_length_geometry_supported": 0.0,
                "empirical_fs_k2_absolute_error": 0.0,
                "empirical_fs_k4_absolute_error": 0.0,
                "empirical_fs_k7p25_absolute_error": 0.0,
            }
            for replicate in (1, 2, 3)
            for lag in (20, 100, 200, 500, 1000)
        ]

        gate = self.summary.classify_geometry_gate(
            rows,
            stationarity_pass={0.45: True},
            provenance_pass={0.45: True},
        )[0]

        self.assertEqual(gate["empirical_support_fraction"], 2.0 / 3.0)
        self.assertEqual(gate["empirical_support_coverage_pass"], 0.0)
        self.assertEqual(gate["all_primary_replicates_supported"], 0.0)
        self.assertEqual(
            gate["empirical_activated_jump_geometry_supported_exploratory"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
