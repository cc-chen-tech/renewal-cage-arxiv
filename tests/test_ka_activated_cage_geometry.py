import importlib.util
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


if __name__ == "__main__":
    unittest.main()
