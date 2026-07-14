import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ka_anchor_semi_markov import extract_anchor_transition_kernel  # noqa: E402


def exact_anchor_events() -> dict[str, np.ndarray]:
    particle = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    time = np.array([1, 3, 8, 13, 2, 5, 9, 15], dtype=int)
    pre_center = np.array(
        [
            [0.00, 0.0, 0.0],
            [1.00, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [2.00, 0.0, 0.0],
            [10.00, 0.0, 0.0],
            [11.00, 0.0, 0.0],
            [10.04, 0.0, 0.0],
            [12.00, 0.0, 0.0],
        ],
        dtype=float,
    )
    post_center = np.array(
        [
            [1.00, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [2.00, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [11.00, 0.0, 0.0],
            [10.04, 0.0, 0.0],
            [12.00, 0.0, 0.0],
            [10.02, 0.0, 0.0],
        ],
        dtype=float,
    )
    return {
        "particle": particle,
        "time": time,
        "pre_center": pre_center,
        "post_center": post_center,
        "jump_vector": post_center - pre_center,
    }


class AnchorTransitionExtractionTests(unittest.TestCase):
    def test_exact_cage_centers_define_return_escape_records(self):
        kernel = extract_anchor_transition_kernel(
            exact_anchor_events(),
            debye_waller_factor=0.01,
            radial_bin_count=8,
        )

        np.testing.assert_array_equal(kernel["particle"], [0, 0, 1, 1])
        np.testing.assert_array_equal(kernel["source_event_index"], [1, 2, 5, 6])
        np.testing.assert_array_equal(kernel["current_state"], [1, 0, 1, 0])
        np.testing.assert_array_equal(kernel["next_state"], [0, 1, 0, 1])
        np.testing.assert_allclose(kernel["holding_time"], [5.0, 5.0, 4.0, 6.0])
        np.testing.assert_allclose(kernel["closure_distance"], [1.0, 0.01, 1.0, 0.02])
        self.assertTrue(np.all(np.asarray(kernel["source_radius_bin"]) >= 0))
        self.assertTrue(np.all(np.asarray(kernel["source_radius_bin"]) < 8))
        self.assertEqual(kernel["transition_count"], 4)
        self.assertEqual(kernel["active_particle_ids"], (0, 1))
        self.assertEqual(kernel["non_propagating_particle_ids"], ())

    def test_particle_profiles_preserve_wait_and_radius_scales(self):
        kernel = extract_anchor_transition_kernel(
            exact_anchor_events(),
            debye_waller_factor=0.01,
            radial_bin_count=8,
        )

        profiles = kernel["particle_profiles"]
        self.assertEqual(set(profiles), {0, 1})
        self.assertAlmostEqual(profiles[0]["mean_wait"], 4.0)
        self.assertAlmostEqual(profiles[1]["mean_wait"], 13.0 / 3.0)
        np.testing.assert_allclose(
            profiles[0]["jump_radii"],
            np.sort(np.linalg.norm(exact_anchor_events()["jump_vector"][:4], axis=1)),
        )
        self.assertEqual(profiles[0]["initial_state"], 1)
        np.testing.assert_allclose(
            profiles[0]["initial_vector"],
            exact_anchor_events()["jump_vector"][1],
        )
        expected = np.array([5.0 / 4.0, 5.0 / 4.0, 4.0 / (13.0 / 3.0), 6.0 / (13.0 / 3.0)])
        np.testing.assert_allclose(kernel["normalized_holding_time"], expected)

    def test_sparse_particles_are_reported_not_joined_to_neighbors(self):
        events = exact_anchor_events()
        events = {
            key: np.concatenate((value, value[-1:] + (2 if key == "particle" else 0)), axis=0)
            if key in {"particle", "time"}
            else np.concatenate((value, value[-1:]), axis=0)
            for key, value in events.items()
        }
        events["particle"][-1] = 3
        events["time"][-1] = 1
        kernel = extract_anchor_transition_kernel(
            events,
            debye_waller_factor=0.01,
            radial_bin_count=8,
        )

        self.assertEqual(kernel["transition_count"], 4)
        self.assertEqual(kernel["active_particle_ids"], (0, 1))
        self.assertEqual(kernel["non_propagating_particle_ids"], (3,))

    def test_invalid_event_arrays_and_parameters_fail(self):
        events = exact_anchor_events()
        missing = dict(events)
        missing.pop("post_center")
        with self.assertRaises(ValueError):
            extract_anchor_transition_kernel(
                missing,
                debye_waller_factor=0.01,
                radial_bin_count=8,
            )

        unsorted = {key: value.copy() for key, value in events.items()}
        unsorted["time"][2] = 2
        with self.assertRaises(ValueError):
            extract_anchor_transition_kernel(
                unsorted,
                debye_waller_factor=0.01,
                radial_bin_count=8,
            )

        interleaved = {key: value[[0, 4, 1, 2, 3, 5, 6, 7]] for key, value in events.items()}
        with self.assertRaises(ValueError):
            extract_anchor_transition_kernel(
                interleaved,
                debye_waller_factor=0.01,
                radial_bin_count=8,
            )

        for debye_waller_factor, radial_bin_count in ((0.0, 8), (0.01, 0), (0.01, True)):
            with self.assertRaises(ValueError):
                extract_anchor_transition_kernel(
                    events,
                    debye_waller_factor=debye_waller_factor,
                    radial_bin_count=radial_bin_count,
                )


if __name__ == "__main__":
    unittest.main()
