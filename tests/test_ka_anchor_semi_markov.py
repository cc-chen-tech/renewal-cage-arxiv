import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ka_anchor_semi_markov import (  # noqa: E402
    anchor_path_quality,
    extract_anchor_transition_kernel,
    simulate_anchor_semi_markov,
)


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


def compact_simulation_kernel(*, closure_distance: float = 0.1) -> dict[str, object]:
    profiles = {
        particle: {
            "event_count": 8,
            "mean_wait": 2.0 + 0.1 * particle,
            "jump_radii": np.array([0.96, 0.98, 1.0, 1.02, 1.04]),
            "initial_state": particle % 2,
            "initial_vector": np.array([1.0, 0.0, 0.0]),
        }
        for particle in range(64)
    }
    return {
        "particle": np.array([0, 0, 0, 0], dtype=int),
        "source_event_index": np.array([1, 2, 3, 4], dtype=int),
        "target_event_index": np.array([2, 3, 4, 5], dtype=int),
        "current_state": np.array([0, 0, 1, 1], dtype=int),
        "next_state": np.array([0, 1, 0, 1], dtype=int),
        "holding_time": np.array([1.0, 3.0, 1.5, 2.5]),
        "normalized_holding_time": np.array([0.5, 1.5, 0.75, 1.25]),
        "source_radius": np.array([1.0, 0.96, 0.98, 1.02]),
        "target_radius": np.array([0.96, 0.98, 1.02, 1.04]),
        "source_radius_bin": np.zeros(4, dtype=int),
        "target_radius_quantile": np.array([0.1, 0.3, 0.7, 0.9]),
        "relative_cosine": np.array([-0.2, -0.8, 0.3, -0.95]),
        "closure_distance": np.array([1.2, closure_distance, 1.4, closure_distance]),
        "debye_waller_factor": 0.04,
        "radius_threshold": 0.2,
        "radial_bin_count": 1,
        "transition_count": 4,
        "particle_profiles": profiles,
        "active_particle_ids": tuple(profiles),
        "non_propagating_particle_ids": (),
    }


class AnchorSemiMarkovSimulationTests(unittest.TestCase):
    def test_fixed_seed_is_deterministic_and_return_geometry_is_exact(self):
        kernel = compact_simulation_kernel()
        first = simulate_anchor_semi_markov(
            kernel,
            np.random.default_rng(71),
            duration=80,
            maximum_lag=20,
            model="anchor_aware_semi_markov",
        )
        second = simulate_anchor_semi_markov(
            kernel,
            np.random.default_rng(71),
            duration=80,
            maximum_lag=20,
            model="anchor_aware_semi_markov",
        )

        self.assertEqual(set(first), set(second))
        for key in first:
            np.testing.assert_allclose(first[key], second[key])
        returned = np.asarray(first["scheduled_state"]) == 1
        closure = np.linalg.norm(
            np.asarray(first["source_vector"])[returned]
            + np.asarray(first["jump_vector"])[returned],
            axis=1,
        )
        np.testing.assert_allclose(
            closure,
            np.asarray(first["sampled_closure_distance"])[returned],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        self.assertTrue(np.all(np.asarray(first["geometric_return"])[returned] == 1))
        self.assertEqual(float(first["unsupported_tuple_count"][0]), 0.0)

    def test_control_shares_state_wait_and_radius_draws_but_removes_anchor_geometry(self):
        kernel = compact_simulation_kernel()
        anchor = simulate_anchor_semi_markov(
            kernel,
            np.random.default_rng(118),
            duration=100,
            maximum_lag=20,
            model="anchor_aware_semi_markov",
        )
        control = simulate_anchor_semi_markov(
            kernel,
            np.random.default_rng(118),
            duration=100,
            maximum_lag=20,
            model="state_schedule_without_anchor_geometry",
        )

        for key in ("particle", "time", "scheduled_state", "holding_time", "target_radius"):
            np.testing.assert_allclose(anchor[key], control[key])
        scheduled_return = np.asarray(control["scheduled_state"]) == 1
        control_closure = np.linalg.norm(
            np.asarray(control["source_vector"])[scheduled_return]
            + np.asarray(control["jump_vector"])[scheduled_return],
            axis=1,
        )
        self.assertGreater(
            float(np.max(np.abs(control_closure - np.asarray(control["sampled_closure_distance"])[scheduled_return]))),
            0.05,
        )

    def test_generated_azimuths_are_three_dimensional(self):
        synthetic = simulate_anchor_semi_markov(
            compact_simulation_kernel(),
            np.random.default_rng(991),
            duration=200,
            maximum_lag=20,
            model="anchor_aware_semi_markov",
        )
        jumps = np.asarray(synthetic["jump_vector"])
        self.assertGreater(len(jumps), 1000)
        self.assertGreater(float(np.std(jumps[:, 1])), 0.2)
        self.assertGreater(float(np.std(jumps[:, 2])), 0.2)

    def test_impossible_return_support_and_invalid_arguments_fail(self):
        impossible = compact_simulation_kernel(closure_distance=3.0)
        impossible["current_state"] = np.zeros(4, dtype=int)
        impossible["next_state"] = np.ones(4, dtype=int)
        for profile in impossible["particle_profiles"].values():
            profile["initial_state"] = 0
        with self.assertRaises(ValueError):
            simulate_anchor_semi_markov(
                impossible,
                np.random.default_rng(4),
                duration=20,
                maximum_lag=5,
                model="anchor_aware_semi_markov",
            )
        for rng, duration, maximum_lag, model in (
            (object(), 20, 5, "anchor_aware_semi_markov"),
            (np.random.default_rng(1), 0, 5, "anchor_aware_semi_markov"),
            (np.random.default_rng(1), 20, -1, "anchor_aware_semi_markov"),
            (np.random.default_rng(1), 20, 5, "unknown"),
        ):
            with self.assertRaises(ValueError):
                simulate_anchor_semi_markov(
                    compact_simulation_kernel(),
                    rng,
                    duration=duration,
                    maximum_lag=maximum_lag,
                    model=model,
                )

    def test_quality_gates_anchor_geometry_only_for_anchor_model(self):
        kernel = compact_simulation_kernel()
        anchor = simulate_anchor_semi_markov(
            kernel,
            np.random.default_rng(515),
            duration=1000,
            maximum_lag=20,
            model="anchor_aware_semi_markov",
        )
        control = simulate_anchor_semi_markov(
            kernel,
            np.random.default_rng(515),
            duration=1000,
            maximum_lag=20,
            model="state_schedule_without_anchor_geometry",
        )
        anchor_quality = anchor_path_quality(kernel, anchor, model="anchor_aware_semi_markov")
        control_quality = anchor_path_quality(
            kernel,
            control,
            model="state_schedule_without_anchor_geometry",
        )

        self.assertEqual(anchor_quality["geometric_return_quality_required"], 1.0)
        self.assertEqual(control_quality["geometric_return_quality_required"], 0.0)
        self.assertLess(anchor_quality["geometric_return_fraction_absolute_error"], 0.05)
        self.assertGreater(control_quality["geometric_return_fraction_absolute_error"], 0.05)
        self.assertTrue(np.isfinite(anchor_quality["return_closure_quantile_maximum_error_over_dw"]))


if __name__ == "__main__":
    unittest.main()
