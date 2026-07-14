import sys
import importlib.util
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


def load_script_module(filename: str, module_name: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def passing_transfer_rows(
    *,
    model: str = "anchor_aware_semi_markov",
    replicate_count: int = 2,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    quality_rows = []
    for replicate in range(1, replicate_count + 1):
        for realization in range(16):
            quality_rows.append(
                {
                    "model": model,
                    "replicate": float(replicate),
                    "realization": float(realization),
                    "temperature": 0.45,
                    "calibration_time": 5000.0,
                    "block_size": 20.0,
                    "radial_bin_count": 8.0,
                    "required_realizations_per_replicate": 16.0,
                    "surrogate_base_seed": 84031.0,
                    "scheduled_return_fraction_absolute_error": 0.01,
                    "scheduled_return_given_return_absolute_error": 0.02,
                    "scheduled_return_given_escape_absolute_error": 0.02,
                    "return_holding_time_mean_relative_error": 0.04,
                    "escape_holding_time_mean_relative_error": 0.04,
                    "return_holding_time_quantile_maximum_relative_error": 0.08,
                    "escape_holding_time_quantile_maximum_relative_error": 0.08,
                    "radial_mean_relative_error": 0.01,
                    "radial_standard_deviation_relative_error": 0.01,
                    "lag_one_cosine_mean_absolute_error": 0.01,
                    "lag_one_cosine_quantile_maximum_absolute_error": 0.02,
                    "geometric_return_fraction_absolute_error": 0.01,
                    "return_closure_quantile_maximum_error_over_dw": 0.04,
                    "geometric_return_quality_required": float(
                        model == "anchor_aware_semi_markov"
                    ),
                    "unsupported_tuple_count": 0.0,
                    "calibration_events_only": 1.0,
                    "heldout_events_used_in_calibration": 0.0,
                    "heldout_cage_residual_used_in_prediction": 0.0,
                    "macro_fit_parameter_count": 0.0,
                    "microdynamic_closure_claim_allowed": 0.0,
                    "spatial_facilitation_claim_allowed": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
    summary_rows = []
    for lag in (20.0, 40.0):
        summary_rows.append(
            {
                "model": model,
                "lag": lag,
                "independent_replicate_count": float(replicate_count),
                "ensemble_msd_relative_error": 0.05,
                "ensemble_ngp_absolute_error": 0.10,
                "ensemble_absolute_error_fs_k2": 0.01,
                "ensemble_absolute_error_fs_k4": 0.02,
                "ensemble_absolute_error_fs_k7p25": 0.02,
                "ensemble_msd_mc_relative_se": 0.005,
                "ensemble_ngp_mc_se": 0.01,
                "ensemble_fs_k2_mc_se": 0.001,
                "ensemble_fs_k4_mc_se": 0.001,
                "ensemble_fs_k7p25_mc_se": 0.002,
            }
        )
    replicate_rows = []
    for replicate in range(1, replicate_count + 1):
        for lag in (20.0, 40.0):
            replicate_rows.append(
                {
                    "model": model,
                    "replicate": float(replicate),
                    "lag": lag,
                    "msd_relative_error": 0.05,
                    "ngp_absolute_error": 0.10,
                    "absolute_error_fs_k2": 0.01,
                    "absolute_error_fs_k4": 0.02,
                    "absolute_error_fs_k7p25": 0.02,
                }
            )
    return quality_rows, summary_rows, replicate_rows


class AnchorTransferAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_script_module(
            "analyze_ka_anchor_semi_markov_transfer.py",
            "analyze_ka_anchor_semi_markov_transfer_test",
        )

    def test_classifier_accepts_complete_quality_precision_and_curves(self):
        quality, summary, replicates = passing_transfer_rows()
        verdict = self.module.classify_anchor_transfer(
            quality,
            summary,
            replicates,
            model="anchor_aware_semi_markov",
            expected_replicates=2,
        )
        self.assertEqual(verdict["quality_realization_completeness_pass"], 1.0)
        self.assertEqual(verdict["quality_pass"], 1.0)
        self.assertEqual(verdict["precision_pass"], 1.0)
        self.assertEqual(verdict["curve_transfer_pass"], 1.0)
        self.assertEqual(verdict["mechanism_state"], "curve_closed")

    def test_each_shared_quality_limit_can_reject_transfer(self):
        cases = {
            "scheduled_return_fraction_absolute_error": 0.021,
            "scheduled_return_given_return_absolute_error": 0.031,
            "scheduled_return_given_escape_absolute_error": 0.031,
            "return_holding_time_mean_relative_error": 0.051,
            "escape_holding_time_mean_relative_error": 0.051,
            "return_holding_time_quantile_maximum_relative_error": 0.101,
            "escape_holding_time_quantile_maximum_relative_error": 0.101,
            "radial_mean_relative_error": 0.021,
            "radial_standard_deviation_relative_error": 0.021,
            "lag_one_cosine_mean_absolute_error": 0.021,
            "lag_one_cosine_quantile_maximum_absolute_error": 0.031,
            "unsupported_tuple_count": 1.0,
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                quality, summary, replicates = passing_transfer_rows()
                quality[0][key] = value
                verdict = self.module.classify_anchor_transfer(
                    quality,
                    summary,
                    replicates,
                    model="anchor_aware_semi_markov",
                    expected_replicates=2,
                )
                self.assertEqual(verdict["quality_pass"], 0.0)
                self.assertEqual(verdict["curve_transfer_pass"], 0.0)

    def test_geometry_limits_apply_only_to_anchor_model(self):
        for key, value in (
            ("geometric_return_fraction_absolute_error", 0.021),
            ("return_closure_quantile_maximum_error_over_dw", 0.051),
        ):
            anchor_quality, anchor_summary, anchor_replicates = passing_transfer_rows()
            anchor_quality[0][key] = value
            anchor = self.module.classify_anchor_transfer(
                anchor_quality,
                anchor_summary,
                anchor_replicates,
                model="anchor_aware_semi_markov",
                expected_replicates=2,
            )
            self.assertEqual(anchor["quality_pass"], 0.0)

            control_quality, control_summary, control_replicates = passing_transfer_rows(
                model="state_schedule_without_anchor_geometry"
            )
            control_quality[0][key] = 10.0
            control = self.module.classify_anchor_transfer(
                control_quality,
                control_summary,
                control_replicates,
                model="state_schedule_without_anchor_geometry",
                expected_replicates=2,
            )
            self.assertEqual(control["quality_pass"], 1.0)

    def test_precision_and_each_curve_class_are_independent_gates(self):
        for key, value, expected_gate in (
            ("ensemble_msd_mc_relative_se", 0.011, "precision_pass"),
            ("ensemble_ngp_mc_se", 0.031, "precision_pass"),
            ("ensemble_fs_k7p25_mc_se", 0.0031, "precision_pass"),
            ("ensemble_msd_relative_error", 0.101, "raw_curve_transfer_pass"),
            ("ensemble_ngp_absolute_error", 0.301, "raw_curve_transfer_pass"),
            ("ensemble_absolute_error_fs_k4", 0.031, "raw_curve_transfer_pass"),
        ):
            with self.subTest(key=key):
                quality, summary, replicates = passing_transfer_rows()
                summary[0][key] = value
                verdict = self.module.classify_anchor_transfer(
                    quality,
                    summary,
                    replicates,
                    model="anchor_aware_semi_markov",
                    expected_replicates=2,
                )
                self.assertEqual(verdict[expected_gate], 0.0)
                self.assertEqual(verdict["curve_transfer_pass"], 0.0)

    def test_provenance_and_exact_realization_grid_are_mandatory(self):
        quality, summary, replicates = passing_transfer_rows()
        for mutation in ("missing", "duplicate", "calibration", "claim"):
            with self.subTest(mutation=mutation):
                changed = [dict(row) for row in quality]
                if mutation == "missing":
                    changed.pop()
                elif mutation == "duplicate":
                    changed.append(dict(changed[0]))
                elif mutation == "calibration":
                    changed[0]["calibration_time"] = 4999.0
                else:
                    changed[0]["thermodynamic_claim_allowed"] = 1.0
                with self.assertRaises(ValueError):
                    self.module.classify_anchor_transfer(
                        changed,
                        summary,
                        replicates,
                        model="anchor_aware_semi_markov",
                        expected_replicates=2,
                    )

    def test_protocol_is_frozen_by_temperature(self):
        self.module.validate_anchor_protocol(
            temperature=0.45,
            calibration_time=5000,
            block_size=20,
            radial_bin_count=8,
            surrogate_realizations=16,
            replicate_count=3,
        )
        for key, value in (
            ("calibration_time", 4999),
            ("block_size", 10),
            ("radial_bin_count", 7),
            ("surrogate_realizations", 15),
            ("replicate_count", 2),
        ):
            arguments = {
                "temperature": 0.45,
                "calibration_time": 5000,
                "block_size": 20,
                "radial_bin_count": 8,
                "surrogate_realizations": 16,
                "replicate_count": 3,
            }
            arguments[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.module.validate_anchor_protocol(**arguments)


if __name__ == "__main__":
    unittest.main()
