import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class NonlinearBathDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def controls(**updates):
        from nonlinear_bath_gle import NonlinearBathControls

        values = {
            "temperature": 0.58,
            "friction": 1.0,
            "period": 1.0,
            "barrier": 0.0,
            "rates": np.array([0.0]),
            "amplitudes": np.array([1.0]),
            "modulation": np.array([0.0]),
            "phases": np.array([0.0]),
            "time_step": 0.01,
        }
        values.update(updates)
        return NonlinearBathControls(**values)

    def test_equilibrium_diagnostic_recognizes_uniform_gibbs_null(self):
        from nonlinear_bath_diagnostics import equilibrium_diagnostics

        controls = self.controls()
        bin_centers = (np.arange(40) + 0.5) / 40.0 - 0.5
        position = np.tile(bin_centers, 500)
        rng = np.random.default_rng(7)
        momentum = rng.normal(scale=np.sqrt(controls.temperature), size=len(position))
        auxiliary = rng.normal(
            scale=np.sqrt(controls.temperature),
            size=(len(position), 1),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = equilibrium_diagnostics(
                position,
                momentum,
                auxiliary,
                controls=controls,
                position_bin_count=40,
            )
        self.assertLess(result["momentum_temperature_relative_error"], 0.03)
        self.assertLess(result["maximum_auxiliary_temperature_relative_error"], 0.03)
        self.assertLess(result["maximum_momentum_auxiliary_correlation"], 0.03)
        self.assertLess(result["position_gibbs_total_variation"], 1e-12)
        self.assertEqual(result["position_coordinate"], "wrapped_modulo_period")
        self.assertEqual(result["unwrapped_position_gibbs_probability_allowed"], 0.0)
        self.assertEqual(result["equilibrium_gate_pass"], 1.0)

    def test_fixed_path_bath_replay_matches_constant_kernel(self):
        from nonlinear_bath_diagnostics import bath_replay_covariance

        controls = self.controls()
        time_count, path_count, replay_count = 20, 12, 8
        positions = np.broadcast_to(
            (np.arange(path_count) + 0.5) / path_count - 0.5,
            (time_count, path_count),
        )
        signs = np.array([1.0, -1.0] * (replay_count // 2))
        replay_forces = (
            3.0
            + signs[:, None, None]
            * np.sqrt(controls.temperature)
            * np.ones((replay_count, time_count, path_count))
        )
        result = bath_replay_covariance(
            positions,
            replay_forces,
            lag_steps=np.array([1, 5, 10]),
            controls=controls,
            position_bin_count=12,
        )
        np.testing.assert_allclose(result["empirical_covariance"], controls.temperature)
        np.testing.assert_allclose(result["exact_covariance"], controls.temperature)
        self.assertLess(result["pooled_normalized_rmse"], 1e-12)
        self.assertEqual(result["bath_replay_gate_pass"], 1.0)

    def test_periodic_event_filter_rejects_recrossing_and_records_waits(self):
        from nonlinear_bath_diagnostics import accepted_periodic_cage_events

        position = np.array(
            [
                [0.00, 0.00],
                [0.20, 0.10],
                [0.55, 0.45],
                [0.49, 0.60],
                [0.20, 0.80],
                [0.10, 1.05],
                [0.15, 1.10],
                [0.20, 1.15],
                [0.25, 1.20],
            ]
        )
        records = accepted_periodic_cage_events(
            position,
            sample_time=0.05,
            period=1.0,
            persistence=0.15,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["trajectory_index"], 1.0)
        self.assertEqual(records[0]["from_cage"], 0.0)
        self.assertEqual(records[0]["to_cage"], 1.0)
        self.assertAlmostEqual(records[0]["event_time"], 0.15)
        self.assertAlmostEqual(records[0]["waiting_time"], 0.15)

    def test_hazard_fits_use_censored_exposure_and_delayed_grid(self):
        from nonlinear_bath_diagnostics import (
            fit_constant_hazard,
            fit_delayed_square_hazard,
        )

        waits = np.array([2.0, 2.5, 3.0, 0.20, 0.25, 0.30])
        censored = np.array([False, False, False, True, True, True])
        constant = fit_constant_hazard(waits, censored=censored)
        self.assertAlmostEqual(constant["rate"], 3.0 / np.sum(waits))
        delayed = fit_delayed_square_hazard(waits, censored=censored)
        self.assertGreater(delayed["delay_time"], 0.0)
        self.assertLess(delayed["negative_log_likelihood"], constant["negative_log_likelihood"])
        self.assertEqual(delayed["fit_uses_held_samples"], 0.0)

    def test_classifier_separates_exact_bath_and_delayed_clock_claims(self):
        from nonlinear_bath_diagnostics import classify_nonlinear_bath_gate

        verdict = classify_nonlinear_bath_gate(
            maximum_reconstruction_relative_error=1e-12,
            half_step_equilibrium_not_worse=True,
            equilibrium_gate_pass=True,
            bath_replay_gate_pass=True,
            held_constant_negative_log_likelihood=10.0,
            held_delayed_negative_log_likelihood=8.0,
            held_constant_integrated_survival_error=0.20,
            held_delayed_integrated_survival_error=0.15,
            delay_time_bootstrap_ci95_low=0.10,
        )
        self.assertEqual(verdict["exact_nonlinear_bath_elimination_supported"], 1.0)
        self.assertEqual(verdict["synthetic_bath_level_fdt_replay_supported"], 1.0)
        self.assertEqual(verdict["synthetic_delayed_hazard_emerges"], 1.0)
        self.assertEqual(verdict["real_ka_position_dependent_kernel_authorized"], 1.0)
        self.assertEqual(verdict["positive_prony_kernel_identified_in_ka"], 0.0)
        self.assertEqual(verdict["finite_auxiliary_rank_identified_in_ka"], 0.0)
        self.assertEqual(verdict["oscillatory_matrix_bath_authorized"], 0.0)
        self.assertEqual(verdict["real_ka_kernel_identifiability_test_required"], 1.0)
        for claim in (
            "autonomous_single_particle_gle_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "spatial_facilitation_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(verdict[claim], 0.0)

        no_clock = classify_nonlinear_bath_gate(
            maximum_reconstruction_relative_error=1e-12,
            half_step_equilibrium_not_worse=True,
            equilibrium_gate_pass=True,
            bath_replay_gate_pass=True,
            held_constant_negative_log_likelihood=10.0,
            held_delayed_negative_log_likelihood=10.5,
            held_constant_integrated_survival_error=0.20,
            held_delayed_integrated_survival_error=0.19,
            delay_time_bootstrap_ci95_low=-0.01,
        )
        self.assertEqual(no_clock["synthetic_delayed_hazard_emerges"], 0.0)
        self.assertEqual(no_clock["real_ka_position_dependent_kernel_authorized"], 1.0)

        grid_boundary = classify_nonlinear_bath_gate(
            maximum_reconstruction_relative_error=1e-12,
            half_step_equilibrium_not_worse=True,
            equilibrium_gate_pass=True,
            bath_replay_gate_pass=True,
            held_constant_negative_log_likelihood=10.0,
            held_delayed_negative_log_likelihood=8.0,
            held_constant_integrated_survival_error=0.20,
            held_delayed_integrated_survival_error=0.15,
            delay_time_bootstrap_ci95_low=1e-3,
        )
        self.assertEqual(grid_boundary["synthetic_delayed_hazard_emerges"], 0.0)
        self.assertEqual(grid_boundary["delay_grid_boundary_excluded"], 0.0)


if __name__ == "__main__":
    unittest.main()
