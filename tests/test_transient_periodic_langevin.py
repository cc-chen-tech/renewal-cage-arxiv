import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from transient_periodic_langevin import (  # noqa: E402
    TransientPeriodicParams,
    conservative_forces,
    displacement_observables,
    event_clock_statistics,
    potential_energy,
    simulate_transient_periodic_langevin,
    stable_cage_events,
)


def load_script(filename: str, module_name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TransientPotentialTests(unittest.TestCase):
    def setUp(self):
        self.params = TransientPeriodicParams(
            temperature=0.7,
            period=1.3,
            base_barrier=2.5,
            elastic_stiffness=0.8,
            barrier_stiffness=1.1,
            barrier_coupling=0.6,
            gamma_x=1.2,
            gamma_q=4.0,
            gamma_z=3.0,
        )

    def test_forces_are_negative_potential_gradients(self):
        x = np.array([[0.17, -0.28]])
        q = np.array([[-0.04, 0.09]])
        z = np.array([0.31])
        force_x, force_q, force_z = conservative_forces(x, q, z, self.params)
        epsilon = 1e-6

        for coordinate in range(x.shape[1]):
            plus = x.copy()
            minus = x.copy()
            plus[0, coordinate] += epsilon
            minus[0, coordinate] -= epsilon
            derivative = (
                potential_energy(plus, q, z, self.params)[0]
                - potential_energy(minus, q, z, self.params)[0]
            ) / (2.0 * epsilon)
            self.assertAlmostEqual(force_x[0, coordinate], -derivative, delta=2e-6)

        for coordinate in range(q.shape[1]):
            plus = q.copy()
            minus = q.copy()
            plus[0, coordinate] += epsilon
            minus[0, coordinate] -= epsilon
            derivative = (
                potential_energy(x, plus, z, self.params)[0]
                - potential_energy(x, minus, z, self.params)[0]
            ) / (2.0 * epsilon)
            self.assertAlmostEqual(force_q[0, coordinate], -derivative, delta=2e-6)

        plus = z.copy()
        minus = z.copy()
        plus[0] += epsilon
        minus[0] -= epsilon
        derivative = (
            potential_energy(x, q, plus, self.params)[0]
            - potential_energy(x, q, minus, self.params)[0]
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(force_z[0], -derivative, delta=2e-6)

    def test_joint_integer_period_translation_is_invariant(self):
        x = np.array([[0.17, -0.28], [0.4, 0.2]])
        q = np.array([[-0.04, 0.09], [0.1, -0.3]])
        z = np.array([0.31, -0.2])
        shift = 2.0 * self.params.period

        before_energy = potential_energy(x, q, z, self.params)
        before_forces = conservative_forces(x, q, z, self.params)
        after_energy = potential_energy(x + shift, q + shift, z, self.params)
        after_forces = conservative_forces(x + shift, q + shift, z, self.params)

        np.testing.assert_allclose(after_energy, before_energy, atol=1e-12)
        for after, before in zip(after_forces, before_forces):
            np.testing.assert_allclose(after, before, atol=1e-12)

    def test_parameters_reject_nonphysical_domains(self):
        base = dict(
            temperature=1.0,
            period=1.0,
            base_barrier=2.0,
            elastic_stiffness=0.5,
            barrier_stiffness=1.0,
            barrier_coupling=0.5,
            gamma_x=1.0,
            gamma_q=1.0,
            gamma_z=1.0,
        )
        for key in (
            "temperature",
            "period",
            "base_barrier",
            "barrier_stiffness",
            "gamma_x",
            "gamma_q",
            "gamma_z",
        ):
            values = dict(base)
            values[key] = 0.0
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    TransientPeriodicParams(**values)
        for key in ("elastic_stiffness", "barrier_coupling"):
            values = dict(base)
            values[key] = -0.1
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    TransientPeriodicParams(**values)


class TransientIntegratorTests(unittest.TestCase):
    @staticmethod
    def params(**updates):
        values = dict(
            temperature=0.7,
            period=1.0,
            base_barrier=2.5,
            elastic_stiffness=0.6,
            barrier_stiffness=1.0,
            barrier_coupling=0.4,
            gamma_x=1.0,
            gamma_q=4.0,
            gamma_z=3.0,
        )
        values.update(updates)
        return TransientPeriodicParams(**values)

    def test_simulator_is_seed_deterministic(self):
        kwargs = dict(
            trajectory_count=12,
            dimension=2,
            dt=0.001,
            burnin_steps=30,
            production_steps=50,
            record_stride=5,
        )
        first = simulate_transient_periodic_langevin(
            self.params(), seed=17, **kwargs
        )
        second = simulate_transient_periodic_langevin(
            self.params(), seed=17, **kwargs
        )
        third = simulate_transient_periodic_langevin(
            self.params(), seed=18, **kwargs
        )

        for key in ("positions", "environment_positions", "barrier_coordinates"):
            self.assertTrue(np.array_equal(first[key], second[key]))
        self.assertFalse(np.array_equal(first["positions"], third["positions"]))
        self.assertEqual(first["all_finite"], 1.0)
        self.assertAlmostEqual(first["record_dt"], 0.005)

    def test_high_barrier_wrapped_variance_matches_local_equipartition(self):
        params = self.params(
            temperature=0.5,
            base_barrier=6.0,
            elastic_stiffness=0.0,
            barrier_coupling=0.0,
        )
        result = simulate_transient_periodic_langevin(
            params,
            trajectory_count=800,
            dimension=1,
            dt=0.0005,
            burnin_steps=3000,
            production_steps=2000,
            record_stride=20,
            seed=29,
        )
        positions = result["positions"]
        wrapped = positions - np.floor(positions / params.period + 0.5) * params.period
        observed = float(np.var(wrapped))
        expected = params.temperature / (
            2.0 * np.pi**2 * params.base_barrier / params.period**2
        )
        self.assertAlmostEqual(observed / expected, 1.0, delta=0.15)

    def test_elastic_environment_does_not_pin_common_translation(self):
        params = self.params(
            temperature=1.0,
            base_barrier=0.8,
            elastic_stiffness=0.5,
            barrier_coupling=0.0,
            gamma_q=2.0,
        )
        result = simulate_transient_periodic_langevin(
            params,
            trajectory_count=256,
            dimension=1,
            dt=0.002,
            burnin_steps=1000,
            production_steps=4000,
            record_stride=100,
            seed=31,
        )
        endpoint = result["positions"][-1, :, 0] - result["positions"][0, :, 0]
        self.assertGreater(float(np.mean(endpoint**2)), 0.1)

    def test_elastic_ablation_keeps_uncoupled_barrier_noise_paired(self):
        kwargs = dict(
            trajectory_count=16,
            dimension=2,
            dt=0.001,
            burnin_steps=20,
            production_steps=40,
            record_stride=5,
            seed=37,
        )
        without_elastic = simulate_transient_periodic_langevin(
            self.params(elastic_stiffness=0.0, barrier_coupling=0.0),
            **kwargs,
        )
        with_elastic = simulate_transient_periodic_langevin(
            self.params(elastic_stiffness=0.6, barrier_coupling=0.0),
            **kwargs,
        )

        self.assertTrue(
            np.array_equal(
                without_elastic["barrier_coordinates"],
                with_elastic["barrier_coordinates"],
            )
        )

    def test_integrator_rejects_reference_curvature_instability(self):
        with self.assertRaisesRegex(ValueError, "stability"):
            simulate_transient_periodic_langevin(
                self.params(base_barrier=3.0),
                trajectory_count=4,
                dimension=1,
                dt=0.01,
                burnin_steps=1,
                production_steps=2,
                record_stride=1,
                seed=1,
            )


class TransientEventTests(unittest.TestCase):
    def test_nonrecrossing_dwell_rejects_transient_index_change(self):
        positions = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0])[:, None, None]

        events = stable_cage_events(
            positions,
            period=1.0,
            dwell_frames=3,
            frame_dt=0.5,
        )

        self.assertEqual(len(events["frame"]), 1)
        self.assertEqual(int(events["frame"][0]), 3)
        self.assertAlmostEqual(float(events["time"][0]), 1.5)
        self.assertEqual(int(events["signed_cage_step"][0]), 1)
        np.testing.assert_allclose(events["vector_step"][0], [1.0])

    def test_displacement_observables_match_direct_moments(self):
        positions = np.zeros((3, 2, 3), dtype=float)
        positions[1, 0] = [1.0, 0.0, 0.0]
        positions[1, 1] = [0.0, 2.0, 0.0]
        positions[2] = 2.0 * positions[1]

        rows = displacement_observables(
            positions,
            lag_frames=[1],
            wave_numbers=[2.0],
        )

        displacements = positions[1:] - positions[:-1]
        squared = np.sum(displacements**2, axis=2)
        msd = float(np.mean(squared))
        fourth = float(np.mean(squared**2))
        ngp = 3.0 * fourth / (5.0 * msd**2) - 1.0
        self.assertAlmostEqual(rows[0]["msd"], msd)
        self.assertAlmostEqual(rows[0]["ngp"], ngp)
        self.assertAlmostEqual(rows[0]["fs_k2"], np.mean(np.cos(2.0 * displacements)))

    def test_empty_event_clock_has_explicit_unsupported_waiting_fields(self):
        positions = np.zeros((11, 2, 3), dtype=float)
        events = stable_cage_events(
            positions,
            period=1.0,
            dwell_frames=2,
            frame_dt=0.1,
        )

        result = event_clock_statistics(
            events,
            trajectory_count=2,
            dimension=3,
            duration=1.0,
            count_window=0.5,
        )

        self.assertEqual(result["event_count"], 0.0)
        self.assertEqual(result["mean_window_count"], 0.0)
        self.assertEqual(result["count_fano_factor"], 0.0)
        self.assertEqual(result["persistence_supported"], 0.0)
        self.assertEqual(result["exchange_supported"], 0.0)
        self.assertTrue(np.isnan(result["mean_persistence_time"]))
        self.assertTrue(np.isnan(result["mean_exchange_time"]))


class TransientAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_script(
            "analyze_transient_periodic_langevin.py",
            "analyze_transient_periodic_langevin",
        )

    def test_quick_ablation_is_reproducible_and_separates_mechanisms(self):
        first_rows, first_gate = self.analysis.run_ablation(seed=20260718, quick=True)
        second_rows, second_gate = self.analysis.run_ablation(seed=20260718, quick=True)

        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first_gate, second_gate)
        self.assertEqual(
            [row["model"] for row in first_rows],
            ["static_periodic", "rate_only", "elastic_only", "full_transient"],
        )
        for row in first_rows:
            self.assertEqual(float(row["all_finite"]), 1.0)
            self.assertEqual(float(row["trajectory_continuity_pass"]), 1.0)
            self.assertGreater(float(row["event_count"]), 0.0)
        self.assertEqual(float(first_gate["rate_disorder_count_fano_increase"]), 1.0)
        self.assertEqual(
            float(first_gate["elastic_memory_more_negative_step_correlation"]),
            1.0,
        )
        self.assertEqual(float(first_gate["full_model_joint_signature_pass"]), 1.0)
        self.assertEqual(float(first_gate["synthetic_capability_only"]), 1.0)
        for key in self.analysis.CLAIM_FLAGS:
            self.assertEqual(float(first_gate[key]), 0.0)

    def test_ablation_gate_is_a_deterministic_function_of_rows(self):
        rows, gate = self.analysis.run_ablation(seed=20260718, quick=True)

        rebuilt = self.analysis.classify_ablation(rows)

        self.assertEqual(rebuilt, gate)
        self.assertEqual(
            rebuilt["stochastic_artifact_validation"],
            "stored_rows_gate_and_figure_self_consistency",
        )
        self.assertEqual(
            float(rebuilt["cross_platform_exact_trajectory_reproduction_required"]),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
