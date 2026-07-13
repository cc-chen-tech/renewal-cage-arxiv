import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SmoothCageTests(unittest.TestCase):
    @staticmethod
    def microscopic_configuration():
        return {
            "positions": np.array(
                [
                    [0.10, -0.20, 0.30],
                    [1.05, -0.10, 0.25],
                    [-0.65, 0.55, 0.35],
                    [0.05, -1.15, 0.60],
                ]
            ),
            "particle_types": np.array([0, 0, 1, 0]),
            "box_lengths": np.array([20.0, 20.0, 20.0]),
            "target_index": 0,
        }

    def test_wendland_weight_and_derivative_are_compact_and_smooth(self):
        from ka_smooth_cage import wendland_c4_weight

        scaled_distance = np.array([0.0, 0.5, 1.0, 1.1])
        weight, derivative = wendland_c4_weight(scaled_distance)

        self.assertAlmostEqual(weight[0], 3.0)
        self.assertGreater(weight[1], 0.0)
        np.testing.assert_array_equal(weight[2:], 0.0)
        np.testing.assert_array_equal(derivative[2:], 0.0)

    def test_smooth_cage_is_translation_invariant_and_decomposes_position(self):
        from ka_smooth_cage import smooth_force_support_cage

        inputs = self.microscopic_configuration()
        result = smooth_force_support_cage(**inputs)
        translation = np.array([2.3, -1.7, 0.8])
        translated = smooth_force_support_cage(
            **{**inputs, "positions": inputs["positions"] + translation}
        )

        np.testing.assert_allclose(
            result["relative_position"],
            inputs["positions"][inputs["target_index"]] - result["cage_position"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            translated["relative_position"], result["relative_position"], atol=1e-12
        )
        np.testing.assert_allclose(
            translated["cage_position"], result["cage_position"] + translation, atol=1e-12
        )
        np.testing.assert_allclose(
            np.sum(result["jacobian"], axis=0), np.zeros((3, 3)), atol=1e-12
        )

    def test_smooth_cage_analytic_jacobian_matches_centered_difference(self):
        from ka_smooth_cage import smooth_force_support_cage

        inputs = self.microscopic_configuration()
        result = smooth_force_support_cage(**inputs)
        step = 1e-6
        numerical = np.empty_like(result["jacobian"])
        for particle in range(len(inputs["positions"])):
            for component in range(3):
                plus_positions = inputs["positions"].copy()
                minus_positions = inputs["positions"].copy()
                plus_positions[particle, component] += step
                minus_positions[particle, component] -= step
                plus = smooth_force_support_cage(
                    **{**inputs, "positions": plus_positions}
                )["relative_position"]
                minus = smooth_force_support_cage(
                    **{**inputs, "positions": minus_positions}
                )["relative_position"]
                numerical[particle, :, component] = (plus - minus) / (2.0 * step)

        relative_l2 = np.linalg.norm(result["jacobian"] - numerical) / np.linalg.norm(
            numerical
        )
        self.assertLess(relative_l2, 1e-6)

    def test_projected_observables_obey_kinematics_and_fdt(self):
        from ka_smooth_cage import smooth_cage_projected_observables

        inputs = self.microscopic_configuration()
        velocities = np.array(
            [
                [0.20, -0.10, 0.30],
                [-0.15, 0.25, -0.05],
                [0.10, 0.05, -0.20],
                [-0.05, -0.30, 0.15],
            ]
        )
        friction = 0.7
        temperature = 0.58
        result = smooth_cage_projected_observables(
            **inputs,
            velocities=velocities,
            friction=friction,
            temperature=temperature,
            directional_step=1e-5,
            potential_protocol="ka_lj_c3_switch",
        )

        expected_velocity = np.einsum("nab,nb->a", result["jacobian"], velocities)
        np.testing.assert_allclose(result["relative_velocity"], expected_velocity, atol=1e-12)
        np.testing.assert_allclose(
            result["noise_covariance_rate"],
            2.0 * friction * temperature * result["jacobian_gram"],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result["effective_mass"] @ result["jacobian_gram"],
            np.eye(3),
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertGreater(result["jacobian_gram_minimum_eigenvalue"], 0.0)
        self.assertEqual(result["thermodynamic_claim_allowed"], 0.0)

    def test_projected_observables_accept_exact_microscopic_forces(self):
        from ka_smooth_cage import smooth_cage_projected_observables

        inputs = self.microscopic_configuration()
        velocities = np.arange(12, dtype=float).reshape(4, 3) / 20.0
        forces = np.arange(12, dtype=float).reshape(4, 3) / 10.0
        result = smooth_cage_projected_observables(
            **inputs,
            velocities=velocities,
            forces=forces,
            friction=1.0,
            temperature=0.58,
            directional_step=1e-5,
            potential_protocol="ka_lj_cut",
        )

        expected = np.einsum("nab,nb->a", result["jacobian"], forces)
        np.testing.assert_allclose(result["force_drift"], expected, atol=1e-12)

    def test_smooth_cage_features_are_rotation_invariant(self):
        from ka_smooth_cage import smooth_cage_invariant_features

        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        observable = {
            "relative_position": np.array([0.2, -0.3, 0.4]),
            "relative_velocity": np.array([-0.5, 0.1, 0.3]),
            "projected_drift": np.array([0.7, -0.2, -0.4]),
            "jacobian_gram": np.array(
                [
                    [0.9, 0.1, 0.0],
                    [0.1, 1.1, 0.2],
                    [0.0, 0.2, 0.8],
                ]
            ),
        }
        rotated = {
            "relative_position": rotation @ observable["relative_position"],
            "relative_velocity": rotation @ observable["relative_velocity"],
            "projected_drift": rotation @ observable["projected_drift"],
            "jacobian_gram": rotation @ observable["jacobian_gram"] @ rotation.T,
        }

        feature = smooth_cage_invariant_features(observable)
        rotated_feature = smooth_cage_invariant_features(rotated)
        self.assertEqual(feature["geometry"].shape, (4,))
        self.assertEqual(feature["kinematic"].shape, (6,))
        self.assertEqual(feature["full"].shape, (9,))
        for key in ("geometry", "kinematic", "full"):
            np.testing.assert_allclose(feature[key], rotated_feature[key], atol=1e-12)

    def test_grouped_exponential_escape_recovers_transferable_microscopic_rate(self):
        from ka_smooth_cage import grouped_exponential_escape_diagnostic

        rng = np.random.default_rng(20260714)
        groups = np.repeat(np.arange(5), 800)
        feature = rng.normal(size=(len(groups), 2))
        rate = np.exp(-3.0 + 0.8 * feature[:, 0] - 0.5 * feature[:, 1])
        raw_first_passage = rng.exponential(1.0 / rate)
        horizon = 20.0
        escaped = raw_first_passage <= horizon
        first_passage = np.minimum(raw_first_passage, horizon)
        result = grouped_exponential_escape_diagnostic(
            feature,
            first_passage,
            escaped,
            groups,
            horizon=horizon,
            survival_times=np.array([1, 2, 4, 8, 12, 16, 20], dtype=float),
            l2_regularization=1.0,
        )

        self.assertGreater(result["mean_heldout_brier_skill"], 0.05)
        self.assertGreater(
            result["mean_heldout_log_likelihood_gain_per_observation"], 0.01
        )
        self.assertLess(result["maximum_heldout_survival_calibration_error"], 0.05)
        self.assertGreater(np.corrcoef(result["out_of_group_rate"], rate)[0, 1], 0.95)

    def test_grouped_exponential_escape_rejects_invalid_censoring_and_groups(self):
        from ka_smooth_cage import grouped_exponential_escape_diagnostic

        feature = np.ones((4, 1))
        escaped = np.ones(4, dtype=bool)
        with self.assertRaisesRegex(ValueError, "at least two parent groups"):
            grouped_exponential_escape_diagnostic(
                feature,
                np.ones(4),
                escaped,
                np.zeros(4),
                horizon=2.0,
                survival_times=np.array([1.0, 2.0]),
            )
        with self.assertRaisesRegex(ValueError, "first_passage"):
            grouped_exponential_escape_diagnostic(
                feature,
                np.full(4, 21.0),
                escaped,
                np.arange(4) % 2,
                horizon=20.0,
                survival_times=np.array([1.0, 20.0]),
            )

    def test_projected_drift_matches_phase_space_directional_derivative(self):
        from ka_local_cage import ka_lj_force_and_isotropic_curvature
        from ka_smooth_cage import (
            smooth_cage_projected_observables,
            smooth_force_support_cage,
        )

        inputs = self.microscopic_configuration()
        positions = inputs["positions"]
        velocities = np.array(
            [
                [0.20, -0.10, 0.30],
                [-0.15, 0.25, -0.05],
                [0.10, 0.05, -0.20],
                [-0.05, -0.30, 0.15],
            ]
        )
        force, _ = ka_lj_force_and_isotropic_curvature(
            positions,
            particle_types=inputs["particle_types"],
            box_lengths=inputs["box_lengths"],
            potential_protocol="ka_lj_c3_switch",
        )
        result = smooth_cage_projected_observables(
            **inputs,
            velocities=velocities,
            friction=0.0,
            temperature=0.0,
            directional_step=1e-5,
            potential_protocol="ka_lj_c3_switch",
        )

        errors = []
        for step in (8e-4, 4e-4, 2e-4):
            plus = smooth_force_support_cage(
                **{**inputs, "positions": positions + step * velocities}
            )
            minus = smooth_force_support_cage(
                **{**inputs, "positions": positions - step * velocities}
            )
            plus_velocity = np.einsum(
                "nab,nb->a", plus["jacobian"], velocities + step * force
            )
            minus_velocity = np.einsum(
                "nab,nb->a", minus["jacobian"], velocities - step * force
            )
            numerical = (plus_velocity - minus_velocity) / (2.0 * step)
            errors.append(np.linalg.norm(numerical - result["projected_drift"]))

        self.assertLess(errors[1], 0.35 * errors[0])
        self.assertLess(errors[2], 0.35 * errors[1])
        self.assertLess(
            errors[-1] / np.linalg.norm(result["projected_drift"]),
            2e-4,
        )

    def test_extract_smooth_cage_path_reads_full_lammps_state(self):
        from ka_smooth_cage import extract_smooth_cage_path

        inputs = self.microscopic_configuration()
        velocities = np.array(
            [
                [0.20, -0.10, 0.30],
                [-0.15, 0.25, -0.05],
                [0.10, 0.05, -0.20],
                [-0.05, -0.30, 0.15],
            ]
        )
        rows = []
        for timestep, shift in ((0, 0.0), (5, 0.001)):
            rows.extend(
                [
                    "ITEM: TIMESTEP",
                    str(timestep),
                    "ITEM: NUMBER OF ATOMS",
                    "4",
                    "ITEM: BOX BOUNDS pp pp pp",
                    "-10 10",
                    "-10 10",
                    "-10 10",
                    "ITEM: ATOMS id type x y z ix iy iz vx vy vz",
                ]
            )
            for index, (position, velocity) in enumerate(
                zip(inputs["positions"] + shift * velocities, velocities), start=1
            ):
                particle_type = inputs["particle_types"][index - 1] + 1
                rows.append(
                    f"{index} {particle_type} {position[0]} {position[1]} {position[2]} "
                    f"0 0 0 {velocity[0]} {velocity[1]} {velocity[2]}"
                )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.lammpstrj"
            path.write_text("\n".join(rows) + "\n")
            result = extract_smooth_cage_path(
                path,
                target_id=1,
                friction=1.0,
                temperature=0.58,
                integration_time_step=0.001,
                directional_step=1e-5,
                potential_protocol="ka_lj_c3_switch",
            )

        self.assertEqual(result["relative_position"].shape, (2, 3))
        self.assertEqual(result["relative_velocity"].shape, (2, 3))
        self.assertEqual(result["projected_drift"].shape, (2, 3))
        self.assertEqual(result["jacobian"].shape, (2, 4, 3, 3))
        self.assertEqual(result["noise_covariance_rate"].shape, (2, 3, 3))
        np.testing.assert_allclose(result["time"], np.array([0.0, 0.005]))

    def test_matched_smooth_cage_tangent_has_exact_delta_j_covariance(self):
        from ka_smooth_cage import matched_smooth_cage_tangent

        epsilon = 0.01
        friction = 0.8
        temperature = 0.58
        time = np.arange(6, dtype=float) * 0.1
        delta_velocity = time[:, None] * np.array([[1.0, -0.5, 0.25]])
        delta_drift = np.broadcast_to(np.array([1.0, -0.5, 0.25]), (6, 3))
        base_jacobian = np.zeros((6, 4, 3, 3))
        base_jacobian[:, 0] = np.eye(3)
        delta_jacobian = np.zeros_like(base_jacobian)
        delta_jacobian[:, 0] = np.diag([0.2, 0.3, 0.4])
        delta_jacobian[:, 1] = -delta_jacobian[:, 0]
        base_position = np.zeros((6, 3))
        base_velocity = np.zeros((6, 3))
        base_drift = np.zeros((6, 3))

        def path(sign):
            return {
                "time": time,
                "relative_position": base_position,
                "relative_velocity": base_velocity + sign * epsilon * delta_velocity,
                "projected_drift": base_drift + sign * epsilon * delta_drift,
                "jacobian": base_jacobian + sign * epsilon * delta_jacobian,
                "frame_time": 0.1,
                "friction": friction,
                "temperature": temperature,
            }

        result = matched_smooth_cage_tangent(path(1.0), path(-1.0), epsilon=epsilon)
        expected_rate = 2.0 * friction * temperature * np.einsum(
            "tnab,tncb->tac", delta_jacobian, delta_jacobian
        )
        np.testing.assert_allclose(result["relative_velocity_response"], delta_velocity)
        np.testing.assert_allclose(result["projected_drift_response"], delta_drift)
        np.testing.assert_allclose(result["tangent_noise_covariance_rate"], expected_rate)

    def test_integrated_tangent_covariance_uses_nonoverlapping_trapezoids(self):
        from ka_smooth_cage import integrated_smooth_cage_tangent_covariance

        time = np.arange(5, dtype=float) * 0.1
        drift = np.broadcast_to(np.array([1.0, -0.5, 0.25]), (5, 3))
        velocity = time[:, None] * drift[0]
        rate = np.broadcast_to(np.diag([2.0, 3.0, 4.0]), (5, 3, 3))
        result = integrated_smooth_cage_tangent_covariance(
            {
                "time": time,
                "relative_velocity_response": velocity,
                "projected_drift_response": drift,
                "tangent_noise_covariance_rate": rate,
                "frame_time": 0.1,
            },
            stride=2,
        )

        np.testing.assert_allclose(result["residual"], np.zeros((2, 3)), atol=1e-14)
        np.testing.assert_allclose(
            result["integrated_covariance"],
            np.broadcast_to(0.2 * rate[0], (2, 3, 3)),
        )

    def test_smooth_cage_runner_has_c3_low_disk_contract(self):
        script = ROOT / "scripts" / "run_ka_smooth_cage_response.py"
        source = script.read_text()

        for option in (
            "--parent-restart",
            "--lammps",
            "--output-directory",
            "--members",
            "--epsilons",
            "--run-steps",
            "--dump-interval",
            "--directional-step",
        ):
            self.assertIn(option, source)
        self.assertIn('potential_protocol="ka_lj_c3_switch"', source)
        self.assertIn("temporary.replace(output_path)", source)
        self.assertGreater(
            source.index("trajectory_path.unlink()"),
            source.index("temporary.replace(output_path)"),
        )

    def test_smooth_cage_analysis_reports_preregistered_gates(self):
        script = ROOT / "scripts" / "analyze_ka_smooth_cage_tangent.py"
        source = script.read_text()

        self.assertIn("tangent_noise_covariance_diagnostic", source)
        self.assertIn("cross_epsilon_covariance_correlation", source)
        for gate in (
            "integrity_gate_pass",
            "cross_epsilon_gate_pass",
            "trace_variance_gate_pass",
            "mahalanobis_gate_pass",
            "whitened_lag1_gate_pass",
            "microscopic_smooth_cage_tangent_gate_pass",
            "without_delta_j_null_rejected",
            "thermodynamic_claim_allowed",
        ):
            self.assertIn(gate, source)
        self.assertIn("PRIMARY_STRIDE = 5", source)


if __name__ == "__main__":
    unittest.main()
