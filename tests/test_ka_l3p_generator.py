import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class L3pGeneratorTests(unittest.TestCase):
    @staticmethod
    def cage_inputs():
        return {
            "positions": np.array(
                [
                    [0.10, -0.20, 0.30],
                    [1.05, -0.10, 0.25],
                    [-0.65, 0.55, 0.35],
                    [0.05, -1.15, 0.60],
                ],
                dtype=float,
            ),
            "particle_types": np.array([0, 0, 1, 0], dtype=int),
            "box_lengths": np.array([20.0, 20.0, 20.0]),
            "target_indices": np.array([0, 1], dtype=int),
        }

    @staticmethod
    def scaled_coordinate_probes(particle_count):
        dimension = 3 * particle_count
        probes = np.eye(dimension) * np.sqrt(dimension)
        return probes.reshape(dimension, particle_count, 3)

    def exhaustive_cage_laplacian(self, inputs, step):
        from ka_smooth_cage import smooth_force_support_cage_batch

        positions = np.asarray(inputs["positions"])
        targets = np.asarray(inputs["target_indices"])
        output = np.zeros((len(targets), 3))
        common = {
            "velocities": np.zeros_like(positions),
            "particle_types": inputs["particle_types"],
            "box_lengths": inputs["box_lengths"],
            "target_indices": targets,
            "compute_gram": False,
            "return_jacobian": True,
        }
        for particle in range(len(positions)):
            for component in range(3):
                direction = np.zeros_like(positions)
                direction[particle, component] = 1.0
                plus = smooth_force_support_cage_batch(
                    positions + step * direction,
                    **common,
                )["jacobian"]
                minus = smooth_force_support_cage_batch(
                    positions - step * direction,
                    **common,
                )["jacobian"]
                derivative = (np.asarray(plus) - np.asarray(minus)) / (2.0 * step)
                output += derivative[:, particle, :, component]
        return output

    def test_cage_laplacian_prefix_matches_exhaustive_basis(self):
        from ka_l3p_generator import smooth_cage_laplacian_prefixes

        inputs = self.cage_inputs()
        probes = self.scaled_coordinate_probes(len(inputs["positions"]))
        step = 5e-6
        result = smooth_cage_laplacian_prefixes(
            **inputs,
            trace_probes=probes,
            directional_step=step,
            prefix_counts=(len(probes),),
        )
        expected = self.exhaustive_cage_laplacian(inputs, step)

        np.testing.assert_allclose(
            result["laplacian_prefixes"][-1],
            expected,
            rtol=2e-7,
            atol=2e-8,
        )
        translated = smooth_cage_laplacian_prefixes(
            **{
                **inputs,
                "positions": inputs["positions"] + np.array([2.3, -1.7, 0.8]),
            },
            trace_probes=probes,
            directional_step=step,
            prefix_counts=(len(probes),),
        )
        np.testing.assert_allclose(
            translated["laplacian_prefixes"],
            result["laplacian_prefixes"],
            rtol=2e-8,
            atol=2e-9,
        )
        self.assertEqual(result["thermodynamic_claim_allowed"], 0.0)

    def test_two_particle_linear_cage_has_zero_laplacian(self):
        from ka_l3p_generator import smooth_cage_laplacian_prefixes

        positions = np.array([[0.0, 0.0, 0.0], [0.9, -0.2, 0.1]])
        probes = self.scaled_coordinate_probes(len(positions))
        result = smooth_cage_laplacian_prefixes(
            positions,
            particle_types=np.array([0, 0]),
            box_lengths=np.full(3, 20.0),
            target_indices=np.array([0]),
            trace_probes=probes,
            directional_step=1e-5,
            prefix_counts=(len(probes),),
        )

        np.testing.assert_allclose(
            result["laplacian_prefixes"],
            np.zeros((1, 1, 3)),
            atol=2e-10,
        )

    def test_l3p_component_assembly_matches_exact_identity(self):
        from ka_l3p_generator import assemble_l3p_generator

        rng = np.random.default_rng(20260731)
        target_count = 2
        particle_count = 4
        prefix_count = 3
        position = rng.normal(size=(target_count, 3))
        jacobian = rng.normal(size=(target_count, 3, particle_count, 3))
        acceleration = rng.normal(size=(particle_count, 3))
        laplacian = rng.normal(size=(prefix_count, target_count, 3))
        laplacian_gradient = rng.normal(size=laplacian.shape)
        friction = 0.7
        temperature = 0.58
        result = assemble_l3p_generator(
            position_transport_term=position,
            l2p_velocity_jacobian=jacobian,
            acceleration=acceleration,
            laplacian_prefixes=laplacian,
            laplacian_velocity_derivative_prefixes=laplacian_gradient,
            friction=friction,
            temperature=temperature,
        )

        acceleration_term = np.einsum("tanb,nb->ta", jacobian, acceleration)
        thermal_gradient = 8.0 * friction * temperature * laplacian_gradient
        thermal_friction = -6.0 * friction**2 * temperature * laplacian
        expected = (
            position[None]
            + acceleration_term[None]
            + thermal_gradient
            + thermal_friction
        )
        np.testing.assert_allclose(result["l3p_prefixes"], expected)
        np.testing.assert_allclose(
            result["acceleration_response_term"], acceleration_term
        )
        np.testing.assert_allclose(
            result["thermal_gradient_prefixes"], thermal_gradient
        )
        np.testing.assert_allclose(
            result["thermal_friction_prefixes"], thermal_friction
        )
        self.assertEqual(result["thermodynamic_claim_allowed"], 0.0)

        zero_temperature = assemble_l3p_generator(
            position_transport_term=position,
            l2p_velocity_jacobian=jacobian,
            acceleration=acceleration,
            laplacian_prefixes=laplacian,
            laplacian_velocity_derivative_prefixes=laplacian_gradient,
            friction=friction,
            temperature=0.0,
        )
        np.testing.assert_allclose(
            zero_temperature["l3p_prefixes"],
            np.broadcast_to(
                position[None] + acceleration_term[None],
                (prefix_count, target_count, 3),
            ),
        )
        np.testing.assert_array_equal(
            zero_temperature["thermal_gradient_prefixes"],
            np.zeros_like(laplacian),
        )
        np.testing.assert_array_equal(
            zero_temperature["thermal_friction_prefixes"],
            np.zeros_like(laplacian),
        )

    def test_harmonic_l3p_identity(self):
        from ka_l3p_generator import assemble_l3p_generator

        rng = np.random.default_rng(20260801)
        dimension = 6
        target_count = 2
        raw = rng.normal(size=(dimension, dimension))
        stiffness = raw.T @ raw + 0.4 * np.eye(dimension)
        projection = rng.normal(size=(3 * target_count, dimension))
        position = rng.normal(size=dimension)
        velocity = rng.normal(size=dimension)
        friction = 0.7
        force = -stiffness @ position
        acceleration = force - friction * velocity
        velocity_jacobian = projection @ (
            friction**2 * np.eye(dimension) - stiffness
        )
        position_transport = friction * projection @ stiffness @ velocity
        result = assemble_l3p_generator(
            position_transport_term=position_transport.reshape(target_count, 3),
            l2p_velocity_jacobian=velocity_jacobian.reshape(
                target_count, 3, dimension // 3, 3
            ),
            acceleration=acceleration.reshape(dimension // 3, 3),
            laplacian_prefixes=np.zeros((1, target_count, 3)),
            laplacian_velocity_derivative_prefixes=np.zeros(
                (1, target_count, 3)
            ),
            friction=friction,
            temperature=0.58,
        )
        expected = projection @ (
            (stiffness @ stiffness - friction**2 * stiffness) @ position
            + (2.0 * friction * stiffness - friction**3 * np.eye(dimension))
            @ velocity
        )

        np.testing.assert_allclose(
            result["l3p"],
            expected.reshape(target_count, 3),
            rtol=2e-10,
            atol=2e-10,
        )

    def test_l3p_matches_direct_zero_temperature_phase_generator(self):
        from ka_l3p_generator import smooth_cage_l3p_generator_batch
        from ka_local_cage import ka_lj_sparse_force_generator_observables
        from ka_smooth_cage import smooth_cage_second_generator_batch

        inputs = self.cage_inputs()
        positions = inputs["positions"]
        velocities = np.array(
            [
                [0.20, -0.10, 0.30],
                [-0.15, 0.25, -0.05],
                [0.10, 0.05, -0.20],
                [-0.05, -0.30, 0.15],
            ]
        )
        friction = 0.7
        all_particles = np.arange(len(positions))
        common = {
            "particle_types": inputs["particle_types"],
            "box_lengths": inputs["box_lengths"],
            "target_indices": inputs["target_indices"],
            "friction": friction,
            "temperature": 0.0,
            "directional_step": 1e-5,
            "phase_space_step": 3e-6,
            "trace_probes": None,
        }

        def c0(configuration, current_velocity):
            force_state = ka_lj_sparse_force_generator_observables(
                configuration,
                velocities=current_velocity,
                particle_types=inputs["particle_types"],
                box_lengths=inputs["box_lengths"],
                target_indices=all_particles,
                potential_protocol="ka_lj_c3_switch",
            )
            return np.asarray(
                smooth_cage_second_generator_batch(
                    configuration,
                    velocities=current_velocity,
                    forces=np.asarray(force_state["force"]),
                    force_generator=np.asarray(force_state["force_generator"]),
                    **common,
                )["second_relative_generator"]
            )

        base_force = ka_lj_sparse_force_generator_observables(
            positions,
            velocities=velocities,
            particle_types=inputs["particle_types"],
            box_lengths=inputs["box_lengths"],
            target_indices=all_particles,
            potential_protocol="ka_lj_c3_switch",
        )["force"]
        acceleration = np.asarray(base_force) - friction * velocities
        direct_step = 8e-7
        direct = (
            c0(
                positions + direct_step * velocities,
                velocities + direct_step * acceleration,
            )
            - c0(
                positions - direct_step * velocities,
                velocities - direct_step * acceleration,
            )
        ) / (2.0 * direct_step)
        probes = self.scaled_coordinate_probes(len(positions))
        result = smooth_cage_l3p_generator_batch(
            positions,
            velocities=velocities,
            particle_types=inputs["particle_types"],
            box_lengths=inputs["box_lengths"],
            target_indices=inputs["target_indices"],
            friction=friction,
            temperature=0.0,
            trace_probes=probes,
            prefix_counts=(len(probes),),
            position_step=1e-5,
            cage_hessian_step=1e-5,
            jacobian_step=1e-5,
            l2p_directional_step=1e-5,
            l2p_phase_space_step=3e-6,
            potential_protocol="ka_lj_c3_switch",
        )

        np.testing.assert_allclose(
            result["l3p"],
            direct,
            rtol=2e-3,
            atol=2e-3,
        )

    def test_l3p_matches_exhaustive_finite_temperature_generator(self):
        from ka_l3p_generator import smooth_cage_l3p_generator_batch
        from ka_local_cage import ka_lj_sparse_force_generator_observables
        from ka_smooth_cage import (
            smooth_cage_l2p_velocity_directional_derivative_batch,
            smooth_cage_second_generator_batch,
        )

        inputs = self.cage_inputs()
        positions = inputs["positions"]
        velocities = np.arange(12, dtype=float).reshape(4, 3) / 17.0 - 0.28
        friction = 0.7
        temperature = 0.58
        probes = self.scaled_coordinate_probes(len(positions))
        all_particles = np.arange(len(positions))

        def force_state(configuration, current_velocity):
            return ka_lj_sparse_force_generator_observables(
                configuration,
                velocities=current_velocity,
                particle_types=inputs["particle_types"],
                box_lengths=inputs["box_lengths"],
                target_indices=all_particles,
                potential_protocol="ka_lj_c3_switch",
            )

        def c(configuration, current_velocity):
            microscopic = force_state(configuration, current_velocity)
            return np.asarray(
                smooth_cage_second_generator_batch(
                    configuration,
                    velocities=current_velocity,
                    forces=np.asarray(microscopic["force"]),
                    force_generator=np.asarray(microscopic["force_generator"]),
                    particle_types=inputs["particle_types"],
                    box_lengths=inputs["box_lengths"],
                    target_indices=inputs["target_indices"],
                    friction=friction,
                    temperature=temperature,
                    directional_step=1e-5,
                    phase_space_step=3e-6,
                    trace_probes=probes,
                )["second_relative_generator"]
            )

        base_force_state = force_state(positions, velocities)
        forces = np.asarray(base_force_state["force"])
        acceleration = forces - friction * velocities
        drift_step = 1e-5
        phase_drift = (
            c(
                positions + drift_step * velocities,
                velocities + drift_step * acceleration,
            )
            - c(
                positions - drift_step * velocities,
                velocities - drift_step * acceleration,
            )
        ) / (2.0 * drift_step)
        velocity_step = 2e-3
        base_c = c(positions, velocities)
        laplacian_actions = []
        for probe in probes:
            laplacian_actions.append(
                (
                    c(positions, velocities + velocity_step * probe)
                    + c(positions, velocities - velocity_step * probe)
                    - 2.0 * base_c
                )
                / velocity_step**2
            )
        direct = phase_drift + friction * temperature * np.mean(
            laplacian_actions,
            axis=0,
        )
        result = smooth_cage_l3p_generator_batch(
            positions,
            velocities=velocities,
            particle_types=inputs["particle_types"],
            box_lengths=inputs["box_lengths"],
            target_indices=inputs["target_indices"],
            friction=friction,
            temperature=temperature,
            trace_probes=probes,
            prefix_counts=(len(probes),),
            position_step=1e-5,
            cage_hessian_step=1e-5,
            jacobian_step=1e-5,
            l2p_directional_step=1e-5,
            l2p_phase_space_step=3e-6,
            potential_protocol="ka_lj_c3_switch",
        )

        np.testing.assert_allclose(
            result["l3p"],
            direct,
            rtol=2e-4,
            atol=2e-6,
        )
        directional_force = force_state(positions, acceleration)
        acceleration_response = smooth_cage_l2p_velocity_directional_derivative_batch(
            positions,
            velocities=velocities,
            velocity_direction=acceleration,
            forces=forces,
            force_generator=np.asarray(base_force_state["force_generator"]),
            force_generator_direction=np.asarray(
                directional_force["force_generator"]
            ),
            particle_types=inputs["particle_types"],
            box_lengths=inputs["box_lengths"],
            target_indices=inputs["target_indices"],
            friction=friction,
            directional_step=1e-5,
            phase_space_step=3e-6,
            velocity_step=2e-5,
        )["l2p_velocity_directional_derivative"]
        np.testing.assert_allclose(
            result["acceleration_response_term"],
            acceleration_response,
            rtol=2e-3,
            atol=2e-3,
        )


if __name__ == "__main__":
    unittest.main()
