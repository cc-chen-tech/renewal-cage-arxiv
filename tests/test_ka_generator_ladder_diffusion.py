import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class GeneratorLadderDiffusionTests(unittest.TestCase):
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
        }

    def test_microscopic_frame_builds_a0_through_a3_and_all_noise_blocks(self):
        from ka_generator_ladder_diffusion import (
            smooth_cage_generator_ladder_velocity_jacobians_batch,
        )
        from ka_smooth_cage import smooth_force_support_cage_batch

        inputs = self.microscopic_configuration()
        positions = inputs["positions"]
        velocities = np.arange(12, dtype=float).reshape(4, 3) / 17.0 - 0.2
        targets = np.array([0, 2])
        friction = 0.7
        temperature = 0.58
        step = 2e-5
        a3 = np.random.default_rng(39).normal(size=(2, 3, 4, 3))
        result = smooth_cage_generator_ladder_velocity_jacobians_batch(
            positions,
            velocities=velocities,
            particle_types=inputs["particle_types"],
            box_lengths=inputs["box_lengths"],
            target_indices=targets,
            l2p_velocity_jacobian=a3,
            friction=friction,
            temperature=temperature,
            position_step=step,
        )

        common = {
            "velocities": velocities,
            "particle_types": inputs["particle_types"],
            "box_lengths": inputs["box_lengths"],
            "target_indices": targets,
            "compute_gram": False,
            "return_jacobian": True,
        }
        base = smooth_force_support_cage_batch(positions, **common)["jacobian"]
        plus = smooth_force_support_cage_batch(
            positions + step * velocities,
            **common,
        )["jacobian"]
        minus = smooth_force_support_cage_batch(
            positions - step * velocities,
            **common,
        )["jacobian"]
        derivative = (plus - minus) / (2.0 * step)
        matrices = np.asarray(result["velocity_jacobians"])
        np.testing.assert_array_equal(matrices[0], 0.0)
        np.testing.assert_allclose(matrices[1], np.transpose(base, (0, 2, 1, 3)))
        np.testing.assert_allclose(
            matrices[2],
            np.transpose(2.0 * derivative - friction * base, (0, 2, 1, 3)),
        )
        np.testing.assert_array_equal(matrices[3], a3)
        expected_blocks = 2.0 * friction * temperature * np.einsum(
            "rtanb,stcnb->rstac",
            matrices,
            matrices,
        )
        np.testing.assert_allclose(result["diffusion_blocks"], expected_blocks)
        self.assertEqual(result["full_microscopic_velocity_jacobians_retained"], 1.0)
        self.assertEqual(result["projected_state_autonomous"], 0.0)

    def test_local_characteristics_shift_generator_chain_and_match_ito_formula(self):
        from ka_generator_ladder_diffusion import (
            generator_ladder_local_characteristics,
            quadratic_test_generator,
        )

        rng = np.random.default_rng(40)
        coordinates = rng.normal(size=(4, 2, 3))
        terminal = rng.normal(size=(2, 3))
        jacobians = rng.normal(size=(4, 2, 3, 5, 3))
        result = generator_ladder_local_characteristics(
            coordinates,
            terminal_generator=terminal,
            velocity_jacobians=jacobians,
            friction=1.1,
            temperature=0.45,
        )
        expected_drift = np.concatenate((coordinates[1:], terminal[None]), axis=0)
        np.testing.assert_allclose(result["drift"], expected_drift)

        target = 1
        state_dimension = 12
        linear = rng.normal(size=state_dimension)
        raw = rng.normal(size=(state_dimension, state_dimension))
        hessian = 0.5 * (raw + raw.T)
        state = np.transpose(coordinates[:, target], (0, 1)).reshape(-1)
        drift = np.asarray(result["drift"])[:, target].reshape(-1)
        diffusion = np.asarray(result["joint_diffusion"])[target]
        expected_generator = (
            (linear + hessian @ state) @ drift
            + 0.5 * np.einsum("ij,ij->", hessian, diffusion)
        )
        self.assertAlmostEqual(
            quadratic_test_generator(
                state,
                drift=drift,
                diffusion=diffusion,
                linear=linear,
                hessian=hessian,
            ),
            expected_generator,
            places=12,
        )
        self.assertEqual(result["microscopic_state_conditioned"], 1.0)
        self.assertEqual(result["projected_state_autonomous"], 0.0)
        self.assertEqual(result["finite_generator_ladder_closure_supported"], 0.0)

    def test_lp_velocity_jacobian_matches_quadratic_coordinate_difference(self):
        from ka_generator_ladder_diffusion import lp_velocity_jacobian

        rng = np.random.default_rng(41)
        particles = 2
        dimension = 3 * particles
        friction = 0.7
        position = rng.normal(size=dimension)
        velocity = rng.normal(size=dimension)
        force = rng.normal(size=dimension)
        direction = rng.normal(size=dimension)
        linear = rng.normal(size=(3, dimension))
        raw = rng.normal(size=(3, dimension, dimension))
        hessian = 0.5 * (raw + np.swapaxes(raw, 1, 2))
        jacobian_flat = linear + np.einsum("aij,j->ai", hessian, position)
        jacobian_velocity_flat = np.einsum(
            "aij,j->ai",
            hessian,
            velocity,
        )
        jacobian = np.transpose(
            jacobian_flat.reshape(1, 3, particles, 3),
            (0, 2, 1, 3),
        )
        jacobian_velocity = np.transpose(
            jacobian_velocity_flat.reshape(1, 3, particles, 3),
            (0, 2, 1, 3),
        )
        matrix = lp_velocity_jacobian(
            jacobian,
            jacobian_velocity,
            friction=friction,
        )["lp_velocity_jacobian"]

        def lp(current_velocity):
            return (
                np.einsum("aij,i,j->a", hessian, current_velocity, current_velocity)
                + jacobian_flat @ force
                - friction * (jacobian_flat @ current_velocity)
            )

        step = 1e-6
        finite_difference = (
            lp(velocity + step * direction)
            - lp(velocity - step * direction)
        ) / (2.0 * step)
        actual = np.einsum(
            "tanb,nb->ta",
            matrix,
            direction.reshape(particles, 3),
        )[0]
        np.testing.assert_allclose(actual, finite_difference, rtol=2e-9, atol=2e-9)

    def test_joint_generator_diffusion_matches_all_cross_grams_and_is_psd(self):
        from ka_generator_ladder_diffusion import (
            generator_ladder_conditional_diffusion,
        )

        rng = np.random.default_rng(42)
        jacobians = rng.normal(size=(3, 4, 3, 5, 3))
        friction = 1.3
        temperature = 0.58
        result = generator_ladder_conditional_diffusion(
            jacobians,
            friction=friction,
            temperature=temperature,
        )
        expected = 2.0 * friction * temperature * np.einsum(
            "rtanb,stcnb->rstac",
            jacobians,
            jacobians,
        )
        np.testing.assert_allclose(result["diffusion_blocks"], expected)
        blocks = np.asarray(result["diffusion_blocks"])
        np.testing.assert_allclose(
            blocks,
            np.transpose(blocks, (1, 0, 2, 4, 3)),
        )
        joint = np.asarray(result["joint_diffusion"])
        self.assertEqual(joint.shape, (4, 9, 9))
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(joint)), -1e-12)
        self.assertGreater(np.linalg.norm(blocks[0, 1]), 0.0)
        self.assertEqual(result["autonomous_single_particle_gle_allowed"], 0.0)
        self.assertEqual(result["thermodynamic_claim_allowed"], 0.0)

    def test_zero_thermostat_has_zero_joint_diffusion(self):
        from ka_generator_ladder_diffusion import (
            generator_ladder_conditional_diffusion,
        )

        jacobians = np.ones((2, 1, 3, 2, 3))
        for friction, temperature in ((0.0, 0.58), (1.0, 0.0)):
            result = generator_ladder_conditional_diffusion(
                jacobians,
                friction=friction,
                temperature=temperature,
            )
            np.testing.assert_array_equal(result["joint_diffusion"], 0.0)


if __name__ == "__main__":
    unittest.main()
