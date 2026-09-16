#!/usr/bin/env python3

import math
import unittest

from racer_control_math import (
    apply_obstacle_velocity_barrier,
    apply_separation_barrier,
    bootstrap_excitation_offset,
    calculate_world_velocity,
    conjugate_quaternion,
    ground_unicycle_command,
    limit_velocity_change,
    multiply_quaternions,
    obstacle_braking_scale,
    position_inside_bounds,
    rotate_body_to_world,
    rotate_world_to_body,
    wrap_angle,
)


class RacerControlMathTest(unittest.TestCase):
    def test_full_attitude_world_to_body_rotation(self):
        # Body yawed +90 degrees: world +X is body -Y.
        half = math.sqrt(0.5)
        result = rotate_world_to_body((1.0, 0.0, 0.0), (0.0, 0.0, half, half))
        self.assertAlmostEqual(result[0], 0.0, places=12)
        self.assertAlmostEqual(result[1], -1.0, places=12)
        self.assertAlmostEqual(result[2], 0.0, places=12)

    def test_roll_and_pitch_are_not_discarded(self):
        # +90 degree pitch maps world +X to body +Z.
        half = math.sqrt(0.5)
        result = rotate_world_to_body((1.0, 0.0, 0.0), (0.0, half, 0.0, half))
        self.assertAlmostEqual(result[0], 0.0, places=12)
        self.assertAlmostEqual(result[1], 0.0, places=12)
        self.assertAlmostEqual(result[2], 1.0, places=12)

    def test_quaternion_inverse_and_forward_rotation(self):
        half = math.sqrt(0.5)
        q = (half, 0.0, 0.0, half)
        identity = multiply_quaternions(q, conjugate_quaternion(q))
        self.assertAlmostEqual(identity[0], 0.0, places=12)
        self.assertAlmostEqual(identity[1], 0.0, places=12)
        self.assertAlmostEqual(identity[2], 0.0, places=12)
        self.assertAlmostEqual(identity[3], 1.0, places=12)
        result = rotate_body_to_world((0.0, 1.0, 0.0), q)
        self.assertAlmostEqual(result[0], 0.0, places=12)
        self.assertAlmostEqual(result[1], 0.0, places=12)
        self.assertAlmostEqual(result[2], 1.0, places=12)

    def test_velocity_saturation_preserves_horizontal_direction(self):
        result = calculate_world_velocity(
            (0.0, 0.0, 0.0),
            (3.0, 4.0, 2.0),
            (0.0, 0.0, 0.0),
            kp=1.0,
            max_horizontal_speed=1.0,
            max_vertical_speed=0.7,
        )
        self.assertAlmostEqual(result[0], 0.6, places=12)
        self.assertAlmostEqual(result[1], 0.8, places=12)
        self.assertAlmostEqual(result[2], 0.7, places=12)

    def test_velocity_change_respects_three_dimensional_acceleration(self):
        result = limit_velocity_change(
            (0.0, 0.0, 0.0),
            (3.0, 4.0, 0.0),
            max_acceleration=2.0,
            dt=0.5,
        )
        self.assertAlmostEqual(result[0], 0.6, places=12)
        self.assertAlmostEqual(result[1], 0.8, places=12)
        self.assertAlmostEqual(result[2], 0.0, places=12)

    def test_velocity_change_passes_a_command_inside_envelope(self):
        result = limit_velocity_change(
            (0.1, -0.1, 0.0),
            (0.2, -0.2, 0.1),
            max_acceleration=2.0,
            dt=0.5,
        )
        self.assertEqual(result, (0.2, -0.2, 0.1))

    def test_obstacle_braking_scale(self):
        self.assertAlmostEqual(
            obstacle_braking_scale(1.5, 2.025, 0.55, 0.35, 1.0),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            obstacle_braking_scale(1.5, 0.90, 0.55, 0.35, 1.0),
            0.0,
            places=6,
        )
        scale = obstacle_braking_scale(1.5, 1.40, 0.55, 0.35, 1.0)
        self.assertGreater(scale, 0.0)
        self.assertLess(scale, 1.0)
        self.assertEqual(
            obstacle_braking_scale(0.0, 0.1, 0.55, 0.35, 1.0),
            1.0,
        )

    def test_obstacle_barrier_limits_new_approach(self):
        result = apply_obstacle_velocity_barrier(
            (1.5, 0.2, 0.0),
            (0.0, 0.0, 0.0),
            ((1.4, 0.0, 0.0),),
            protected_clearance=0.9,
            repulsion_activation_distance=1.1,
            repulsion_speed=0.8,
            braking_acceleration=1.0,
        )
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 0.2)

    def test_obstacle_barrier_actively_reverses_unsafe_momentum(self):
        result = apply_obstacle_velocity_barrier(
            (1.5, 0.0, 0.0),
            (1.5, 0.0, 0.0),
            ((1.0, 0.0, 0.0),),
            protected_clearance=0.9,
            repulsion_activation_distance=1.1,
            repulsion_speed=0.8,
            braking_acceleration=1.0,
        )
        self.assertLess(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.0)

    def test_obstacle_barrier_recovers_from_stationary_deadlock(self):
        result = apply_obstacle_velocity_barrier(
            (0.0, 0.4, 0.0),
            (0.0, 0.0, 0.0),
            ((0.95, 0.0, 0.0),),
            protected_clearance=0.9,
            repulsion_activation_distance=1.1,
            repulsion_speed=0.8,
            braking_acceleration=1.0,
        )
        self.assertLess(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.4)

    def test_separation_barrier_leaves_distant_neighbor_unchanged(self):
        result = apply_separation_barrier(
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            ((4.0, 0.0, 0.0),),
            activation_distance=3.0,
            hard_distance=2.2,
            repulsion_speed=0.8,
            braking_acceleration=1.0,
        )
        self.assertEqual(result, (0.5, 0.0, 0.0))

    def test_separation_barrier_removes_approach_and_repels(self):
        result = apply_separation_barrier(
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            ((1.5, 0.0, 0.0),),
            activation_distance=3.0,
            hard_distance=2.2,
            repulsion_speed=0.8,
            braking_acceleration=1.0,
        )
        self.assertLess(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.0)

    def test_separation_barrier_limits_inward_speed_for_braking(self):
        result = apply_separation_barrier(
            (0.0, 0.0, 0.0),
            (1.5, 0.0, 0.0),
            ((4.0, 0.0, 0.0),),
            activation_distance=4.8,
            hard_distance=2.2,
            repulsion_speed=0.0,
            braking_acceleration=1.0,
        )
        self.assertAlmostEqual(result[0], math.sqrt(1.8))

    def test_bounds_and_angle_wrap(self):
        self.assertTrue(
            position_inside_bounds(
                (-24.5, 0.0, 49.0), (-24.5, -24.5, 1.0), (24.5, 24.5, 49.0)
            )
        )
        self.assertFalse(
            position_inside_bounds(
                (-24.6, 0.0, 49.0), (-24.5, -24.5, 1.0), (24.5, 24.5, 49.0)
            )
        )
        self.assertAlmostEqual(wrap_angle(3.0 * math.pi), math.pi, places=12)

    def test_ground_unicycle_turns_before_large_heading_change(self):
        speed, yaw_rate, executed = ground_unicycle_command(
            (0.0, 1.5, 0.0),
            current_yaw=0.0,
            max_forward_speed=0.8,
            max_yaw_rate=1.2,
            heading_kp=2.0,
            turn_in_place_angle=math.radians(55.0),
        )
        self.assertEqual(speed, 0.0)
        self.assertAlmostEqual(yaw_rate, 1.2)
        self.assertEqual(executed, (0.0, 0.0, 0.0))

    def test_ground_unicycle_forward_projection_respects_heading(self):
        speed, yaw_rate, executed = ground_unicycle_command(
            (1.0, 1.0, 0.0),
            current_yaw=0.0,
            max_forward_speed=0.8,
            max_yaw_rate=1.2,
            heading_kp=1.0,
            turn_in_place_angle=math.pi / 2.0,
        )
        self.assertAlmostEqual(speed, 0.8 / math.sqrt(2.0))
        self.assertAlmostEqual(yaw_rate, math.pi / 4.0)
        self.assertAlmostEqual(executed[0], speed)
        self.assertAlmostEqual(executed[1], 0.0)
        self.assertEqual(executed[2], 0.0)

    def test_bootstrap_excitation_is_closed_and_three_dimensional(self):
        kwargs = dict(
            radius=0.6,
            climb_height=0.7,
            vertical_amplitude=0.2,
            climb_duration=4.0,
            loop_duration=16.0,
        )
        self.assertEqual(
            bootstrap_excitation_offset(0.0, **kwargs),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            bootstrap_excitation_offset(24.0, **kwargs),
            (0.0, 0.0, 0.0),
        )
        quarter = bootstrap_excitation_offset(8.0, **kwargs)
        self.assertAlmostEqual(quarter[0], 0.6, places=12)
        self.assertAlmostEqual(quarter[1], 0.6, places=12)
        self.assertAlmostEqual(quarter[2], 0.7, places=12)
        eighth = bootstrap_excitation_offset(6.0, **kwargs)
        self.assertNotAlmostEqual(eighth[0], 0.0, places=12)
        self.assertNotAlmostEqual(eighth[1], 0.0, places=12)
        self.assertNotAlmostEqual(eighth[2], 0.7, places=12)

    def test_bootstrap_direction_rotates_horizontal_loop(self):
        base = bootstrap_excitation_offset(
            8.0, 0.6, 0.7, 0.2, 4.0, 16.0, 0.0
        )
        rotated = bootstrap_excitation_offset(
            8.0, 0.6, 0.7, 0.2, 4.0, 16.0, math.pi / 2.0
        )
        self.assertAlmostEqual(rotated[0], -base[1], places=12)
        self.assertAlmostEqual(rotated[1], base[0], places=12)
        self.assertAlmostEqual(rotated[2], base[2], places=12)

    def test_ground_bootstrap_can_use_a_safe_corridor_ellipse(self):
        circular = bootstrap_excitation_offset(
            8.0, 1.8, 0.0, 0.0, 4.0, 16.0, 0.0
        )
        corridor = bootstrap_excitation_offset(
            8.0, 1.8, 0.0, 0.0, 4.0, 16.0, 0.0, 0.25
        )
        self.assertAlmostEqual(corridor[0], circular[0], places=12)
        self.assertAlmostEqual(corridor[1], 0.25 * circular[1], places=12)
        self.assertEqual(corridor[2], 0.0)


if __name__ == "__main__":
    unittest.main()
