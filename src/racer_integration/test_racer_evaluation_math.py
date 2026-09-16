#!/usr/bin/env python3

import math
import unittest

import numpy as np

from racer_evaluation_math import (
    apply_rigid_transform_se3,
    fit_rigid_transform_se3,
)


class RacerEvaluationMathTest(unittest.TestCase):
    def test_recovers_full_se3_transform(self):
        source = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.2],
                [0.0, 2.0, -0.3],
                [0.4, -0.2, 1.5],
                [-0.8, 0.5, 0.7],
            ]
        )
        roll, pitch, yaw = (0.2, -0.3, 0.4)
        rx = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(roll), -math.sin(roll)],
                [0.0, math.sin(roll), math.cos(roll)],
            ]
        )
        ry = np.asarray(
            [
                [math.cos(pitch), 0.0, math.sin(pitch)],
                [0.0, 1.0, 0.0],
                [-math.sin(pitch), 0.0, math.cos(pitch)],
            ]
        )
        rz = np.asarray(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        expected_rotation = rz @ ry @ rx
        expected_translation = np.asarray([1.2, -2.3, 0.8])
        target = apply_rigid_transform_se3(
            source, expected_rotation, expected_translation
        )

        rotation, translation = fit_rigid_transform_se3(source, target)
        np.testing.assert_allclose(rotation, expected_rotation, atol=1e-12)
        np.testing.assert_allclose(translation, expected_translation, atol=1e-12)

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            fit_rigid_transform_se3([[0.0, 0.0, 0.0]], [[0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
