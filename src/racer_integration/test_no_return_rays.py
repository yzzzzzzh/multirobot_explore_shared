#!/usr/bin/env python3

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from no_return_rays import (
    FREE_RAY_INTENSITY,
    closest_vertical_row,
    no_return_azimuths,
    planar_free_ray_endpoints,
)


class NoReturnRayTest(unittest.TestCase):
    def test_selects_nearest_horizontal_mid360_ring(self):
        self.assertEqual(closest_vertical_row(64, -0.126, 0.907, 0.0), 8)

    def test_only_nonfinite_and_max_range_samples_are_free_rays(self):
        ring = np.asarray(
            [
                [4.0, 0.0, 0.0],
                [math.nan, math.nan, math.nan],
                [9.97, 0.0, 0.0],
                [9.80, 0.0, 0.0],
            ]
        )
        angles = no_return_azimuths(
            ring,
            -math.pi,
            math.pi,
            maximum_range=10.0,
            maximum_range_margin=0.05,
        )
        expected = np.linspace(-math.pi, math.pi, 4)[[1, 2]]
        np.testing.assert_allclose(angles, expected)

    def test_builds_free_endpoints_in_common_frame(self):
        endpoints = planar_free_ray_endpoints(
            np.asarray([0.0, math.pi / 2.0]),
            np.asarray([1.0, 2.0, 0.645]),
            np.asarray(
                [
                    0.0,
                    0.0,
                    math.sin(math.pi / 4.0),
                    math.cos(math.pi / 4.0),
                ]
            ),
            10.0,
        )
        np.testing.assert_allclose(
            endpoints[:, :3],
            np.asarray([[1.0, 12.0, 0.645], [-9.0, 2.0, 0.645]]),
            atol=1.0e-5,
        )
        self.assertTrue(np.all(endpoints[:, 3] == FREE_RAY_INTENSITY))


if __name__ == "__main__":
    unittest.main()
