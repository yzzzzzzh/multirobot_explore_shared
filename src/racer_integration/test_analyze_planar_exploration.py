#!/usr/bin/env python3

import unittest

import numpy as np

from analyze_planar_exploration import (
    build_masks,
    coverage_checkpoints,
    raycast_visible,
    select_motion_samples,
)


class PlanarExplorationAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.layout = {
            "bounds_xy_m": [-5.0, 5.0, -5.0, 5.0],
            "boxes": [
                {
                    "center": [0.0, 0.0, 1.0],
                    "size": [0.25, 8.0, 2.0],
                },
                {
                    "center": [0.0, 0.0, -0.05],
                    "size": [10.0, 10.0, 0.10],
                },
            ],
        }

    def test_floor_is_not_a_planar_obstacle(self):
        masks = build_masks(self.layout, 0.25, 0.4, 0.65)
        self.assertGreater(np.count_nonzero(masks["navigable"]), 0)
        self.assertLess(
            np.count_nonzero(masks["navigable"]), masks["navigable"].size
        )

    def test_wall_occludes_cells_behind_it(self):
        masks = build_masks(self.layout, 0.25, 0.4, 0.65)
        visible = raycast_visible((-2.0, 0.0), masks, 5.0, 720)
        left_index = np.argmin(np.abs(masks["x"] + 1.0))
        right_index = np.argmin(np.abs(masks["x"] - 1.0))
        center_y = np.argmin(np.abs(masks["y"]))
        self.assertTrue(visible[center_y, left_index])
        self.assertFalse(visible[center_y, right_index])

    def test_motion_sampling_keeps_time_and_distance_progress(self):
        times = np.asarray([0.0, 0.2, 0.4, 1.2, 1.4])
        positions = np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [0.5, 0.0], [0.5, 0.0], [1.0, 0.0]]
        )
        selected = select_motion_samples(times, positions, 0.4, 1.0)
        self.assertEqual(selected.tolist(), [0, 2, 4])

    def test_coverage_checkpoints_report_incremental_unknown_area_rate(self):
        checkpoints = coverage_checkpoints(
            np.asarray([0.0, 30.0, 60.0, 90.0]),
            np.asarray([0.10, 0.25, 0.40, 0.50]),
            navigable_area_m2=1000.0,
            duration_s=90.0,
            interval_s=60.0,
        )
        self.assertEqual([item["time_s"] for item in checkpoints], [60.0, 90.0])
        self.assertAlmostEqual(checkpoints[0]["observed_area_m2"], 400.0)
        self.assertAlmostEqual(checkpoints[0]["new_area_rate_m2_per_s"], 400.0 / 60.0)
        self.assertAlmostEqual(checkpoints[1]["new_area_in_interval_m2"], 100.0)
        self.assertAlmostEqual(checkpoints[1]["new_area_rate_m2_per_s"], 100.0 / 30.0)


if __name__ == "__main__":
    unittest.main()
