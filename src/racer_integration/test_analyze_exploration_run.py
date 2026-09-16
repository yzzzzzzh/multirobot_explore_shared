import unittest

import numpy as np

from analyze_exploration_run import (
    longest_confined_duration,
    motion_summary,
    planner_summary,
    viewpoint_information_summary,
)


class ExplorationMotionAnalysisTest(unittest.TestCase):
    def test_detects_long_confined_jitter(self):
        times = np.arange(0.0, 121.0)
        positions = np.column_stack(
            (
                0.4 * np.sin(times),
                0.4 * np.cos(times),
                0.2 * np.sin(0.5 * times),
            )
        )
        summary = motion_summary(times, positions)
        self.assertGreater(summary["mean_speed_mps"], 0.08)
        self.assertEqual(summary["longest_confined_s"], 120.0)
        self.assertLess(summary["last_60s_displacement_m"], 1.0)

    def test_does_not_call_directed_motion_confined(self):
        times = np.arange(0.0, 121.0)
        positions = np.column_stack((0.2 * times, np.zeros_like(times), np.zeros_like(times)))
        self.assertLess(longest_confined_duration(times, positions), 20.0)

    def test_summarizes_viewpoint_unknown_gain_and_runtime(self):
        records = [
            (
                1.0,
                "viewpoint_gain",
                {
                    "drone": "1",
                    "views": "2",
                    "min_cells": "10",
                    "mean_cells": "20",
                    "max_cells": "30",
                    "mean_planar_area_m2": "1.25",
                    "max_planar_area_m2": "1.875",
                },
            ),
            (
                2.0,
                "viewpoint_gain",
                {
                    "drone": "2",
                    "views": "1",
                    "min_cells": "40",
                    "mean_cells": "40",
                    "max_cells": "40",
                    "mean_planar_area_m2": "2.5",
                    "max_planar_area_m2": "2.5",
                },
            ),
            (2.0, "frontier", {"view_ms": "3.0", "total_ms": "5.0"}),
        ]
        gain = viewpoint_information_summary(records)
        self.assertEqual(gain["selected_viewpoints"], 3)
        self.assertAlmostEqual(
            gain["unknown_cells_per_selected_view_mean"], 80.0 / 3.0
        )
        self.assertEqual(gain["unknown_cells_per_selected_view_min"], 10)
        self.assertEqual(gain["unknown_cells_per_selected_view_max"], 40)
        self.assertEqual(set(gain["by_drone"]), {"bot1", "bot2"})

        planning = planner_summary(
            [(1.0, "plan", {"drone": "1", "wall_ms": "7", "result": "2"})]
            + records
        )
        self.assertEqual(planning["viewpoint_evaluation"]["mean"], 3.0)
        self.assertEqual(planning["frontier_update"]["mean"], 5.0)


if __name__ == "__main__":
    unittest.main()
