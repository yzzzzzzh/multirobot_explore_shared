#!/usr/bin/env python3
"""Regression tests for RACER task-ID ROS message widths."""

from pathlib import Path
import unittest


MSG_DIR = (
    Path(__file__).resolve().parents[1]
    / "RACER"
    / "swarm_exploration"
    / "exploration_manager"
    / "msg"
)


class RacerRosMessageTests(unittest.TestCase):
    def test_hgrid_task_ids_do_not_use_int8(self) -> None:
        expected_fields = {
            "DroneState.msg": (
                "int32[] grid_ids",
                "uint64 assignment_epoch",
                "uint64[] grid_epochs",
                "int32[] failed_grid_ids",
                "float64[] failed_grid_until",
            ),
            "GridIds.msg": ("int32[] ids",),
            "PairOpt.msg": (
                "int32[] ego_ids",
                "int32[] other_ids",
                "uint64 expected_ego_epoch",
                "uint64 expected_other_epoch",
                "uint64 assignment_epoch",
            ),
            "PairOptResponse.msg": (
                "uint64 expected_ego_epoch",
                "uint64 expected_other_epoch",
                "uint64 assignment_epoch",
            ),
            "FrontierShare.msg": (
                "uint64 frontier_epoch",
                "uint64 claimed_signature",
                "uint64[] signatures",
                "uint8[] reserved",
                "int32[] cell_offsets",
            ),
        }
        for filename, fields in expected_fields.items():
            text = (MSG_DIR / filename).read_text(encoding="utf-8")
            # FrontierShare.reserved is deliberately a byte mask; only task-ID
            # arrays must avoid the upstream int8 overflow.
            if filename != "FrontierShare.msg":
                self.assertNotIn("int8[]", text, filename)
            for field in fields:
                self.assertIn(field, text, filename)


if __name__ == "__main__":
    unittest.main()
