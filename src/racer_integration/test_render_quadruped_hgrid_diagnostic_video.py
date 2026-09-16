import pathlib
import sys
import unittest

import numpy as np


MODULE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from render_quadruped_hgrid_diagnostic_video import (  # noqa: E402
    TargetRecord,
    fit_log_time_offset,
    time_alignment_diagnostics,
    validate_time_alignment,
)


class RenderQuadrupedHGridDiagnosticVideoTest(unittest.TestCase):
    def setUp(self):
        self.true_offset = 17.25
        times = np.linspace(0.0, 30.0, 301)
        estimate = np.column_stack(
            (0.4 * times, 2.0 * np.sin(0.25 * times), np.full_like(times, 0.55))
        )
        self.run_data = {"t_1": times, "estimate_1": estimate}
        self.targets = []
        for relative_t in np.linspace(2.0, 28.0, 53):
            start = np.asarray(
                [
                    np.interp(relative_t, times, estimate[:, axis])
                    for axis in range(3)
                ]
            )
            self.targets.append(
                TargetRecord(
                    stamp=self.true_offset + relative_t,
                    drone=1,
                    grid_id=100,
                    frontier_id=1,
                    start=start,
                    target=start + np.asarray([1.0, 0.0, 0.0]),
                    frontier_average=start + np.asarray([2.0, 0.0, 0.0]),
                    distance_m=1.0,
                    hgrid_unknown=100,
                    hgrid_gain=80,
                    view_gain=40,
                )
            )

    def test_auto_alignment_recovers_known_offset(self):
        fitted = fit_log_time_offset(self.targets, self.run_data, 30.0)
        self.assertAlmostEqual(fitted, self.true_offset, delta=0.002)
        diagnostics = time_alignment_diagnostics(
            self.targets, self.run_data, 30.0, fitted
        )
        validate_time_alignment(diagnostics)
        self.assertLess(diagnostics["p90_m"], 0.01)

    def test_stale_manual_alignment_is_rejected(self):
        diagnostics = time_alignment_diagnostics(
            self.targets, self.run_data, 30.0, self.true_offset - 2.0
        )
        with self.assertRaisesRegex(RuntimeError, "alignment rejected"):
            validate_time_alignment(diagnostics)


if __name__ == "__main__":
    unittest.main()
