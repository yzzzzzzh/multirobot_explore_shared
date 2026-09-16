#!/usr/bin/env python3
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from no_return_rays import (
    FREE_RAY_INTENSITY,
    max_range_hits_to_free_rays,
    organized_xyz,
    quaternion_rotation_matrix,
    spherical_free_ray_endpoints,
    spherical_no_return_directions,
)


def _far_inf(el, az):
    """Gazebo-style no-return sample: +inf times the direction cosines."""
    d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return np.where(np.abs(d) < 1e-6, np.inf, np.sign(d) * np.inf).astype(np.float32)


def _far_inf_grid(h, w, v_min, v_max, h_min, h_max):
    els = np.linspace(v_min, v_max, h) if h > 1 else np.array([v_min])
    azs = np.linspace(h_min, h_max, w) if w > 1 else np.array([h_min])
    out = np.empty((h, w, 3), dtype=np.float32)
    for i, e in enumerate(els):
        for j, a in enumerate(azs):
            out[i, j] = _far_inf(e, a)
    return out


class _Field:
    def __init__(self, name, offset):
        self.name, self.offset, self.datatype, self.count = name, offset, 7, 1


class _Msg:
    def __init__(self, xyz, point_step=16):
        h, w = xyz.shape[:2]
        self.height, self.width, self.point_step = h, w, point_step
        self.row_step = w * point_step
        self.is_bigendian = False
        self.fields = [_Field("x", 0), _Field("y", 4), _Field("z", 8), _Field("intensity", 12)]
        table = np.zeros((h * w, point_step), dtype=np.uint8)
        flat = xyz.reshape(-1, 3).astype("<f4")
        for a in range(3):
            table[:, 4 * a : 4 * a + 4] = flat[:, a : a + 1].view(np.uint8)
        self.data = table.tobytes()


class SphericalNoReturnTest(unittest.TestCase):
    def test_organized_decode_keeps_nonfinite(self):
        xyz = np.full((2, 3, 3), -np.inf, dtype=np.float32)
        xyz[1, 2] = (1.0, 2.0, 3.0)
        out = organized_xyz(_Msg(xyz))
        self.assertEqual(out.shape, (2, 3, 3))
        self.assertTrue(np.isneginf(out[0, 0]).all())
        np.testing.assert_allclose(out[1, 2], (1.0, 2.0, 3.0))

    def test_directions_follow_row_and_column_angles(self):
        h, w = 5, 8
        xyz = _far_inf_grid(h, w, -0.5, 0.5, -math.pi, math.pi)  # everything no-return
        xyz[2, 0] = (4.0, 0.0, 0.0)                           # one real hit
        xyz[2, 4] = (-9.98, 0.0, 0.0)                         # max-range artefact
        dirs = spherical_no_return_directions(
            xyz, -math.pi, math.pi, -0.5, 0.5, 10.0, 0.05, 1, 1
        )
        self.assertEqual(dirs.shape[0], h * w - 1)
        np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-9)
        # row 0 = lowest elevation, row 4 = highest
        self.assertLess(dirs[:, 2].min(), -0.47)
        self.assertGreater(dirs[:, 2].max(), 0.47)

    def test_strides_subsample(self):
        xyz = _far_inf_grid(6, 6, -0.7, 0.9, -math.pi, math.pi)
        dirs = spherical_no_return_directions(
            xyz, -math.pi, math.pi, -0.7, 0.9, 10.0, 0.05, 3, 2
        )
        self.assertEqual(dirs.shape[0], 2 * 3)

    def test_endpoints_use_full_rotation(self):
        d = np.asarray([[1.0, 0.0, 0.0]])
        # 90 deg about z: +x -> +y
        q = np.asarray([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
        e = spherical_free_ray_endpoints(d, np.asarray([1.0, 2.0, 3.0]), q, 5.0)
        np.testing.assert_allclose(e[0, :3], (1.0, 7.0, 3.0), atol=1e-6)
        self.assertEqual(e[0, 3], FREE_RAY_INTENSITY)
        # 90 deg about y: +x -> -z (pitch must not be discarded)
        q = np.asarray([0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)])
        e = spherical_free_ray_endpoints(d, np.zeros(3), q, 2.0)
        np.testing.assert_allclose(e[0, :3], (0.0, 0.0, -2.0), atol=1e-6)

    def test_rotation_matrix_is_orthonormal(self):
        r = quaternion_rotation_matrix(0.1, -0.2, 0.3, 0.9)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)

    def test_max_range_hits_relabelled(self):
        pts = np.asarray([[4.0, 0.0, 0.0, 10.0], [9.98, 0.0, 0.0, 10.0]])
        out = max_range_hits_to_free_rays(pts, np.zeros(3), 10.0, 0.05, 9.9)
        self.assertEqual(out[0, 3], 10.0)
        np.testing.assert_allclose(out[1, :3], (9.9, 0.0, 0.0), atol=1e-5)
        self.assertEqual(out[1, 3], FREE_RAY_INTENSITY)




class BlockedNearBeamTest(unittest.TestCase):
    def test_minus_inf_beams_are_not_free_rays(self):
        # 1 row at elevation 0, 4 columns; column 0 is a +inf no-return,
        # column 2 is a -inf (blocked inside range_min) sample.
        xyz = np.full((1, 4, 3), np.nan, dtype=np.float32)
        az = np.linspace(-math.pi, math.pi, 4)
        d0 = np.array([math.cos(az[0]), math.sin(az[0]), 0.0])
        d2 = np.array([math.cos(az[2]), math.sin(az[2]), 0.0])
        xyz[0, 0] = np.where(np.abs(d0) < 1e-6, 0.0, np.sign(d0) * np.inf)
        xyz[0, 2] = np.where(np.abs(d2) < 1e-6, 0.0, -np.sign(d2) * np.inf)
        xyz[0, 1] = (3.0, 0.0, 0.0)  # real hit
        xyz[0, 3] = (3.0, 0.0, 0.0)  # real hit
        dirs = spherical_no_return_directions(xyz, -math.pi, math.pi, 0.0, 0.0, 10.0, 0.05, 1, 1)
        self.assertEqual(dirs.shape[0], 1)
        np.testing.assert_allclose(dirs[0], d0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
