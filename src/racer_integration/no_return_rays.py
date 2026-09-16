"""Geometry helpers for preserving explicit planar LiDAR no-return rays.

The registered sparse cloud used by Swarm-LIO2 contains only finite returns.
For occupancy mapping that is insufficient: a beam which was emitted but had
no return still observes free space up to the sensor's maximum range. These
helpers reconstruct those beam directions from the organized Gazebo cloud and
encode their endpoints with a sentinel intensity understood by RACER.
"""

from __future__ import annotations

import math

import numpy as np


FREE_RAY_INTENSITY = np.float32(-np.finfo(np.float32).max)


def closest_vertical_row(
    height: int,
    minimum_elevation: float,
    maximum_elevation: float,
    target_elevation: float = 0.0,
) -> int:
    if height <= 0:
        raise ValueError("organized cloud height must be positive")
    if not all(
        math.isfinite(value)
        for value in (minimum_elevation, maximum_elevation, target_elevation)
    ):
        raise ValueError("vertical scan angles must be finite")
    if maximum_elevation < minimum_elevation:
        raise ValueError("maximum elevation is below minimum elevation")
    if height == 1 or maximum_elevation == minimum_elevation:
        return 0
    scale = (target_elevation - minimum_elevation) / (
        maximum_elevation - minimum_elevation
    )
    return int(np.clip(round(scale * (height - 1)), 0, height - 1))


def horizontal_azimuths(
    width: int, minimum_azimuth: float, maximum_azimuth: float
) -> np.ndarray:
    if width <= 0:
        raise ValueError("organized cloud width must be positive")
    if not math.isfinite(minimum_azimuth) or not math.isfinite(maximum_azimuth):
        raise ValueError("horizontal scan angles must be finite")
    if maximum_azimuth < minimum_azimuth:
        raise ValueError("maximum azimuth is below minimum azimuth")
    if width == 1:
        return np.asarray([minimum_azimuth], dtype=np.float64)
    return np.linspace(
        minimum_azimuth, maximum_azimuth, width, dtype=np.float64
    )


def no_return_azimuths(
    ring_xyz: np.ndarray,
    minimum_azimuth: float,
    maximum_azimuth: float,
    maximum_range: float,
    maximum_range_margin: float,
) -> np.ndarray:
    """Return angles for beams explicitly reporting no obstacle return.

    Gazebo represents most no-return beams as non-finite XYZ and occasionally
    emits a finite sample at the configured maximum range. A finite sample
    shorter than that is a real hit and is deliberately excluded.
    """

    points = np.asarray(ring_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("ring_xyz must have shape (width, 3)")
    if maximum_range <= 0.0 or not math.isfinite(maximum_range):
        raise ValueError("maximum range must be finite and positive")
    if maximum_range_margin < 0.0 or not math.isfinite(maximum_range_margin):
        raise ValueError("maximum range margin must be finite and nonnegative")

    finite = np.isfinite(points).all(axis=1)
    ranges = np.full(points.shape[0], np.nan, dtype=np.float64)
    ranges[finite] = np.linalg.norm(points[finite], axis=1)
    no_return = ~finite | (
        finite & (ranges >= maximum_range - maximum_range_margin)
    )
    return horizontal_azimuths(
        points.shape[0], minimum_azimuth, maximum_azimuth
    )[no_return]


def quaternion_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    values = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("quaternion must be finite")
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm is zero")
    qx, qy, qz, qw = values / norm
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def planar_free_ray_endpoints(
    azimuths: np.ndarray,
    position_xyz: np.ndarray,
    quaternion_xyzw: np.ndarray,
    maximum_range: float,
) -> np.ndarray:
    """Build common-frame XYZ+sentinel endpoints for planar free rays."""

    angles = np.asarray(azimuths, dtype=np.float64).reshape(-1)
    position = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    if not np.isfinite(angles).all() or not np.isfinite(position).all():
        raise ValueError("ray angles and position must be finite")
    if maximum_range <= 0.0 or not math.isfinite(maximum_range):
        raise ValueError("maximum range must be finite and positive")

    yaw = quaternion_yaw(*quaternion)
    world_angles = angles + yaw
    endpoints = np.empty((len(angles), 4), dtype="<f4")
    endpoints[:, 0] = position[0] + maximum_range * np.cos(world_angles)
    endpoints[:, 1] = position[1] + maximum_range * np.sin(world_angles)
    endpoints[:, 2] = position[2]
    endpoints[:, 3] = FREE_RAY_INTENSITY
    return endpoints


# ---------------------------------------------------------------------------
# 3-D (spherical) variant for a flying 360-degree LiDAR.
#
# The planar helpers above pick one ring; a UAV needs every beam.  The Gazebo
# gpu_lidar cloud is organized height x width with elevation linear in the row
# index and azimuth linear in the column index (verified on quad_mid360:
# row 0 = -40.1 deg, row 95 = +51.6 deg, col 0 = -180 deg, col 179 = +180 deg;
# no-return beams are (-inf,-inf,-inf), some are finite at exactly the
# maximum range).
# ---------------------------------------------------------------------------


def organized_xyz(message) -> np.ndarray:
    """Vectorised decode of an organized PointCloud2 into (height, width, 3)
    float32 with non-finite samples preserved."""
    offsets = {}
    for field in message.fields:
        if field.name in ("x", "y", "z"):
            if int(field.datatype) != 7 or int(field.count) != 1:  # FLOAT32
                raise ValueError(f"{field.name} must be one float32")
            offsets[field.name] = int(field.offset)
    for required in ("x", "y", "z"):
        if required not in offsets:
            raise ValueError(f"PointCloud2 has no {required} field")
    if message.is_bigendian:
        raise ValueError("big-endian PointCloud2 is unsupported")
    height, width, step = int(message.height), int(message.width), int(message.point_step)
    if height <= 0 or width <= 0 or step < 12:
        raise ValueError("invalid organized PointCloud2 layout")
    if int(message.row_step) != width * step:
        raise ValueError("row_step does not match width * point_step")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size < height * width * step:
        raise ValueError("PointCloud2 data shorter than declared layout")
    table = raw[: height * width * step].reshape(height * width, step)
    xyz = np.empty((height * width, 3), dtype="<f4")
    for axis, name in enumerate(("x", "y", "z")):
        o = offsets[name]
        xyz[:, axis] = table[:, o : o + 4].copy().view("<f4").reshape(-1)
    return xyz.reshape(height, width, 3)


def spherical_no_return_directions(
    xyz: np.ndarray,
    minimum_azimuth: float,
    maximum_azimuth: float,
    minimum_elevation: float,
    maximum_elevation: float,
    maximum_range: float,
    maximum_range_margin: float,
    row_stride: int = 1,
    column_stride: int = 1,
) -> np.ndarray:
    """Unit directions (sensor frame) of the beams that reported no return.

    A beam is a no-return when its sample is non-finite or lies at (or beyond)
    maximum_range - maximum_range_margin.  Rows/columns are sub-sampled with
    the given strides to bound the raycasting load downstream.
    """

    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("xyz must have shape (height, width, 3)")
    if maximum_range <= 0.0 or not math.isfinite(maximum_range):
        raise ValueError("maximum range must be finite and positive")
    if maximum_range_margin < 0.0 or not math.isfinite(maximum_range_margin):
        raise ValueError("maximum range margin must be finite and nonnegative")
    if row_stride < 1 or column_stride < 1:
        raise ValueError("strides must be >= 1")
    height, width = points.shape[0], points.shape[1]
    if maximum_elevation < minimum_elevation or maximum_azimuth < minimum_azimuth:
        raise ValueError("scan angle limits are inverted")
    elevations = (
        np.linspace(minimum_elevation, maximum_elevation, height, dtype=np.float64)
        if height > 1
        else np.asarray([minimum_elevation], dtype=np.float64)
    )
    azimuths = horizontal_azimuths(width, minimum_azimuth, maximum_azimuth)
    sub = points[::row_stride, ::column_stride]
    el = elevations[::row_stride]
    az = azimuths[::column_stride]
    # Beam directions of the whole sub-sampled grid (sensor frame).
    ce_all = np.cos(el)[:, None]
    dirs_all = np.stack(
        (
            ce_all * np.cos(az)[None, :],
            ce_all * np.sin(az)[None, :],
            np.repeat(np.sin(el)[:, None], az.size, axis=1),
        ),
        axis=2,
    )
    finite = np.isfinite(sub).all(axis=2)
    ranges = np.full(sub.shape[:2], np.nan, dtype=np.float64)
    ranges[finite] = np.linalg.norm(sub[finite], axis=1)
    # Gazebo's gpu_rays shader clamps r > far to +inf and r < near to -inf
    # and multiplies by the direction cosines: a beam blocked INSIDE the
    # minimum range therefore shows up with every non-zero component's sign
    # flipped.  Only the +inf (nothing within range) case is a free ray; the
    # -inf case is a near obstacle and must not be cast through.
    with np.errstate(invalid="ignore"):
        sign_match = (np.sign(sub) == np.sign(dirs_all)) | (np.abs(dirs_all) < 1e-6)
    far_nonfinite = ~finite & sign_match.all(axis=2)
    no_return = far_nonfinite | (finite & (ranges >= maximum_range - maximum_range_margin))
    rows, cols = np.nonzero(no_return)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return dirs_all[rows, cols]


def quaternion_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    values = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("quaternion must be finite")
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = values / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def spherical_free_ray_endpoints(
    directions: np.ndarray,
    position_xyz: np.ndarray,
    quaternion_xyzw: np.ndarray,
    free_range: float,
) -> np.ndarray:
    """Common-frame XYZ+sentinel endpoints for 3-D free rays."""

    dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    position = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    if not np.isfinite(dirs).all() or not np.isfinite(position).all():
        raise ValueError("ray directions and position must be finite")
    if free_range <= 0.0 or not math.isfinite(free_range):
        raise ValueError("free range must be finite and positive")
    rotation = quaternion_rotation_matrix(*np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4))
    world = dirs @ rotation.T
    endpoints = np.empty((dirs.shape[0], 4), dtype="<f4")
    endpoints[:, :3] = position[None, :] + free_range * world
    endpoints[:, 3] = FREE_RAY_INTENSITY
    return endpoints


def max_range_hits_to_free_rays(
    points_xyzi: np.ndarray,
    position_xyz: np.ndarray,
    maximum_range: float,
    maximum_range_margin: float,
    free_range: float,
) -> np.ndarray:
    """Re-label registered returns at >= maximum_range - margin from the sensor
    as free rays (Gazebo encodes some no-return beams as a finite sample at
    the configured maximum range; registered they would become phantom
    obstacles floating in mid-air)."""

    pts = np.asarray(points_xyzi, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 4:
        raise ValueError("points must have shape (n, 4)")
    if pts.shape[0] == 0:
        return pts.astype("<f4")
    position = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    if free_range <= 0.0 or maximum_range <= 0.0:
        raise ValueError("ranges must be positive")
    offsets = pts[:, :3] - position[None, :]
    ranges = np.linalg.norm(offsets, axis=1)
    far = ranges >= maximum_range - maximum_range_margin
    out = pts.copy()
    if far.any():
        scale = free_range / np.maximum(ranges[far], 1e-9)
        out[far, :3] = position[None, :] + offsets[far] * scale[:, None]
        out[far, 3] = FREE_RAY_INTENSITY
    return out.astype("<f4")
