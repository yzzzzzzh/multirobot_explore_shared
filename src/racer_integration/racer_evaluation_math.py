#!/usr/bin/env python3
"""Small, testable helpers for standard SE(3)-aligned trajectory metrics."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def fit_rigid_transform_se3(
    source_points: Sequence[Sequence[float]],
    target_points: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit target = R * source + t without scale using Kabsch."""
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching Nx3 shapes")
    if source.shape[0] < 3:
        raise ValueError("at least three paired positions are required")

    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    covariance = source_centered.T @ target_centered
    u_matrix, _, v_transpose = np.linalg.svd(covariance)
    rotation = v_transpose.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        v_transpose[-1, :] *= -1.0
        rotation = v_transpose.T @ u_matrix.T
    translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
    return rotation, translation


def apply_rigid_transform_se3(
    points: Sequence[Sequence[float]],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape Nx3")
    return (rotation @ values.T).T + translation
