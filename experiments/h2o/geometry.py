"""Numerically stable rotation comparisons for rounded H2O annotations."""
from __future__ import annotations

import numpy as np


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    """Project a nearly orthogonal stored matrix onto SO(3), for scoring only.

    This is not a fitted camera/world alignment. Text serialization can make
    R.T @ R differ from I; applying acos(trace) directly can then report an
    angular error even when comparing a matrix with itself.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 3x3 rotation matrix")
    if np.linalg.det(matrix) <= 0 or np.linalg.norm(matrix.T @ matrix - np.eye(3)) > 0.01:
        raise ValueError("Rotation is reflected or too far from SO(3)")
    u, _, vt = np.linalg.svd(matrix)
    return u @ vt


def rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    """SO(3) geodesic distance, stable near both zero and 180 degrees."""
    relative = nearest_rotation(a).T @ nearest_rotation(b)
    sine = 0.5 * np.linalg.norm([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ])
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arctan2(sine, cosine)))
