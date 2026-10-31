from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np


def _camera_matrix_from_meta(camera_meta: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(camera_meta, dict):
        return None
    K = camera_meta.get("K")
    if K is None:
        return None
    try:
        K_np = np.array(K, dtype=np.float64).reshape(3, 3)
    except Exception:
        return None
    if K_np[0, 0] <= 0.0 or K_np[1, 1] <= 0.0:
        return None
    return K_np


def backproject_masked_depth(
    *,
    mask: np.ndarray,
    depth_u16: np.ndarray,
    camera_meta: dict[str, Any] | None,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Back-project masked depth pixels to camera-frame 3D points."""
    if depth_u16 is None or mask is None:
        return np.zeros((0, 3), dtype=np.float64)
    if depth_u16.ndim != 2:
        return np.zeros((0, 3), dtype=np.float64)

    K = _camera_matrix_from_meta(camera_meta)
    if K is None:
        return np.zeros((0, 3), dtype=np.float64)

    depth_scale_m = 0.001
    if isinstance(camera_meta, dict):
        try:
            depth_scale_m = float(camera_meta.get("depth_scale_m", 0.001))
        except Exception:
            depth_scale_m = 0.001
    if depth_scale_m <= 0.0:
        depth_scale_m = 0.001

    mask_bin = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask_bin)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    d_m = depth_u16[ys, xs].astype(np.float64) * depth_scale_m
    valid = np.isfinite(d_m) & (d_m > min_depth_m) & (d_m < max_depth_m)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)

    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = d_m[valid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if fx <= 1e-9 or fy <= 1e-9:
        return np.zeros((0, 3), dtype=np.float64)

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    pts = np.stack([x, y, z], axis=1)
    return pts.astype(np.float64)


def transform_points(
    T_world_cam: np.ndarray | None, points_cam: np.ndarray
) -> np.ndarray:
    if T_world_cam is None or points_cam.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if T_world_cam.shape != (4, 4):
        return np.zeros((0, 3), dtype=np.float64)
    ones = np.ones((points_cam.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([points_cam, ones])
    pts_w = (T_world_cam @ pts_h.T).T[:, :3]
    return pts_w


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if points.size == 0:
        return points
    voxel = max(float(voxel_size_m), 1e-4)
    idx = np.floor(points / voxel).astype(np.int64)
    uniq, inv = np.unique(idx, axis=0, return_inverse=True)
    out = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    for i in range(uniq.shape[0]):
        out[i] = points[inv == i].mean(axis=0)
    return out


def remove_outliers_statistical(
    points: np.ndarray,
    *,
    min_neighbors: int,
    std_mul: float,
) -> np.ndarray:
    """Simple outlier removal using k-nearest mean distance threshold."""
    if points.size == 0 or points.shape[0] < max(min_neighbors + 1, 8):
        return points

    # O(N^2) distance matrix - acceptable for downsampled clouds.
    diff = points[:, None, :] - points[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)

    k = max(int(min_neighbors), 1)
    # kth partition to avoid full sort cost
    part = np.partition(d2, kth=min(k - 1, d2.shape[1] - 1), axis=1)
    neigh = np.sqrt(part[:, :k])
    mean_dist = np.mean(neigh, axis=1)

    mu = float(np.mean(mean_dist))
    sigma = float(np.std(mean_dist))
    if sigma <= 1e-9:
        return points

    thr = mu + float(std_mul) * sigma
    keep = mean_dist <= thr
    return points[keep]


def largest_cluster(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Keep largest connected component in voxelized space."""
    if points.size == 0:
        return points

    voxel = max(float(voxel_size_m), 1e-4)
    vidx = np.floor(points / voxel).astype(np.int64)
    uniq, inv = np.unique(vidx, axis=0, return_inverse=True)
    if uniq.shape[0] <= 1:
        return points

    key_to_i = {tuple(v.tolist()): i for i, v in enumerate(uniq)}
    nbrs = defaultdict(list)
    offsets = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]
    for i, v in enumerate(uniq):
        base = tuple(v.tolist())
        for dx, dy, dz in offsets:
            nb = (base[0] + dx, base[1] + dy, base[2] + dz)
            j = key_to_i.get(nb)
            if j is not None:
                nbrs[i].append(j)

    seen = np.zeros((uniq.shape[0],), dtype=bool)
    components: list[list[int]] = []
    for i in range(uniq.shape[0]):
        if seen[i]:
            continue
        q = deque([i])
        seen[i] = True
        comp: list[int] = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in nbrs[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        components.append(comp)

    largest_comp = max(components, key=len)
    keep_vox = np.zeros((uniq.shape[0],), dtype=bool)
    keep_vox[largest_comp] = True
    keep_points = keep_vox[inv]
    return points[keep_points]
