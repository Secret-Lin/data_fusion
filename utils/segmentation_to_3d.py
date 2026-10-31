#!/usr/bin/env python3
"""Utility to project 2D segmentation masks to 3D world coordinates.

Given:
- Segmentation mask (H x W, bool or uint8)
- Depth map (H x W, in meters)
- Camera intrinsics (K matrix)
- Camera to world transform (T_world_camera)

Returns:
- 3D point cloud of segmented object in world coordinates
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional


def compute_camera_matrix(
    fx: float, fy: float, cx: float, cy: float
) -> NDArray[np.float64]:
    """Build camera intrinsics matrix K."""
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def project_pixel_to_camera(
    u: float, v: float, depth: float, K: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Backproject a single pixel to 3D camera frame.

    Args:
        u, v: Pixel coordinates
        depth: Depth in meters
        K: Camera intrinsics 3x3

    Returns:
        3D point in camera frame [X, Y, Z]
    """
    K_inv = np.linalg.inv(K)
    pixel_hom = np.array([u, v, 1.0], dtype=np.float64)
    point_cam = depth * (K_inv @ pixel_hom)
    return point_cam


def segmentation_to_3d_points(
    mask: NDArray[np.uint8],
    depth: NDArray[np.float32],
    K: NDArray[np.float64],
    T_world_camera: NDArray[np.float64],
    depth_scale: float = 1.0,
    max_depth: float = 10.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project segmentation mask to 3D world point cloud.

    Args:
        mask: Binary mask (H x W), non-zero = foreground
        depth: Depth map (H x W) in meters
        K: Camera intrinsics 3x3
        T_world_camera: 4x4 transform from camera to world
        depth_scale: Multiplier for depth (e.g., 0.001 for mm→m)
        max_depth: Maximum valid depth in meters

    Returns:
        Tuple of:
            - points: N x 3 array of 3D points in world frame
            - colors: N x 3 array of RGB colors (if available in mask)

    Note:
        For RGB colors, pass a colored mask or retrieve from original image.
    """
    if mask.shape != depth.shape:
        raise ValueError(f"Mask shape {mask.shape} != depth shape {depth.shape}")

    # Get valid foreground pixels
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    # Get depths for these pixels
    depths = depth[ys, xs].astype(np.float64) * depth_scale

    # Filter by valid depth
    valid_mask = (depths > 0) & (depths < max_depth)
    xs = xs[valid_mask]
    ys = ys[valid_mask]
    depths = depths[valid_mask]

    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    # Backproject to camera frame
    K_inv = np.linalg.inv(K)
    points_cam = np.zeros((len(xs), 3), dtype=np.float64)

    for i, (u, v, d) in enumerate(zip(xs, ys, depths)):
        pixel_hom = np.array([u, v, 1.0], dtype=np.float64)
        points_cam[i] = d * (K_inv @ pixel_hom)

    # Transform to world frame
    # T_world_camera is 4x4, points_cam is Nx3
    ones = np.ones((len(points_cam), 1), dtype=np.float64)
    points_cam_hom = np.hstack([points_cam, ones])  # N x 4

    points_world = (T_world_camera @ points_cam_hom.T).T[:, :3]  # N x 3

    return points_world, np.zeros((len(points_world), 3), dtype=np.float64)


def segmentation_to_3d_points_batch(
    masks: list[NDArray[np.uint8]],
    depths: list[NDArray[np.float32]],
    K_list: list[NDArray[np.float64]],
    T_world_camera_list: list[NDArray[np.float64]],
    depth_scale: float = 1.0,
    max_depth: float = 10.0,
) -> NDArray[np.float64]:
    """Fuse segmentation point clouds from multiple views.

    Args:
        masks: List of binary masks, one per camera
        depths: List of depth maps
        K_list: List of camera intrinsics
        T_world_camera_list: List of camera to world transforms
        depth_scale: Depth scaling factor
        max_depth: Maximum valid depth

    Returns:
        Combined point cloud in world frame (N x 3)
    """
    all_points = []

    for mask, depth, K, T_cam_w in zip(masks, depths, K_list, T_world_camera_list):
        pts, _ = segmentation_to_3d_points(
            mask, depth, K, T_cam_w, depth_scale, max_depth
        )
        if len(pts) > 0:
            all_points.append(pts)

    if not all_points:
        return np.zeros((0, 3), dtype=np.float64)

    return np.vstack(all_points)


def point_cloud_to_pose(
    points: NDArray[np.float64],
    orientation: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Compute SE(3) pose from point cloud centroid.

    Args:
        points: N x 3 point cloud in world frame (already transformed).
        orientation: Optional 3x3 rotation matrix. If None, identity.

    Returns:
        4x4 SE(3) pose matrix: translation = centroid, rotation = orientation (or I).
        Returns identity matrix if points is empty.
    """
    T = np.eye(4, dtype=np.float64)
    if len(points) == 0:
        return T

    centroid = points.mean(axis=0)
    T[:3, 3] = centroid

    if orientation is not None:
        R = np.array(orientation, dtype=np.float64)
        if R.shape == (3, 3):
            T[:3, :3] = R

    return T


def downsample_point_cloud(
    points: NDArray[np.float64],
    voxel_size: float = 0.01,
) -> NDArray[np.float64]:
    """Downsample point cloud using voxel grid.

    Args:
        points: N x 3 point cloud
        voxel_size: Size of voxel in meters

    Returns:
        Downsampled point cloud
    """
    if len(points) == 0:
        return points

    # Compute voxel indices
    voxel_indices = (points / voxel_size).astype(np.int32)
    unique_voxels, inverse_indices = np.unique(
        voxel_indices, axis=0, return_inverse=True
    )

    # Average points in each voxel
    downsampled = np.zeros((len(unique_voxels), 3), dtype=np.float64)
    for i in range(len(unique_voxels)):
        downsampled[i] = points[inverse_indices == i].mean(axis=0)

    return downsampled


if __name__ == "__main__":
    # Simple test
    # Create dummy data
    H, W = 480, 640
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[100:200, 200:300] = 1  # Rectangle in image

    depth = np.ones((H, W), dtype=np.float32) * 2.0  # 2 meters depth

    K = compute_camera_matrix(500, 500, 320, 240)
    T = np.eye(4, dtype=np.float64)

    points, colors = segmentation_to_3d_points(mask, depth, K, T)
    print(f"Generated {len(points)} points")
    print(f"Sample point: {points[0]}")
