from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from core.coordinate_frames import transform_to_y_down


TagPointKey = tuple[int, int, int]
TagPoseMap = dict[int, np.ndarray]


@dataclass(frozen=True)
class RigidAlignment:
    transform: np.ndarray
    common_points: int
    median_residual_m: float
    p95_residual_m: float


def _detector() -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    return cv2.aruco.ArucoDetector(
        dictionary, cv2.aruco.DetectorParameters()
    )


def extract_tag_camera_poses(
    color_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    tag_size_m: float = 0.04,
    detector: cv2.aruco.ArucoDetector | None = None,
) -> TagPoseMap:
    """Estimate camera<-tag poses without relying on cube face geometry."""
    detector = detector or _detector()
    image = np.asarray(color_bgr)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    half = float(tag_size_m) / 2.0
    object_points = np.array(
        [
            [-half, -half, 0.0],
            [+half, -half, 0.0],
            [+half, +half, 0.0],
            [-half, +half, 0.0],
        ],
        dtype=np.float64,
    )
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    poses: TagPoseMap = {}
    for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
        tag_id = int(marker_id)
        if not 0 <= tag_id <= 5:
            continue
        image_points = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
        candidates: list[tuple[float, np.ndarray]] = []
        try:
            result = cv2.solvePnPGeneric(
                object_points,
                image_points,
                K,
                np.zeros(5),
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error:
            result = None
        if result and bool(result[0]):
            pose_candidates = zip(result[1], result[2])
        else:
            pose_candidates = []
        for rvec, tvec in pose_candidates:
            rotation, _ = cv2.Rodrigues(rvec)
            camera_points = (
                rotation @ object_points.T + np.asarray(tvec).reshape(3, 1)
            ).T
            if np.any(camera_points[:, 2] <= 0.0):
                continue
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, K, np.zeros(5)
            )
            error = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (projected.reshape(-1, 2) - image_points) ** 2,
                            axis=1,
                        )
                    )
                )
            )
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation
            transform[:3, 3] = np.asarray(tvec).reshape(3)
            candidates.append((error, transform))
        if candidates:
            poses[tag_id] = min(candidates, key=lambda item: item[0])[1]
    return poses


def _average_pose_values(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("Cannot average an empty transform collection")
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = Rotation.from_matrix(
        np.stack([value[:3, :3] for value in transforms])
    ).mean().as_matrix()
    output[:3, 3] = np.median(
        np.stack([value[:3, 3] for value in transforms]), axis=0
    )
    return output


def aggregate_fixed_tag_poses(
    h5_path: str | Path,
    *,
    max_frames: int = 80,
    tag_size_m: float = 0.04,
) -> TagPoseMap:
    samples: dict[int, list[np.ndarray]] = defaultdict(list)
    detector = _detector()
    with h5py.File(Path(h5_path), "r") as handle:
        K = _fixed_h5_camera_matrix(handle)
        count = int(handle["color"].shape[0])
        indices = np.unique(
            np.linspace(0, count - 1, min(max_frames, count), dtype=np.int64)
        )
        for index in indices:
            poses = extract_tag_camera_poses(
                np.asarray(handle["color"][index]),
                K,
                tag_size_m=tag_size_m,
                detector=detector,
            )
            for tag_id, pose in poses.items():
                samples[tag_id].append(pose)
    return {
        tag_id: _average_pose_values(values)
        for tag_id, values in samples.items()
        if len(values) >= 3
    }


def align_common_tag_poses(
    source_camera_tags: TagPoseMap,
    target_camera_tags: TagPoseMap,
) -> tuple[np.ndarray, dict[str, float | int | list[int]]]:
    """Return target-camera<-source-camera from common physical tag frames."""
    common = sorted(set(source_camera_tags) & set(target_camera_tags))
    if not common:
        raise ValueError("Cameras do not share a visible AprilTag")
    candidates = [
        target_camera_tags[tag_id] @ np.linalg.inv(source_camera_tags[tag_id])
        for tag_id in common
    ]
    transform = _average_pose_values(candidates)
    translation_residuals = [
        float(np.linalg.norm(candidate[:3, 3] - transform[:3, 3]))
        for candidate in candidates
    ]
    rotation_residuals = [
        float(
            np.degrees(
                Rotation.from_matrix(
                    transform[:3, :3].T @ candidate[:3, :3]
                ).magnitude()
            )
        )
        for candidate in candidates
    ]
    return transform, {
        "common_tag_ids": common,
        "common_tags": len(common),
        "max_translation_disagreement_m": max(translation_residuals),
        "max_rotation_disagreement_deg": max(rotation_residuals),
    }


def _sample_depth_m(
    depth_raw: np.ndarray,
    x: float,
    y: float,
    depth_scale_m: float,
    patch_radius: int,
) -> float | None:
    height, width = depth_raw.shape[:2]
    cx = int(round(x))
    cy = int(round(y))
    x1 = max(0, cx - patch_radius)
    x2 = min(width, cx + patch_radius + 1)
    y1 = max(0, cy - patch_radius)
    y2 = min(height, cy + patch_radius + 1)
    values = np.asarray(depth_raw[y1:y2, x1:x2], dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return None
    depth_m = float(np.median(values) * depth_scale_m)
    if not 0.05 <= depth_m <= 5.0:
        return None
    return depth_m


def extract_tag_rgbd_points(
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    camera_matrix: np.ndarray,
    depth_scale_m: float,
    *,
    detector: cv2.aruco.ArucoDetector | None = None,
    grid_positions: tuple[float, ...] = (0.25, 0.5, 0.75),
    patch_radius: int = 2,
) -> dict[TagPointKey, np.ndarray]:
    """Deproject corresponding interior points on every detected cube tag."""
    detector = detector or _detector()
    image = np.asarray(color_bgr)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    canonical = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    points: dict[TagPointKey, np.ndarray] = {}
    for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
        tag_id = int(marker_id)
        if not 0 <= tag_id <= 5:
            continue
        quad = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        homography = cv2.getPerspectiveTransform(canonical, quad)
        normalized = np.array(
            [
                [u, v]
                for v in grid_positions
                for u in grid_positions
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        pixels = cv2.perspectiveTransform(normalized, homography).reshape(-1, 2)
        for index, ((u, v), (x, y)) in enumerate(
            zip(normalized.reshape(-1, 2), pixels)
        ):
            depth_m = _sample_depth_m(
                depth_raw, float(x), float(y), depth_scale_m, patch_radius
            )
            if depth_m is None:
                continue
            point = np.array(
                [
                    (float(x) - cx) * depth_m / fx,
                    (float(y) - cy) * depth_m / fy,
                    depth_m,
                ],
                dtype=np.float64,
            )
            grid_size = len(grid_positions)
            points[(tag_id, index // grid_size, index % grid_size)] = point
    return points


def rigid_align_points(
    source: np.ndarray, target: np.ndarray
) -> RigidAlignment:
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if len(source) < 3 or len(source) != len(target):
        raise ValueError("Rigid alignment requires at least three paired points")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    residuals = np.linalg.norm(
        (rotation @ source.T).T + translation - target, axis=1
    )
    return RigidAlignment(
        transform=transform,
        common_points=int(len(source)),
        median_residual_m=float(np.median(residuals)),
        p95_residual_m=float(np.percentile(residuals, 95)),
    )


def align_tag_point_maps(
    source: dict[TagPointKey, np.ndarray],
    target: dict[TagPointKey, np.ndarray],
) -> RigidAlignment:
    common = sorted(set(source) & set(target))
    if len(common) < 3:
        raise ValueError(
            f"Fewer than three common RGB-D tag points ({len(common)})"
        )
    return rigid_align_points(
        np.stack([source[key] for key in common]),
        np.stack([target[key] for key in common]),
    )


def _fixed_h5_camera_matrix(handle: h5py.File) -> np.ndarray:
    value = handle.attrs.get("K_json")
    if value is None:
        raise ValueError("Fixed RGB-D H5 has no K_json attribute")
    return np.asarray(json.loads(str(value)), dtype=np.float64).reshape(3, 3)


def aggregate_fixed_tag_points(
    h5_path: str | Path,
    *,
    max_frames: int = 80,
) -> dict[TagPointKey, np.ndarray]:
    samples: dict[TagPointKey, list[np.ndarray]] = defaultdict(list)
    detector = _detector()
    with h5py.File(Path(h5_path), "r") as handle:
        K = _fixed_h5_camera_matrix(handle)
        depth_scale = float(handle.attrs.get("depth_scale_m", 0.001))
        count = int(handle["color"].shape[0])
        indices = np.unique(
            np.linspace(0, count - 1, min(max_frames, count), dtype=np.int64)
        )
        for index in indices:
            frame_points = extract_tag_rgbd_points(
                np.asarray(handle["color"][index]),
                np.asarray(handle["depth"][index]),
                K,
                depth_scale,
                detector=detector,
            )
            for key, point in frame_points.items():
                samples[key].append(point)
    return {
        key: np.median(np.stack(values), axis=0)
        for key, values in samples.items()
        if len(values) >= 3
    }


def _tracking_camera(
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
    coordinate_system: object,
) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("Quest camera quaternion has zero length")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform_to_y_down(transform, coordinate_system)


def _average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    translations = np.stack([value[:3, 3] for value in transforms])
    center = np.median(translations, axis=0)
    distances = np.linalg.norm(translations - center, axis=1)
    median_distance = float(np.median(distances))
    translation_keep = distances <= max(0.02, median_distance * 3.0)
    translation_selected = [
        value for value, accepted in zip(transforms, translation_keep) if accepted
    ]
    if len(translation_selected) < 3:
        translation_selected = transforms
        translation_keep = np.ones(len(transforms), dtype=bool)
    initial_rotation = Rotation.from_matrix(
        np.stack([value[:3, :3] for value in translation_selected])
    ).mean().as_matrix()
    rotation_errors = np.array(
        [
            np.degrees(
                Rotation.from_matrix(initial_rotation.T @ value[:3, :3]).magnitude()
            )
            for value in transforms
        ],
        dtype=np.float64,
    )
    rotation_median = float(np.median(rotation_errors[translation_keep]))
    rotation_mad = float(
        np.median(
            np.abs(rotation_errors[translation_keep] - rotation_median)
        )
    )
    rotation_gate = max(3.0, rotation_median + 3.0 * 1.4826 * rotation_mad)
    keep = translation_keep & (rotation_errors <= rotation_gate)
    selected = [value for value, accepted in zip(transforms, keep) if accepted]
    if len(selected) < 3:
        selected = translation_selected
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = Rotation.from_matrix(
        np.stack([value[:3, :3] for value in selected])
    ).mean().as_matrix()
    output[:3, 3] = np.median(
        np.stack([value[:3, 3] for value in selected]), axis=0
    )
    return output


def estimate_quest_world_tracking_from_rgbd_tag(
    quest_h5_path: str | Path,
    reference_points: dict[TagPointKey, np.ndarray],
    T_world_reference_camera: np.ndarray,
    *,
    max_frames: int = 80,
    max_alignment_p95_m: float = 0.03,
) -> tuple[np.ndarray, dict[str, float | int]]:
    estimates: list[np.ndarray] = []
    residuals: list[float] = []
    detector = _detector()
    with h5py.File(Path(quest_h5_path), "r") as handle:
        count = int(handle["color"].shape[0])
        indices = np.unique(
            np.linspace(0, count - 1, min(max_frames, count), dtype=np.int64)
        )
        coordinate_system = handle.attrs.get(
            "coordinate_system", handle.attrs.get("coord", "y_down")
        )
        color_format = str(handle.attrs.get("color_format", "")).lower()
        for index in indices:
            color = np.asarray(handle["color"][index])
            if color_format == "rgb8":
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            K = np.array(
                [
                    [float(handle["fx"][index]), 0.0, float(handle["cx"][index])],
                    [0.0, float(handle["fy"][index]), float(handle["cy"][index])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            quest_points = extract_tag_rgbd_points(
                color,
                np.asarray(handle["depth"][index]),
                K,
                0.001,
                detector=detector,
            )
            try:
                reference_quest = align_tag_point_maps(
                    quest_points, reference_points
                )
            except ValueError:
                continue
            if reference_quest.p95_residual_m > max_alignment_p95_m:
                continue
            tracking_camera = _tracking_camera(
                np.asarray(handle["camera_position"][index]),
                np.asarray(handle["camera_rotation_xyzw"][index]),
                coordinate_system,
            )
            world_tracking = (
                np.asarray(T_world_reference_camera, dtype=np.float64)
                @ reference_quest.transform
                @ np.linalg.inv(tracking_camera)
            )
            estimates.append(world_tracking)
            residuals.append(reference_quest.p95_residual_m)
    if len(estimates) < 3:
        raise ValueError(
            "Quest RGB-D tag alignment produced fewer than three valid poses"
        )
    result = _average_transforms(estimates)
    return result, {
        "samples": len(estimates),
        "median_tag_alignment_p95_m": float(np.median(residuals)),
        "max_tag_alignment_p95_m": float(np.max(residuals)),
    }


def estimate_quest_world_tracking_from_common_tags(
    quest_h5_path: str | Path,
    reference_camera_tags: TagPoseMap,
    T_world_reference_camera: np.ndarray,
    *,
    max_frames: int = 80,
    tag_size_m: float = 0.04,
) -> tuple[np.ndarray, dict[str, float | int]]:
    estimates: list[np.ndarray] = []
    detector = _detector()
    common_tag_counts: list[int] = []
    with h5py.File(Path(quest_h5_path), "r") as handle:
        count = int(handle["color"].shape[0])
        indices = np.unique(
            np.linspace(0, count - 1, min(max_frames, count), dtype=np.int64)
        )
        coordinate_system = handle.attrs.get(
            "coordinate_system", handle.attrs.get("coord", "y_down")
        )
        color_format = str(handle.attrs.get("color_format", "")).lower()
        for index in indices:
            color = np.asarray(handle["color"][index])
            if color_format == "rgb8":
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            K = np.array(
                [
                    [float(handle["fx"][index]), 0.0, float(handle["cx"][index])],
                    [0.0, float(handle["fy"][index]), float(handle["cy"][index])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            quest_tags = extract_tag_camera_poses(
                color,
                K,
                tag_size_m=tag_size_m,
                detector=detector,
            )
            try:
                reference_quest, quality = align_common_tag_poses(
                    quest_tags, reference_camera_tags
                )
            except ValueError:
                continue
            tracking_camera = _tracking_camera(
                np.asarray(handle["camera_position"][index]),
                np.asarray(handle["camera_rotation_xyzw"][index]),
                coordinate_system,
            )
            estimates.append(
                np.asarray(T_world_reference_camera, dtype=np.float64)
                @ reference_quest
                @ np.linalg.inv(tracking_camera)
            )
            common_tag_counts.append(int(quality["common_tags"]))
    if len(estimates) < 3:
        raise ValueError(
            "Quest common-tag alignment produced fewer than three valid poses"
        )
    result = _average_transforms(estimates)
    translations = np.stack([value[:3, 3] for value in estimates])
    translation_errors = np.linalg.norm(
        translations - result[:3, 3], axis=1
    )
    rotation_errors = np.degrees(
        Rotation.from_matrix(
            np.stack(
                [result[:3, :3].T @ value[:3, :3] for value in estimates]
            )
        ).magnitude()
    )
    return result, {
        "samples": len(estimates),
        "median_common_tags": float(np.median(common_tag_counts)),
        "translation_consistency_median_m": float(np.median(translation_errors)),
        "translation_consistency_p95_m": float(np.percentile(translation_errors, 95)),
        "rotation_consistency_median_deg": float(np.median(rotation_errors)),
        "rotation_consistency_p95_deg": float(np.percentile(rotation_errors, 95)),
    }
