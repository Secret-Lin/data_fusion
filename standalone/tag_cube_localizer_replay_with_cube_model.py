#!/usr/bin/env python3
"""Replay HDF5 camera recordings and localize cameras with cube_model.yaml.

The script is a cube-model-based replacement for the previous standalone
replay calibrator. It keeps the original session layout and the main JSON
output fields, but no longer hard-codes tag IDs to cube faces and no longer
searches cyclic corner rolls.

Transform convention (column vectors):
    X_A = T_A_B @ X_B_h

The cube model provides T_world_tag for IDs 0..5. For every accepted frame the
script jointly solves T_cam_world from all visible tag corners, then returns:
    T_world_cam = inv(T_cam_world)

Expected replay layout:
    session_dir/
        camera_metadata.json
        rgbd/
            CAMERA_SERIAL_1.h5
            CAMERA_SERIAL_2.h5

Each HDF5 file must contain a color dataset with shape [N, H, W, C].
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import yaml

from core.coordinate_frames import normalize_coordinate_system, transform_to_y_down
from common.tag_dictionary import cube_dictionary, validate_cube_axis_metadata


SUPPORTED_TAG_IDS = set(range(6))


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))[0].ravel()


def rotvec_to_matrix(rvec: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]


def se3_exp_rot(delta: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(delta))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = np.asarray(delta, dtype=np.float64).reshape(3) / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) * K
        + (1.0 - math.cos(theta)) * (K @ K)
    )


def average_rotation(
    rotations: list[np.ndarray], weights: Optional[np.ndarray] = None
) -> np.ndarray:
    if not rotations:
        raise ValueError("average_rotation requires at least one rotation")
    estimate = np.asarray(rotations[0], dtype=np.float64).reshape(3, 3).copy()
    if weights is None:
        normalized_weights = np.ones(len(rotations), dtype=np.float64)
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(normalized_weights) != len(rotations):
            raise ValueError("weights length must match rotations length")
        normalized_weights = np.maximum(normalized_weights, 1e-12)
    normalized_weights /= np.sum(normalized_weights)

    for _ in range(50):
        delta = np.zeros(3, dtype=np.float64)
        for weight, rotation in zip(normalized_weights, rotations):
            delta += weight * matrix_to_rotvec(
                np.asarray(rotation, dtype=np.float64).reshape(3, 3) @ estimate.T
            )
        if np.linalg.norm(delta) < 1e-12:
            break
        estimate = se3_exp_rot(delta) @ estimate
    return estimate


def average_translation(
    translations: list[np.ndarray], weights: Optional[np.ndarray] = None
) -> np.ndarray:
    array = np.vstack(
        [np.asarray(translation, dtype=np.float64).reshape(1, 3) for translation in translations]
    )
    if weights is None:
        return np.mean(array, axis=0)
    normalized_weights = np.maximum(
        np.asarray(weights, dtype=np.float64).reshape(-1), 1e-12
    )
    normalized_weights /= np.sum(normalized_weights)
    return np.sum(array * normalized_weights[:, None], axis=0)


def validate_rigid_transform(T: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise RuntimeError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise RuntimeError(f"{name} has an invalid homogeneous last row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(R), 1.0, atol=1e-5
    ):
        raise RuntimeError(f"{name} rotation is not a proper rotation matrix")
    return T


@dataclass(frozen=True)
class CubeModel:
    source_path: Path
    dictionary: str
    cube_size_m: float
    tag_size_m: float
    T_world_tag: dict[int, np.ndarray]
    faces: dict[int, str]
    coordinate_convention: dict

    def local_tag_corners(self) -> np.ndarray:
        half = self.tag_size_m / 2.0
        # OpenCV ArUco corner order: top-left, top-right, bottom-right,
        # bottom-left. Models are normalized to the project's Y-down basis.
        return np.array(
            [
                [-half, -half, 0.0],
                [+half, -half, 0.0],
                [+half, +half, 0.0],
                [-half, +half, 0.0],
            ],
            dtype=np.float64,
        )

    def world_tag_corners(self, tag_id: int) -> np.ndarray:
        local = self.local_tag_corners()
        homogeneous = np.column_stack((local, np.ones(4, dtype=np.float64)))
        return (self.T_world_tag[tag_id] @ homogeneous.T).T[:, :3]


def load_cube_model(path: Path) -> CubeModel:
    if not path.exists():
        raise RuntimeError(f"cube model not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse cube model {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("cube model root must be a YAML object")
    dictionary = str(data.get("dictionary", ""))
    cube_dictionary(dictionary)
    validate_cube_axis_metadata(data)
    cube_size_m = float(data.get("cube_size_m", 0.0))
    tag_size_m = float(data.get("tag_size_m", 0.0))
    if cube_size_m <= 0.0 or tag_size_m <= 0.0:
        raise RuntimeError("cube_size_m and tag_size_m must be positive")
    if tag_size_m > cube_size_m + 1e-12:
        raise RuntimeError("tag_size_m cannot exceed cube_size_m")

    tag_entries = data.get("tags")
    if not isinstance(tag_entries, dict):
        raise RuntimeError("cube model must contain a tags mapping")
    transforms: dict[int, np.ndarray] = {}
    faces: dict[int, str] = {}
    coordinate_system = normalize_coordinate_system(
        data.get("coordinate_system", data.get("coord", "y_down"))
    )
    for key, entry in tag_entries.items():
        tag_id = int(key)
        if tag_id not in SUPPORTED_TAG_IDS or not isinstance(entry, dict):
            continue
        if "T_world_tag" not in entry:
            raise RuntimeError(f"Tag {tag_id} has no T_world_tag")
        transforms[tag_id] = transform_to_y_down(
            validate_rigid_transform(
                np.asarray(entry["T_world_tag"], dtype=np.float64),
                f"tags[{tag_id}].T_world_tag",
            ),
            coordinate_system,
        )
        faces[tag_id] = str(entry.get("face", "unknown"))
    missing = sorted(SUPPORTED_TAG_IDS - set(transforms))
    if missing:
        raise RuntimeError(f"cube model is missing Tag IDs: {missing}")

    # Geometric sanity check: all tag centres should be cube_size/2 from origin.
    expected_radius = cube_size_m / 2.0
    for tag_id, T in transforms.items():
        radius = float(np.linalg.norm(T[:3, 3]))
        if abs(radius - expected_radius) > max(0.001, cube_size_m * 0.03):
            raise RuntimeError(
                f"Tag {tag_id} centre radius is {radius:.6f} m, expected "
                f"approximately {expected_radius:.6f} m"
            )
    return CubeModel(
        source_path=path,
        dictionary=dictionary,
        cube_size_m=cube_size_m,
        tag_size_m=tag_size_m,
        T_world_tag=transforms,
        faces=faces,
        coordinate_convention={
            **dict(data.get("coordinate_convention", {})),
            "source_coordinate_system": coordinate_system,
            "internal_coordinate_system": "y_down",
        },
    )


@dataclass
class PoseQuality:
    mean_px: float
    rms_px: float
    max_px: float
    n_points: int
    n_inliers: int


@dataclass
class Observation:
    T_cam_world: np.ndarray
    quality: PoseQuality
    tag_ids: list[int]
    frame_index: int


@dataclass
class CameraStats:
    processed: int = 0
    detected_any_tag: int = 0
    detected_cube_tag: int = 0
    accepted: int = 0
    reject_no_tag: int = 0
    reject_tag_count: int = 0
    reject_pnp: int = 0
    reject_reprojection: int = 0
    rejected_pose_outlier: int = 0

    def asdict(self) -> dict[str, int]:
        return {
            "processed": self.processed,
            "detected_any_tag": self.detected_any_tag,
            "detected_cube_tag": self.detected_cube_tag,
            "detected_valid_cube_tag": self.detected_cube_tag,
            "accepted": self.accepted,
            "reject_no_tag": self.reject_no_tag,
            "reject_tag_count": self.reject_tag_count,
            "reject_faces": self.reject_tag_count,
            "reject_pnp": self.reject_pnp,
            "reject_reprojection": self.reject_reprojection,
            "reject_reproj": self.reject_reprojection,
            "rejected_pose_outlier": self.rejected_pose_outlier,
        }


@dataclass
class SolveCandidate:
    rvec: np.ndarray
    tvec: np.ndarray
    quality: PoseQuality


class CubeModelLocalizer:
    def __init__(
        self,
        model: CubeModel,
        max_reproj_error_px: float,
        max_point_error_px: float,
        ransac_reproj_error_px: float,
        min_tags: int = 1,
        min_observations: int = 3,
        pose_outlier_rot_deg: float = 5.0,
        pose_outlier_trans_m: float = 0.03,
    ):
        self.model = model
        self.max_reproj_error_px = float(max_reproj_error_px)
        self.max_point_error_px = float(max_point_error_px)
        self.ransac_reproj_error_px = float(ransac_reproj_error_px)
        self.min_tags = max(1, int(min_tags))
        self.min_observations = max(1, int(min_observations))
        self.pose_outlier_rot_deg = float(pose_outlier_rot_deg)
        self.pose_outlier_trans_m = float(pose_outlier_trans_m)
        self.observations: dict[str, list[Observation]] = defaultdict(list)
        self.stats: dict[str, CameraStats] = defaultdict(CameraStats)

    def _frame_points(
        self, corners_per_id: dict[int, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        object_blocks = []
        image_blocks = []
        used_ids = []
        for tag_id in sorted(corners_per_id):
            if tag_id not in self.model.T_world_tag:
                continue
            object_blocks.append(self.model.world_tag_corners(tag_id))
            image_blocks.append(
                np.asarray(corners_per_id[tag_id], dtype=np.float64).reshape(4, 2)
            )
            used_ids.append(tag_id)
        if not object_blocks:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                [],
            )
        return np.vstack(object_blocks), np.vstack(image_blocks), used_ids

    @staticmethod
    def _reprojection_errors(
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> np.ndarray:
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
        return np.linalg.norm(
            projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1
        )

    def _make_candidate(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        n_inliers: int,
    ) -> Optional[SolveCandidate]:
        R = rotvec_to_matrix(rvec)
        camera_points = (R @ object_points.T + np.asarray(tvec).reshape(3, 1)).T
        if np.any(camera_points[:, 2] <= 0.0):
            return None
        errors = self._reprojection_errors(
            object_points, image_points, rvec, tvec, K, dist
        )
        if not np.all(np.isfinite(errors)):
            return None
        quality = PoseQuality(
            mean_px=float(np.mean(errors)),
            rms_px=float(np.sqrt(np.mean(errors * errors))),
            max_px=float(np.max(errors)),
            n_points=int(len(object_points)),
            n_inliers=int(n_inliers),
        )
        return SolveCandidate(rvec=rvec, tvec=tvec, quality=quality)

    def _solve_single_tag(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> Optional[SolveCandidate]:
        # IPPE explicitly handles the two-solution ambiguity of planar points.
        try:
            result = cv2.solvePnPGeneric(
                object_points,
                image_points,
                K,
                dist,
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error:
            return None
        if not result or not bool(result[0]):
            return None
        candidates = []
        for rvec, tvec in zip(result[1], result[2]):
            candidate = self._make_candidate(
                object_points,
                image_points,
                rvec,
                tvec,
                K,
                dist,
                n_inliers=4,
            )
            if candidate is not None:
                candidates.append(candidate)
        # IPPE can return only the mirrored planar solution when the model's
        # tag +Y axis (up) is observed by an OpenCV Y-down camera. SQPnP and
        # ITERATIVE provide the physically valid pi-about-X candidate in that
        # fronto-parallel case. Keep all valid candidates and let measured
        # reprojection error select the solution.
        for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE):
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    K,
                    dist,
                    flags=flag,
                )
            except cv2.error:
                continue
            if not ok:
                continue
            candidate = self._make_candidate(
                object_points,
                image_points,
                rvec,
                tvec,
                K,
                dist,
                n_inliers=4,
            )
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return None
        return min(candidates, key=lambda candidate: candidate.quality.rms_px)

    def _solve_multiple_tags(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> Optional[SolveCandidate]:
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                K,
                dist,
                flags=cv2.SOLVEPNP_SQPNP,
                confidence=0.999,
                reprojectionError=self.ransac_reproj_error_px,
                iterationsCount=300,
            )
        except cv2.error:
            try:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_points,
                    image_points,
                    K,
                    dist,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    confidence=0.999,
                    reprojectionError=self.ransac_reproj_error_px,
                    iterationsCount=300,
                )
            except cv2.error:
                return None
        if not ok or rvec is None or tvec is None:
            return None
        # Refine using all observed corners after the robust initialization.
        try:
            refined, refined_rvec, refined_tvec = cv2.solvePnP(
                object_points,
                image_points,
                K,
                dist,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if refined:
                rvec, tvec = refined_rvec, refined_tvec
        except cv2.error:
            pass
        return self._make_candidate(
            object_points,
            image_points,
            rvec,
            tvec,
            K,
            dist,
            n_inliers=0 if inliers is None else len(inliers),
        )

    def add_processed_frame(self, serial: str) -> None:
        self.stats[serial].processed += 1

    def add_no_tag_frame(self, serial: str) -> None:
        self.stats[serial].reject_no_tag += 1

    def add_frame(
        self,
        serial: str,
        frame_index: int,
        corners_per_id: dict[int, np.ndarray],
        K: np.ndarray,
        dist: np.ndarray,
    ) -> bool:
        stats = self.stats[serial]
        if corners_per_id:
            stats.detected_cube_tag += 1
        object_points, image_points, tag_ids = self._frame_points(corners_per_id)
        if len(tag_ids) < self.min_tags:
            stats.reject_tag_count += 1
            return False
        if len(tag_ids) == 1:
            candidate = self._solve_single_tag(
                object_points, image_points, K, dist
            )
        else:
            candidate = self._solve_multiple_tags(
                object_points, image_points, K, dist
            )
        if candidate is None:
            stats.reject_pnp += 1
            return False
        if (
            candidate.quality.rms_px > self.max_reproj_error_px
            or candidate.quality.max_px > self.max_point_error_px
        ):
            stats.reject_reprojection += 1
            return False
        self.observations[serial].append(
            Observation(
                T_cam_world=make_T(
                    rotvec_to_matrix(candidate.rvec), candidate.tvec.ravel()
                ),
                quality=candidate.quality,
                tag_ids=tag_ids,
                frame_index=int(frame_index),
            )
        )
        stats.accepted += 1
        return True

    def _pose_outlier_mask(self, observations: list[Observation]) -> np.ndarray:
        count = len(observations)
        if count < 4:
            return np.ones(count, dtype=bool)
        rotations = [observation.T_cam_world[:3, :3] for observation in observations]
        translations = [observation.T_cam_world[:3, 3] for observation in observations]
        reprojection = np.array(
            [max(observation.quality.rms_px, 1e-3) for observation in observations],
            dtype=np.float64,
        )
        weights = 1.0 / reprojection
        mean_rotation = average_rotation(rotations, weights=weights)
        mean_translation = average_translation(translations, weights=weights)
        rotation_errors = np.array(
            [
                np.linalg.norm(matrix_to_rotvec(rotation @ mean_rotation.T))
                * 180.0
                / math.pi
                for rotation in rotations
            ],
            dtype=np.float64,
        )
        translation_errors = np.array(
            [np.linalg.norm(translation - mean_translation) for translation in translations],
            dtype=np.float64,
        )
        rotation_median = float(np.median(rotation_errors))
        translation_median = float(np.median(translation_errors))
        rotation_mad = float(np.median(np.abs(rotation_errors - rotation_median)))
        translation_mad = float(
            np.median(np.abs(translation_errors - translation_median))
        )
        rotation_gate = min(
            self.pose_outlier_rot_deg,
            rotation_median + max(3.0 * 1.4826 * rotation_mad, 1.0),
        )
        translation_gate = min(
            self.pose_outlier_trans_m,
            translation_median + max(3.0 * 1.4826 * translation_mad, 0.005),
        )
        mask = (rotation_errors <= rotation_gate) & (
            translation_errors <= translation_gate
        )
        if int(np.count_nonzero(mask)) < max(
            2, min(self.min_observations, count)
        ):
            return np.ones(count, dtype=bool)
        return mask

    def estimate_all(self) -> dict[str, Optional[dict]]:
        output: dict[str, Optional[dict]] = {}
        serials = sorted(set(self.stats) | set(self.observations))
        for serial in serials:
            all_observations = self.observations.get(serial, [])
            if len(all_observations) < self.min_observations:
                output[serial] = None
                continue
            mask = self._pose_outlier_mask(all_observations)
            used = [
                observation
                for observation, keep in zip(all_observations, mask)
                if bool(keep)
            ]
            rejected = len(all_observations) - len(used)
            self.stats[serial].rejected_pose_outlier += rejected
            if len(used) < self.min_observations:
                output[serial] = None
                continue
            rms = np.array(
                [max(observation.quality.rms_px, 1e-3) for observation in used],
                dtype=np.float64,
            )
            weights = 1.0 / rms
            mean_rotation = average_rotation(
                [observation.T_cam_world[:3, :3] for observation in used],
                weights=weights,
            )
            mean_translation = average_translation(
                [observation.T_cam_world[:3, 3] for observation in used],
                weights=weights,
            )
            T_cam_world = make_T(mean_rotation, mean_translation)
            output[serial] = {
                "T_cam_world": T_cam_world,
                "T_world_cam": invert_T(T_cam_world),
                "n_observations_total": len(all_observations),
                "n_observations_used": len(used),
                "n_observations_pose_outliers": rejected,
                "quality": {
                    "mean_rms_reproj_px": float(
                        np.mean([observation.quality.rms_px for observation in used])
                    ),
                    "median_rms_reproj_px": float(
                        np.median([observation.quality.rms_px for observation in used])
                    ),
                    "max_rms_reproj_px": float(
                        np.max([observation.quality.rms_px for observation in used])
                    ),
                    "max_point_reproj_px": float(
                        np.max([observation.quality.max_px for observation in used])
                    ),
                    "mean_tags_per_frame": float(
                        np.mean([len(observation.tag_ids) for observation in used])
                    ),
                },
                "tag_observation_counts": {
                    str(tag_id): sum(tag_id in observation.tag_ids for observation in used)
                    for tag_id in sorted(SUPPORTED_TAG_IDS)
                },
            }
        return output


def load_camera_metadata(metadata_json: Path) -> dict[str, dict]:
    if not metadata_json.exists():
        raise RuntimeError(f"camera metadata not found: {metadata_json}")
    try:
        data = json.loads(metadata_json.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse camera metadata {metadata_json}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("camera metadata root must be a JSON object")
    return data


def make_guessed_intrinsics(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    focal = float(max(width, height)) * 0.9
    K = np.array(
        [
            [focal, 0.0, float(width) * 0.5],
            [0.0, focal, float(height) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K, np.zeros(5, dtype=np.float64)


def extract_intrinsics(
    metadata: dict[str, dict],
    serial: str,
    width: int,
    height: int,
    allow_guessed_intrinsics: bool,
) -> tuple[np.ndarray, np.ndarray]:
    entry = metadata.get(serial, {})
    K_value = entry.get("K") if isinstance(entry, dict) else None
    dist_value = entry.get("dist") if isinstance(entry, dict) else None
    if K_value is None:
        if allow_guessed_intrinsics:
            print(f"[WARNING] {serial}: missing K, using guessed intrinsics")
            return make_guessed_intrinsics(width, height)
        raise RuntimeError(
            f"missing camera intrinsics for serial={serial}. Provide metadata or "
            "use --allow-guessed-intrinsics for debugging only."
        )
    try:
        K = np.asarray(K_value, dtype=np.float64).reshape(3, 3)
        dist = (
            np.asarray(dist_value, dtype=np.float64).reshape(-1)
            if dist_value is not None
            else np.zeros(5, dtype=np.float64)
        )
    except Exception as exc:
        raise RuntimeError(f"invalid intrinsics for serial={serial}: {exc}") from exc
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0 or not np.isclose(K[2, 2], 1.0):
        raise RuntimeError(f"invalid K for serial={serial}: {K}")
    return K, dist


def detect_markers(gray: np.ndarray, dictionary, parameters):
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


def make_aruco_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()
    try:
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMaxIterations = 50
        parameters.cornerRefinementMinAccuracy = 0.01
    except Exception:
        pass
    return parameters


def convert_to_gray(frame: np.ndarray, color_order: str) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"unsupported image shape: {frame.shape}")
    conversion = cv2.COLOR_RGB2GRAY if color_order == "rgb" else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(frame[:, :, :3], conversion)


def build_output(
    args: argparse.Namespace,
    model: CubeModel,
    localizer: CubeModelLocalizer,
    estimates: dict[str, Optional[dict]],
    session_dir: Path,
    metadata_json: Path,
    duration_sec: float,
) -> dict:
    cameras: dict[str, dict] = {}
    for serial, estimate in estimates.items():
        if estimate is None:
            cameras[serial] = {
                "valid": False,
                "T_cam_world": None,
                "T_world_cam": None,
                # Compatibility aliases because the Cube frame is world.
                "T_cam_cube": None,
                "T_cube_cam": None,
                "note": "Not enough valid observations after filtering",
            }
            continue
        T_cam_world = np.asarray(estimate["T_cam_world"], dtype=np.float64)
        T_world_cam = np.asarray(estimate["T_world_cam"], dtype=np.float64)
        cameras[serial] = {
            "valid": True,
            "T_cam_world": T_cam_world.tolist(),
            "T_world_cam": T_world_cam.tolist(),
            "T_cam_cube": T_cam_world.tolist(),
            "T_cube_cam": T_world_cam.tolist(),
            "world_frame_definition": model.coordinate_convention,
            "n_observations_total": estimate["n_observations_total"],
            "n_observations_used": estimate["n_observations_used"],
            "n_observations_pose_outliers": estimate[
                "n_observations_pose_outliers"
            ],
            "quality": estimate["quality"],
            "tag_observation_counts": estimate["tag_observation_counts"],
        }

    valid_serials = [
        serial for serial in sorted(cameras) if cameras[serial].get("valid", False)
    ]
    requested_reference = str(args.reference_serial).strip()
    if requested_reference:
        if requested_reference not in valid_serials:
            raise RuntimeError(
                f"--reference-serial {requested_reference!r} is invalid. "
                f"Valid cameras: {valid_serials}"
            )
        reference_serial = requested_reference
    else:
        reference_serial = valid_serials[0] if valid_serials else ""

    T_ref_cam: dict[str, Optional[list]] = {}
    T_cam_ref: dict[str, Optional[list]] = {}
    pairwise: dict[str, dict[str, list]] = {}
    if reference_serial:
        T_world_ref = np.asarray(cameras[reference_serial]["T_world_cam"])
        T_ref_world = invert_T(T_world_ref)
        for serial, camera in cameras.items():
            if not camera.get("valid", False):
                T_ref_cam[serial] = None
                T_cam_ref[serial] = None
                continue
            T_world_camera = np.asarray(camera["T_world_cam"])
            T_reference_camera = T_ref_world @ T_world_camera
            T_ref_cam[serial] = T_reference_camera.tolist()
            T_cam_ref[serial] = invert_T(T_reference_camera).tolist()
            camera["T_ref_cam"] = T_ref_cam[serial]
            camera["T_cam_ref"] = T_cam_ref[serial]
        for target in valid_serials:
            pairwise[target] = {}
            T_world_target = np.asarray(cameras[target]["T_world_cam"])
            T_target_world = invert_T(T_world_target)
            for source in valid_serials:
                T_world_source = np.asarray(cameras[source]["T_world_cam"])
                pairwise[target][source] = (
                    T_target_world @ T_world_source
                ).tolist()

    return {
        "script": "tag_cube_localizer_replay_with_cube_model.py",
        "coordinate_system": "y_down",
        "convention": "column vectors; X_target = T_target_source @ X_source_h",
        "unit": "meter",
        "cube_model": str(model.source_path),
        "cube_model_dictionary": model.dictionary,
        "cube_size_m": model.cube_size_m,
        "tag_size_m": model.tag_size_m,
        "cube_size_mm": model.cube_size_m * 1000.0,
        "tag_size_mm": model.tag_size_m * 1000.0,
        "tag_family": model.dictionary,
        "tag_faces": {str(key): value for key, value in sorted(model.faces.items())},
        "reference_serial": reference_serial,
        "T_ref_cam": T_ref_cam,
        "T_cam_ref": T_cam_ref,
        "pairwise_T_target_source": pairwise,
        "session_dir": str(session_dir),
        "metadata_json": str(metadata_json),
        "color_order": args.color_order,
        "min_tags": max(1, int(args.min_tags)),
        "min_faces": max(1, int(args.min_tags)),
        "min_observations": max(1, int(args.min_observations)),
        "max_reproj_error_px_rms": float(args.max_reproj_error_px),
        "max_point_error_px": float(args.max_point_error_px),
        "ransac_reproj_error_px": float(args.ransac_reproj_error_px),
        "stats": {
            serial: stats.asdict() for serial, stats in sorted(localizer.stats.items())
        },
        "cameras": cameras,
        "T_world_cam": {
            serial: camera["T_world_cam"] for serial, camera in cameras.items()
        },
        "duration_sec": float(duration_sec),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay AprilTag cube observations using cube_model.yaml"
    )
    parser.add_argument(
        "--session-dir", required=True, help="Replay directory containing rgbd/*.h5"
    )
    parser.add_argument(
        "--cube-model", required=True, help="cube_model.yaml generated by cube calibration"
    )
    parser.add_argument(
        "--metadata-json",
        default="",
        help="Camera metadata JSON. Default: session_dir/camera_metadata.json",
    )
    parser.add_argument(
        "--max-reproj-error-px", type=float, default=5.0
    )
    parser.add_argument("--max-point-error-px", type=float, default=12.0)
    parser.add_argument("--ransac-reproj-error-px", type=float, default=2.0)
    parser.add_argument(
        "--min-tags",
        "--min-faces",
        dest="min_tags",
        type=int,
        default=1,
        help="Minimum visible cube tags in an accepted frame. Default: 1",
    )
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--reference-serial", default="")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Per-camera limit. 0 means all"
    )
    parser.add_argument("--color-order", choices=["rgb", "bgr"], default="bgr")
    parser.add_argument("--allow-guessed-intrinsics", action="store_true")
    parser.add_argument("--pose-outlier-rot-deg", type=float, default=5.0)
    parser.add_argument("--pose-outlier-trans-m", type=float, default=0.03)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--out",
        default="",
        help="Default: session_dir/tag_cube_extrinsics_cube_model.json",
    )
    args = parser.parse_args()

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is unavailable. Install opencv-contrib-python")

    session_dir = Path(args.session_dir)
    rgbd_dir = session_dir / "rgbd"
    if not rgbd_dir.is_dir():
        raise RuntimeError(f"rgbd directory not found: {rgbd_dir}")
    h5_files = sorted(rgbd_dir.glob("*.h5"))
    if not h5_files:
        raise RuntimeError(f"no HDF5 files found under {rgbd_dir}")
    model = load_cube_model(Path(args.cube_model))
    metadata_json = (
        Path(args.metadata_json)
        if str(args.metadata_json).strip()
        else session_dir / "camera_metadata.json"
    )
    metadata = load_camera_metadata(metadata_json)
    output_json = (
        Path(args.out)
        if str(args.out).strip()
        else session_dir / "tag_cube_extrinsics_cube_model.json"
    )

    localizer = CubeModelLocalizer(
        model=model,
        max_reproj_error_px=args.max_reproj_error_px,
        max_point_error_px=args.max_point_error_px,
        ransac_reproj_error_px=args.ransac_reproj_error_px,
        min_tags=args.min_tags,
        min_observations=args.min_observations,
        pose_outlier_rot_deg=args.pose_outlier_rot_deg,
        pose_outlier_trans_m=args.pose_outlier_trans_m,
    )
    dictionary = cube_dictionary(model.dictionary)
    parameters = make_aruco_parameters()
    start_time = time.time()

    print(
        f"[i] Cube model: {model.source_path} | cube={model.cube_size_m*1000:.1f} mm "
        f"| tag={model.tag_size_m*1000:.1f} mm"
    )
    print(f"[i] Face mapping: {model.faces}")

    for h5_path in h5_files:
        serial = h5_path.stem
        with h5py.File(str(h5_path), "r") as file:
            if "color" not in file:
                print(f"[!] {serial}: no color dataset, skipped")
                continue
            color = file["color"]
            if color.ndim != 4 or color.shape[0] <= 0:
                print(f"[!] {serial}: invalid color shape {color.shape}, skipped")
                continue
            height, width = int(color.shape[1]), int(color.shape[2])
            K, dist = extract_intrinsics(
                metadata,
                serial,
                width,
                height,
                bool(args.allow_guessed_intrinsics),
            )
            frame_count = int(color.shape[0])
            stride = max(1, int(args.stride))
            processed_for_camera = 0
            print(
                f"[i] {serial}: frames={frame_count}, size={width}x{height}, "
                f"stride={stride}"
            )
            for frame_index in range(0, frame_count, stride):
                if args.max_frames > 0 and processed_for_camera >= args.max_frames:
                    break
                processed_for_camera += 1
                localizer.add_processed_frame(serial)
                frame = np.asarray(color[frame_index])
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                gray = convert_to_gray(frame, args.color_order)
                corners, ids, _ = detect_markers(gray, dictionary, parameters)
                if ids is None or len(ids) == 0:
                    localizer.add_no_tag_frame(serial)
                    continue
                localizer.stats[serial].detected_any_tag += 1
                corners_per_id: dict[int, np.ndarray] = {}
                for marker_corners, marker_id_array in zip(corners, ids):
                    tag_id = int(marker_id_array[0])
                    if tag_id in model.T_world_tag:
                        corners_per_id[tag_id] = np.asarray(
                            marker_corners, dtype=np.float64
                        ).reshape(4, 2)
                accepted = localizer.add_frame(
                    serial,
                    frame_index,
                    corners_per_id,
                    K,
                    dist,
                )

                if args.show:
                    display = frame.copy()
                    if args.color_order == "rgb":
                        display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
                    try:
                        cv2.aruco.drawDetectedMarkers(display, corners, ids)
                    except Exception:
                        pass
                    status = localizer.stats[serial]
                    text = (
                        f"{serial} frame {frame_index+1}/{frame_count} "
                        f"tags={sorted(corners_per_id)} accepted={status.accepted} "
                        f"current={'OK' if accepted else 'REJECT'}"
                    )
                    cv2.putText(
                        display,
                        text,
                        (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(f"cube-model::{serial}", display)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break

    if args.show:
        cv2.destroyAllWindows()

    estimates = localizer.estimate_all()
    output = build_output(
        args,
        model,
        localizer,
        estimates,
        session_dir,
        metadata_json,
        time.time() - start_time,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(f"Wrote: {output_json}")
    if output["reference_serial"]:
        print(f"[i] Reference camera: {output['reference_serial']}")
        print("[i] X_ref = T_ref_cam[serial] @ X_cam")
    for serial, camera in sorted(output["cameras"].items()):
        stats = output["stats"].get(serial, {})
        if camera["valid"]:
            quality = camera["quality"]
            print(
                f"[OK] {serial}: used={camera['n_observations_used']}/"
                f"{camera['n_observations_total']}, "
                f"rms_mean={quality['mean_rms_reproj_px']:.3f}px, "
                f"max_point={quality['max_point_reproj_px']:.3f}px, "
                f"mean_tags={quality['mean_tags_per_frame']:.2f}, "
                f"accepted={stats.get('accepted', 0)}"
            )
        else:
            print(f"[FAIL] {serial}: {camera['note']}; stats={stats}")


if __name__ == "__main__":
    main()
