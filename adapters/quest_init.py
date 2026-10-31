from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

from core.coordinate_frames import normalize_coordinate_system, transform_to_y_down
from common.tag_dictionary import cube_dictionary, validate_cube_axis_metadata


class QuestTagInitializer:
    """Robust multi-frame estimate of world<-Quest-tracking from a tag cube."""

    def __init__(
        self,
        cube_model_path: str | Path,
        min_samples: int = 20,
        reference_extrinsics_path: str | Path | None = None,
        strict_geometry: bool = False,
    ) -> None:
        data = yaml.safe_load(Path(cube_model_path).read_text(encoding="utf-8"))
        validate_cube_axis_metadata(data)
        self.tag_size_m = float(data.get("tag_size_m", 0.04))
        coordinate_system = normalize_coordinate_system(
            data.get("coordinate_system", data.get("coord", "y_down"))
        )
        self.transforms = {
            int(key): transform_to_y_down(
                np.asarray(value["T_world_tag"], dtype=np.float64).reshape(4, 4),
                coordinate_system,
            )
            for key, value in data.get("tags", {}).items()
            if isinstance(value, dict) and "T_world_tag" in value
        }
        self.reference_source = str(cube_model_path)
        if reference_extrinsics_path:
            reference_path = Path(reference_extrinsics_path)
            if reference_path.is_file():
                reference = json.loads(reference_path.read_text(encoding="utf-8"))
                reference_coordinate_system = reference.get(
                    "coordinate_system", reference.get("coord", "y_down")
                )
                reference_transforms = {
                    int(key): transform_to_y_down(
                        np.asarray(value, dtype=np.float64).reshape(4, 4),
                        reference_coordinate_system,
                    )
                    for key, value in (reference.get("T_world_tag") or {}).items()
                }
                if reference_transforms:
                    self.transforms = reference_transforms
                    self.reference_source = str(reference_path)
        if not self.transforms:
            raise ValueError("cube model contains no T_world_tag transforms")
        self.min_samples = max(3, int(min_samples))
        self.dictionary_name = str(data.get("dictionary", "DICT_APRILTAG_16h5"))
        dictionary = cube_dictionary(self.dictionary_name)
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.samples: list[np.ndarray] = []
        self.observations: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.strict_geometry = bool(strict_geometry)
        self.fit_report: dict[str, Any] = {}
        self.geometry_rejections = 0
        self._geometry_solver = None
        if self.strict_geometry:
            from standalone.tag_cube_localizer_replay_with_cube_model import CubeModel, CubeModelLocalizer
            model = CubeModel(Path(cube_model_path), self.dictionary_name,
                              float(data.get("cube_size_m", 0.0)), self.tag_size_m,
                              self.transforms, {}, {})
            self._geometry_solver = CubeModelLocalizer(model, 5.0, 12.0, 2.0)

    def _world_corners(self, tag_id: int) -> np.ndarray:
        h = self.tag_size_m / 2.0
        local = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], np.float64)
        homogeneous = np.column_stack((local, np.ones(4)))
        return (self.transforms[tag_id] @ homogeneous.T).T[:, :3]

    @staticmethod
    def _tracking_camera(meta: dict[str, Any]) -> np.ndarray:
        position = np.asarray(meta["cameraPose"]["position"], dtype=np.float64)
        x, y, z, w = np.asarray(meta["cameraPose"]["rotation"], dtype=np.float64)
        T = np.eye(4)
        T[:3, :3] = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
        ])
        T[:3, 3] = position
        return transform_to_y_down(
            T,
            meta.get("coordinate_system", meta.get("coord", "y_down")),
        )

    def add_frame(self, color_bgr: np.ndarray, meta: dict[str, Any]) -> int:
        corners, ids, _ = self.detector.detectMarkers(color_bgr)
        if ids is None:
            return self.progress
        object_points, image_points = [], []
        for corners_i, tag_id_value in zip(corners, ids.reshape(-1)):
            tag_id = int(tag_id_value)
            if tag_id not in self.transforms:
                continue
            object_points.append(self._world_corners(tag_id))
            image_points.append(np.asarray(corners_i, np.float64).reshape(4, 2))
        if not object_points:
            return self.progress
        intr = meta["intrinsics"]
        K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64)
        if self.strict_geometry:
            points, pixels = np.concatenate(object_points), np.concatenate(image_points)
            solve = (self._geometry_solver._solve_single_tag if len(object_points) == 1
                     else self._geometry_solver._solve_multiple_tags)
            candidate = solve(points, pixels, K, np.zeros(5))
            ok = candidate is not None
            if ok:
                rvec, tvec = candidate.rvec, candidate.tvec
                rotation, _ = cv2.Rodrigues(rvec)
                xyz = points @ rotation.T + tvec.reshape(3)
                ok = (np.isfinite(xyz).all() and np.all(xyz[:, 2] > 0)
                      and np.isfinite(candidate.quality.rms_px)
                      and candidate.quality.rms_px <= 5.0 and candidate.quality.max_px <= 12.0)
            if not ok:
                self.geometry_rejections += 1
        else:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                np.concatenate(object_points), np.concatenate(image_points), K, np.zeros(5),
                flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=3.0, iterationsCount=100,
            )
        if not ok:
            return self.progress
        camera_world = np.eye(4)
        camera_world[:3, :3], _ = cv2.Rodrigues(rvec)
        camera_world[:3, 3] = tvec.reshape(3)
        # PnP and the adapter-normalized tracking pose are both Y-down.
        world_tracking = np.linalg.inv(camera_world) @ np.linalg.inv(
            self._tracking_camera(meta)
        )
        if np.isfinite(world_tracking).all():
            self.samples.append(world_tracking)
            self.observations.append(
                (
                    np.concatenate(object_points),
                    np.concatenate(image_points),
                    K,
                    self._tracking_camera(meta),
                )
            )
        return self.progress

    @property
    def progress(self) -> int:
        return min(99, int(100 * len(self.samples) / self.min_samples))

    def finish(self) -> np.ndarray | None:
        if len(self.samples) < self.min_samples:
            return None
        translations = np.stack([sample[:3, 3] for sample in self.samples])
        center = np.median(translations, axis=0)
        distances = np.linalg.norm(translations - center, axis=1)
        keep = distances <= max(float(np.median(distances)) * 3.0, 0.01)
        selected = [sample for sample, accepted in zip(self.samples, keep) if accepted]
        fit_observations = self.observations
        if self.strict_geometry:
            fit_observations = [o for o, accepted in zip(self.observations, keep) if accepted]
            self.fit_report = {"accepted_geometry_frames": len(self.samples),
                               "fit_frames": len(selected), "pose_outlier_frames": int((~keep).sum()),
                               "geometry_rejections": self.geometry_rejections}
            if len(selected) < self.min_samples:
                self.fit_report["failure"] = "insufficient frames after robust pose filtering"
                return None
        quaternions = []
        for sample in selected:
            rvec, _ = cv2.Rodrigues(sample[:3, :3])
            angle = float(np.linalg.norm(rvec))
            axis = rvec.reshape(3) / angle if angle > 1e-9 else np.array([1.0, 0.0, 0.0])
            quaternions.append(np.r_[axis * np.sin(angle / 2), np.cos(angle / 2)])
        reference = quaternions[0]
        quaternions = [q if np.dot(q, reference) >= 0 else -q for q in quaternions]
        eigenvalues, eigenvectors = np.linalg.eigh(sum(np.outer(q, q) for q in quaternions))
        x, y, z, w = eigenvectors[:, int(np.argmax(eigenvalues))]
        result = np.eye(4)
        result[:3, :3] = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
        ])
        result[:3, 3] = np.median(np.stack([sample[:3, 3] for sample in selected]), axis=0)
        initial_rvec, _ = cv2.Rodrigues(result[:3, :3])
        initial = np.r_[initial_rvec.reshape(3), result[:3, 3]]

        def residual(parameters: np.ndarray) -> np.ndarray:
            world_tracking = np.eye(4)
            world_tracking[:3, :3], _ = cv2.Rodrigues(parameters[:3])
            world_tracking[:3, 3] = parameters[3:]
            errors = []
            for points_world, pixels, K, tracking_camera in fit_observations:
                camera_world = np.linalg.inv(world_tracking @ tracking_camera)
                rvec, _ = cv2.Rodrigues(camera_world[:3, :3])
                projected, _ = cv2.projectPoints(
                    points_world, rvec, camera_world[:3, 3], K, np.zeros(5)
                )
                errors.append((projected.reshape(-1, 2) - pixels).reshape(-1))
            return np.concatenate(errors)

        optimized = least_squares(
            residual,
            initial,
            loss="soft_l1",
            f_scale=2.0,
            max_nfev=100,
        )
        if optimized.success and np.isfinite(optimized.x).all():
            result[:3, :3], _ = cv2.Rodrigues(optimized.x[:3])
            result[:3, 3] = optimized.x[3:]
        elif self.strict_geometry:
            self.fit_report["failure"] = "least-squares did not converge to finite parameters"
            return None
        return result


def save_quest_extrinsic(
    path: str | Path,
    transform: np.ndarray,
    samples: int,
    *,
    quality: dict[str, Any] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "coordinate_system": "y_down",
        "convention": "column vectors; X_world = T_world_tracking @ X_tracking",
        "T_world_tracking": np.asarray(transform).tolist(),
        "samples": samples,
    }
    if quality is not None:
        payload["quality"] = quality
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
