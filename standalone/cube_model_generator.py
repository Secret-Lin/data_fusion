#!/usr/bin/env python3
"""Interactive calibration and camera localization for a six-AprilTag cube.

Cube: 50 mm edge. Tags: selectable DICT_APRILTAG_16h5 or DICT_4X4_50,
IDs 0..5, 40 mm square.

Commands:
    python cube_model_generator.py calibrate --dictionary DICT_APRILTAG_16h5 --output cube_model.yaml
    python cube_model_generator.py calibrate --dictionary DICT_4X4_50 --output cube_model.yaml
    python cube_model_generator.py track --dictionary DICT_4X4_50 --config cube_model.yaml
    python cube_model_generator.py self-test

Coordinate convention:
    T_A_B maps coordinates from frame B to frame A.
    The cube/world origin is the cube center. Tag 0 is the +Z face. World +X
    follows Tag 0's right direction, +Y follows Tag 0's top direction, and +Z
    is Tag 0's outward normal. Units in the YAML file are metres.
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

try:
    import pyrealsense2 as rs
except ImportError:  # Allow self-test without a RealSense installation.
    rs = None


TAG_IDS = tuple(range(6))
DEFAULT_CUBE_SIZE = 0.050
DEFAULT_TAG_SIZE = 0.040
WINDOW_NAME = "AprilTag cube localizer"
DEFAULT_DICTIONARY = "DICT_APRILTAG_16h5"
SUPPORTED_DICTIONARIES = (
    "DICT_APRILTAG_16h5",
    "DICT_4X4_50",
)


def dictionary_name(value: str) -> str:
    """Normalize user-friendly dictionary aliases to OpenCV constant names."""
    normalized = str(value).strip().upper()
    aliases = {
        "16H5": "DICT_APRILTAG_16h5",
        "APRILTAG_16H5": "DICT_APRILTAG_16h5",
        "DICT_APRILTAG_16H5": "DICT_APRILTAG_16h5",
        "4X4_50": "DICT_4X4_50",
        "DICT_4X4_50": "DICT_4X4_50",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        choices = ", ".join(SUPPORTED_DICTIONARIES)
        raise argparse.ArgumentTypeError(
            f"unsupported dictionary {value!r}; choose one of: {choices}"
        )
    return resolved


def opencv_dictionary(name: str):
    dictionary_id = getattr(cv2.aruco, name, None)
    if dictionary_id is None:
        raise RuntimeError(
            f"This OpenCV build does not provide cv2.aruco.{name}. "
            "Install/update opencv-contrib-python."
        )
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def invert_transform(T: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T[:3, 3]
    return result


def make_transform(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(Rm, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def rotation_angle(Rm: np.ndarray) -> float:
    return float(Rotation.from_matrix(Rm).magnitude())


def marker_object_points(tag_size: float) -> np.ndarray:
    """Corners: top-left, top-right, bottom-right, bottom-left.

    Marker +X points right, +Y points toward the marker top, and +Z points
    out of the printed face. This is a right-handed coordinate frame.
    """
    h = tag_size / 2.0
    return np.array(
        [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
        dtype=np.float64,
    )


class RealSenseColorCamera:
    def __init__(self, width: int, height: int, fps: int):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is not installed. Run: pip install pyrealsense2"
            )
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(config)
        sensor = profile.get_device().first_color_sensor()
        try:
            sensor.set_option(rs.option.enable_auto_exposure, 1)
        except RuntimeError:
            pass
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = stream.get_intrinsics()
        self.K = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        # RealSense/OpenCV coefficient order is [k1, k2, p1, p2, k3].
        self.dist = np.asarray(intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)
        self.serial = profile.get_device().get_info(rs.camera_info.serial_number)

    def read(self) -> np.ndarray:
        while True:
            frames = self.pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if color:
                return np.asanyarray(color.get_data()).copy()

    def stop(self) -> None:
        self.pipeline.stop()


class AprilTagDetector:
    def __init__(self, dictionary_name: str):
        dictionary = opencv_dictionary(dictionary_name)
        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(dictionary, params)
            self._modern = True
        else:
            self.detector = (dictionary, params)
            self._modern = False

    def detect(self, image: np.ndarray) -> dict[int, np.ndarray]:
        if self._modern:
            corners, ids, _ = self.detector.detectMarkers(image)
        else:
            dictionary, params = self.detector
            corners, ids, _ = cv2.aruco.detectMarkers(
                image, dictionary, parameters=params
            )
        if ids is None:
            return {}
        result: dict[int, np.ndarray] = {}
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            marker_id = int(marker_id)
            if marker_id in TAG_IDS:
                result[marker_id] = marker_corners.reshape(4, 2).astype(np.float64)
        return result


def estimate_single_tag_pose(
    image_points: np.ndarray,
    tag_size: float,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    obj = marker_object_points(tag_size)
    ok, rvec, tvec = cv2.solvePnP(
        obj,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok or float(tvec[2]) <= 0.0:
        return None
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    error = float(
        np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image_points) ** 2, axis=1)))
    )
    Rm, _ = cv2.Rodrigues(rvec)
    return make_transform(Rm, tvec), error


def polygon_area(corners: np.ndarray) -> float:
    return abs(float(cv2.contourArea(corners.astype(np.float32))))


def robust_mean_transforms(samples: list[np.ndarray]) -> np.ndarray:
    if not samples:
        raise ValueError("Cannot average an empty transform list")
    translations = np.stack([T[:3, 3] for T in samples])
    med_t = np.median(translations, axis=0)
    trans_dev = np.linalg.norm(translations - med_t, axis=1)
    med_dev = float(np.median(trans_dev))
    trans_limit = max(0.002, 3.5 * med_dev)

    seed_R = Rotation.from_matrix(np.stack([T[:3, :3] for T in samples])).mean()
    rot_dev = np.array(
        [(seed_R.inv() * Rotation.from_matrix(T[:3, :3])).magnitude() for T in samples]
    )
    med_rot = float(np.median(rot_dev))
    rot_limit = max(math.radians(4.0), 3.5 * med_rot)
    keep = (trans_dev <= trans_limit) & (rot_dev <= rot_limit)
    if int(np.sum(keep)) < max(3, len(samples) // 3):
        keep[:] = True

    kept = [T for T, good in zip(samples, keep) if good]
    mean_t = np.mean(np.stack([T[:3, 3] for T in kept]), axis=0)
    mean_R = Rotation.from_matrix(np.stack([T[:3, :3] for T in kept])).mean().as_matrix()
    return make_transform(mean_R, mean_t)


def build_root_poses(
    pair_samples: dict[tuple[int, int], list[np.ndarray]], min_samples: int
) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], np.ndarray]]:
    """Build T_tag0_tagi from robust pairwise transforms."""
    edges: dict[tuple[int, int], np.ndarray] = {}
    adjacency: dict[int, list[tuple[int, np.ndarray]]] = collections.defaultdict(list)
    for (a, b), samples in pair_samples.items():
        if len(samples) < min_samples:
            continue
        T_a_b = robust_mean_transforms(samples)
        edges[(a, b)] = T_a_b
        adjacency[a].append((b, T_a_b))
        adjacency[b].append((a, invert_transform(T_a_b)))

    poses = {0: np.eye(4, dtype=np.float64)}
    queue = collections.deque([0])
    while queue:
        a = queue.popleft()
        # Prefer well-sampled edges when several paths are available.
        neighbours = sorted(
            adjacency[a],
            key=lambda item: len(pair_samples.get(tuple(sorted((a, item[0]))), [])),
            reverse=True,
        )
        for b, T_a_b in neighbours:
            if b not in poses:
                poses[b] = poses[a] @ T_a_b
                queue.append(b)
    return poses, edges


def signed_axis_candidates(normal: np.ndarray) -> list[np.ndarray]:
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    candidates = []
    for x in axes + [-a for a in axes]:
        if abs(float(np.dot(x, normal))) > 0.1:
            continue
        y = np.cross(normal, x)  # x cross y = normal
        Rm = np.column_stack((x, y, normal))
        if np.linalg.det(Rm) > 0.99:
            candidates.append(Rm)
    return candidates


def snap_to_exact_cube(
    raw_poses: dict[int, np.ndarray], cube_size: float
) -> tuple[dict[int, np.ndarray], dict[int, str], dict[int, float]]:
    if set(raw_poses) != set(TAG_IDS):
        missing = sorted(set(TAG_IDS) - set(raw_poses))
        raise ValueError(f"Tag graph is incomplete; missing from Tag 0: {missing}")

    normals = {
        "+X": np.array([1.0, 0.0, 0.0]),
        "-X": np.array([-1.0, 0.0, 0.0]),
        "+Y": np.array([0.0, 1.0, 0.0]),
        "-Y": np.array([0.0, -1.0, 0.0]),
        "+Z": np.array([0.0, 0.0, 1.0]),
        "-Z": np.array([0.0, 0.0, -1.0]),
    }
    face_names = list(normals)
    ids = list(TAG_IDS)
    observed_normals = np.stack([raw_poses[i][:3, 2] for i in ids])
    expected_normals = np.stack([normals[name] for name in face_names])
    cost = 1.0 - observed_normals @ expected_normals.T
    # Tag 0 defines +Z by convention.
    cost[ids.index(0), :] += 1e3
    cost[ids.index(0), face_names.index("+Z")] = 0.0
    rows, cols = linear_sum_assignment(cost)
    assignment = {ids[r]: face_names[c] for r, c in zip(rows, cols)}

    exact: dict[int, np.ndarray] = {}
    angular_residuals: dict[int, float] = {}
    half = cube_size / 2.0
    for marker_id in ids:
        normal = normals[assignment[marker_id]]
        candidates = signed_axis_candidates(normal)
        raw_R = raw_poses[marker_id][:3, :3]
        candidate_errors = [rotation_angle(Rm.T @ raw_R) for Rm in candidates]
        index = int(np.argmin(candidate_errors))
        exact[marker_id] = make_transform(candidates[index], half * normal)
        angular_residuals[marker_id] = math.degrees(candidate_errors[index])

    # With the selected convention Tag 0 must be identity rotation on +Z.
    exact[0] = make_transform(np.eye(3), np.array([0.0, 0.0, half]))
    angular_residuals[0] = math.degrees(rotation_angle(raw_poses[0][:3, :3]))
    return exact, assignment, angular_residuals


def draw_text_lines(image: np.ndarray, lines: list[str]) -> None:
    overlay = image.copy()
    height = 30 + 26 * len(lines)
    cv2.rectangle(overlay, (8, 8), (image.shape[1] - 8, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0.0, image)
    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (18, 34 + 26 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_detections(image: np.ndarray, detections: dict[int, np.ndarray]) -> None:
    for marker_id, corners in detections.items():
        points = corners.astype(np.int32)
        cv2.polylines(image, [points], True, (0, 220, 0), 2, cv2.LINE_AA)
        center = np.mean(points, axis=0).astype(int)
        cv2.putText(
            image,
            f"ID {marker_id}",
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def save_model(
    path: Path,
    transforms: dict[int, np.ndarray],
    faces: dict[int, str],
    angular_residuals: dict[int, float],
    cube_size: float,
    tag_size: float,
    serial: str,
    pair_samples: dict[tuple[int, int], list[np.ndarray]],
    dictionary: str,
) -> None:
    data = {
        "format_version": 1,
        "coordinate_system": "y_up",
        "dictionary": dictionary,
        "units": "metres",
        "coordinate_convention": {
            "transform": "T_A_B maps points from frame B to frame A",
            "origin": "cube centre",
            "world_axes": "+Z outward through Tag 0; +X Tag 0 right; +Y Tag 0 top",
            "tag_axes": "+X right; +Y top; +Z outward from printed face",
        },
        "cube_size_m": float(cube_size),
        "tag_size_m": float(tag_size),
        "reference_tag_id": 0,
        "calibration_camera_serial": serial,
        "created_unix_time": time.time(),
        "pair_sample_counts": {
            f"{a}-{b}": len(samples) for (a, b), samples in sorted(pair_samples.items())
        },
        "tags": {
            int(marker_id): {
                "face": faces[marker_id],
                "snap_angular_residual_deg": float(angular_residuals[marker_id]),
                "T_world_tag": np.round(transforms[marker_id], 12).tolist(),
            }
            for marker_id in TAG_IDS
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def calibrate(args: argparse.Namespace) -> int:
    camera = RealSenseColorCamera(args.width, args.height, args.fps)
    detector = AprilTagDetector(args.dictionary)
    pair_samples: dict[tuple[int, int], list[np.ndarray]] = collections.defaultdict(list)
    seen = set()
    last_store_time: dict[tuple[int, int], float] = {}
    message = "Show Tag 0, then slowly tilt the cube to show two adjacent faces."
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            image = camera.read()
            detections = detector.detect(image)
            seen.update(detections)
            poses: dict[int, np.ndarray] = {}
            for marker_id, corners in detections.items():
                result = estimate_single_tag_pose(corners, args.tag_size, camera.K, camera.dist)
                if result is None:
                    continue
                T_c_tag, error = result
                if (
                    error <= args.max_reprojection_error
                    and polygon_area(corners) >= args.min_marker_area
                ):
                    poses[marker_id] = T_c_tag

            now = time.monotonic()
            visible_ids = sorted(poses)
            for index, a in enumerate(visible_ids):
                for b in visible_ids[index + 1 :]:
                    key = (a, b)
                    if len(pair_samples[key]) >= args.max_samples_per_pair:
                        continue
                    if now - last_store_time.get(key, -1e9) < args.sample_interval:
                        continue
                    T_a_b = invert_transform(poses[a]) @ poses[b]
                    distance = np.linalg.norm(T_a_b[:3, 3])
                    angle = math.degrees(rotation_angle(T_a_b[:3, :3]))
                    # Adjacent cube-face centres are sqrt(2)*half apart. The
                    # complete marker-frame rotation can be 90, 120 or 180 deg
                    # depending on each tag's in-plane quarter turn, so it must
                    # not be restricted to a narrow interval around 90 deg.
                    expected_distance = args.cube_size / math.sqrt(2.0)
                    if abs(distance - expected_distance) < 0.018 and 45.0 < angle <= 180.1:
                        pair_samples[key].append(T_a_b)
                        last_store_time[key] = now

            root_poses, _ = build_root_poses(pair_samples, args.min_samples_per_pair)
            missing = sorted(set(TAG_IDS) - set(root_poses))
            if 0 not in seen:
                message = "Show Tag 0 first."
            elif missing:
                weakest = ", ".join(map(str, missing))
                message = (
                    f"Rotate slowly: show an already-linked face together with ID(s) {weakest}."
                )
            else:
                message = "All faces linked. Press S to validate and save."

            draw_detections(image, detections)
            useful_edges = sum(
                len(v) >= args.min_samples_per_pair for v in pair_samples.values()
            )
            lines = [
                message,
                f"Visible: {visible_ids or '-'}   Seen: {sorted(seen) or '-'}",
                f"Linked to Tag 0: {sorted(root_poses)}   usable edges: {useful_edges}",
                "Keep two adjacent faces visible; SPACE pauses; S saves; Q quits.",
            ]
            draw_text_lines(image, lines)
            cv2.imshow(WINDOW_NAME, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 1
            if key == ord(" "):
                cv2.waitKey(0)
            if key == ord("s"):
                root_poses, _ = build_root_poses(pair_samples, args.min_samples_per_pair)
                if set(root_poses) != set(TAG_IDS):
                    print("Cannot save: not all IDs are linked to Tag 0.")
                    continue
                exact, faces, residuals = snap_to_exact_cube(root_poses, args.cube_size)
                worst = max(residuals.values())
                print("Detected face assignment:")
                for marker_id in TAG_IDS:
                    print(
                        f"  ID {marker_id}: {faces[marker_id]:>2}, "
                        f"snap residual {residuals[marker_id]:.2f} deg"
                    )
                if worst > args.max_snap_angle:
                    print(
                        f"Refusing to save: worst snap residual is {worst:.2f} deg; "
                        "collect more views or check that tags are aligned to cube edges."
                    )
                    continue
                save_model(
                    args.output,
                    exact,
                    faces,
                    residuals,
                    args.cube_size,
                    args.tag_size,
                    camera.serial,
                    pair_samples,
                    args.dictionary,
                )
                print(f"Saved cube model to: {args.output.resolve()}")
                return 0
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def load_model(
    path: Path,
    expected_dictionary: str | None = None,
) -> tuple[dict, dict[int, np.ndarray]]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    configured_dictionary = str(data.get("dictionary", ""))
    if configured_dictionary not in SUPPORTED_DICTIONARIES:
        raise ValueError(
            f"Configuration uses unsupported dictionary {configured_dictionary!r}"
        )
    if (
        expected_dictionary is not None
        and configured_dictionary != expected_dictionary
    ):
        raise ValueError(
            "Configuration dictionary does not match --dictionary: "
            f"config={configured_dictionary}, selected={expected_dictionary}"
        )
    transforms = {
        int(marker_id): np.asarray(entry["T_world_tag"], dtype=np.float64)
        for marker_id, entry in data["tags"].items()
    }
    if set(transforms) != set(TAG_IDS):
        raise ValueError("Configuration must contain Tag IDs 0..5")
    return data, transforms


def estimate_cube_pose_joint(
    detections: dict[int, np.ndarray],
    transforms: dict[int, np.ndarray],
    tag_size: float,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, float, int] | None:
    object_blocks = []
    image_blocks = []
    local_corners = marker_object_points(tag_size)
    for marker_id, image_corners in detections.items():
        if marker_id not in transforms:
            continue
        T_w_tag = transforms[marker_id]
        homogeneous = np.column_stack((local_corners, np.ones(4)))
        world_corners = (T_w_tag @ homogeneous.T).T[:, :3]
        object_blocks.append(world_corners)
        image_blocks.append(image_corners)
    if not object_blocks:
        return None
    object_points = np.vstack(object_blocks).astype(np.float64)
    image_points = np.vstack(image_blocks).astype(np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_SQPNP,
        reprojectionError=3.0,
        confidence=0.999,
        iterationsCount=150,
    )
    if not ok:
        # A single planar tag can occasionally be rejected by RANSAC.
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inlier_count = len(object_points) if ok else 0
    else:
        inlier_count = 0 if inliers is None else len(inliers)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            dist,
            rvec,
            tvec,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        return None
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    rms = float(
        np.sqrt(
            np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))
        )
    )
    Rm, _ = cv2.Rodrigues(rvec)
    return make_transform(Rm, tvec), rms, inlier_count  # T_camera_world


def track(args: argparse.Namespace) -> int:
    data, transforms = load_model(args.config, args.dictionary)
    tag_size = float(data["tag_size_m"])
    camera = RealSenseColorCamera(args.width, args.height, args.fps)
    detector = AprilTagDetector(args.dictionary)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            image = camera.read()
            detections = detector.detect(image)
            result = estimate_cube_pose_joint(
                detections, transforms, tag_size, camera.K, camera.dist
            )
            draw_detections(image, detections)
            if result is None:
                lines = ["Cube not localized", "Show at least one Tag; Q quits."]
            else:
                T_c_w, rms, inliers = result
                T_w_c = invert_transform(T_c_w)
                rpy = Rotation.from_matrix(T_w_c[:3, :3]).as_euler("xyz", degrees=True)
                position_mm = 1000.0 * T_w_c[:3, 3]
                rvec, _ = cv2.Rodrigues(T_c_w[:3, :3])
                cv2.drawFrameAxes(
                    image,
                    camera.K,
                    camera.dist,
                    rvec,
                    T_c_w[:3, 3].reshape(3, 1),
                    0.035,
                    2,
                )
                lines = [
                    f"Visible IDs: {sorted(detections)}   RMS: {rms:.2f}px   inliers: {inliers}",
                    "camera position in world [mm]: "
                    f"{position_mm[0]:+.1f}, {position_mm[1]:+.1f}, {position_mm[2]:+.1f}",
                    "camera RPY in world [deg]: "
                    f"{rpy[0]:+.1f}, {rpy[1]:+.1f}, {rpy[2]:+.1f}",
                    "P prints T_world_camera; Q quits.",
                ]
            draw_text_lines(image, lines)
            cv2.imshow(WINDOW_NAME, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("p") and result is not None:
                print("T_world_camera =")
                print(np.array2string(invert_transform(result[0]), precision=8, suppress_small=True))
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def self_test() -> int:
    # Generate an exact cube with arbitrary quarter-turn tag orientations.
    normals = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    exact = {}
    for marker_id, normal in enumerate(normals):
        candidates = signed_axis_candidates(normal)
        Rm = np.eye(3) if marker_id == 0 else candidates[marker_id % len(candidates)]
        exact[marker_id] = make_transform(Rm, 0.025 * normal)
    T_tag0_world = invert_transform(exact[0])
    raw = {i: T_tag0_world @ exact[i] for i in TAG_IDS}
    snapped, faces, residuals = snap_to_exact_cube(raw, DEFAULT_CUBE_SIZE)
    assert faces[0] == "+Z"
    assert max(residuals.values()) < 1e-7
    for marker_id in TAG_IDS:
        assert np.allclose(snapped[marker_id], exact[marker_id], atol=1e-10)

    T = make_transform(Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix(), [1, 2, 3])
    assert np.allclose(T @ invert_transform(T), np.eye(4), atol=1e-12)
    print("Self-test passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--width", type=int, default=1280)
    common.add_argument("--height", type=int, default=720)
    common.add_argument("--fps", type=int, default=30)
    common.add_argument(
        "--dictionary",
        "--dict",
        type=dictionary_name,
        default=DEFAULT_DICTIONARY,
        metavar="NAME",
        help=(
            "marker dictionary: DICT_APRILTAG_16h5 or DICT_4X4_50 "
            f"(default: {DEFAULT_DICTIONARY})"
        ),
    )

    cal = subparsers.add_parser("calibrate", parents=[common])
    cal.add_argument("--output", type=Path, default=Path("cube_model.yaml"))
    cal.add_argument("--cube-size", type=float, default=DEFAULT_CUBE_SIZE, help="metres")
    cal.add_argument("--tag-size", type=float, default=DEFAULT_TAG_SIZE, help="metres")
    cal.add_argument("--min-samples-per-pair", type=int, default=20)
    cal.add_argument("--max-samples-per-pair", type=int, default=160)
    cal.add_argument("--sample-interval", type=float, default=0.08)
    cal.add_argument("--max-reprojection-error", type=float, default=1.8)
    cal.add_argument("--min-marker-area", type=float, default=1200.0)
    cal.add_argument("--max-snap-angle", type=float, default=18.0)
    cal.set_defaults(func=calibrate)

    tracker = subparsers.add_parser("track", parents=[common])
    tracker.add_argument("--config", type=Path, default=Path("cube_model.yaml"))
    tracker.set_defaults(func=track)

    test = subparsers.add_parser("self-test")
    test.set_defaults(func=lambda _args: self_test())
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
