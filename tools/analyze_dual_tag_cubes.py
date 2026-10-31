from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from standalone.cube_model_generator import (
    AprilTagDetector,
    estimate_single_tag_pose,
    invert_transform,
    load_model,
    marker_object_points,
)


Y_UP_TO_Y_DOWN = np.diag([1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class CubePose:
    T_camera_cube: np.ndarray
    tag_ids: tuple[int, ...]
    reprojection_rms_px: float
    reprojection_max_px: float
    inlier_count: int
    accepted: bool
    reason: str


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    return math.degrees(float(Rotation.from_matrix(relative).magnitude()))


def _stats(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"count": 0, "mean": None, "std": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
    }


def _transform_payload(transform: np.ndarray) -> dict[str, Any]:
    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return {
        "matrix": value.tolist(),
        "position_m": value[:3, 3].tolist(),
        "quaternion_xyzw": Rotation.from_matrix(value[:3, :3]).as_quat().tolist(),
    }


def _robust_transform_center(transforms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not transforms:
        raise ValueError("cannot calculate a center from no transforms")
    matrices = np.stack([np.asarray(item, dtype=np.float64).reshape(4, 4) for item in transforms])
    keep = np.ones(len(matrices), dtype=bool)
    for _ in range(3):
        current = matrices[keep]
        center = np.eye(4, dtype=np.float64)
        center[:3, 3] = np.median(current[:, :3, 3], axis=0)
        center[:3, :3] = Rotation.from_matrix(current[:, :3, :3]).mean().as_matrix()
        translation = np.linalg.norm(matrices[:, :3, 3] - center[:3, 3], axis=1)
        rotation = np.asarray(
            [_rotation_error_deg(center, item) for item in matrices], dtype=np.float64
        )
        med_t = float(np.median(translation[keep]))
        mad_t = float(np.median(np.abs(translation[keep] - med_t)))
        med_r = float(np.median(rotation[keep]))
        mad_r = float(np.median(np.abs(rotation[keep] - med_r)))
        new_keep = (translation <= max(0.005, med_t + 4.5 * 1.4826 * mad_t)) & (
            rotation <= max(2.0, med_r + 4.5 * 1.4826 * mad_r)
        )
        if int(np.count_nonzero(new_keep)) < min(3, len(matrices)):
            break
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep
    current = matrices[keep]
    center = np.eye(4, dtype=np.float64)
    center[:3, 3] = np.mean(current[:, :3, 3], axis=0)
    center[:3, :3] = Rotation.from_matrix(current[:, :3, :3]).mean().as_matrix()
    return center, keep


def _project_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    transform: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        transform[:3, 3].reshape(3, 1),
        camera_matrix,
        distortion,
    )
    return np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)


def estimate_cube_pose(
    detections: dict[int, np.ndarray],
    transforms: dict[int, np.ndarray],
    tag_size_m: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    max_reprojection_error_px: float,
    max_point_error_px: float,
    min_tags: int,
    expected_transform: np.ndarray | None = None,
) -> CubePose | None:
    tag_ids = tuple(sorted(set(detections) & set(transforms)))
    if len(tag_ids) < min_tags:
        return None
    local_corners = marker_object_points(tag_size_m)
    homogeneous = np.column_stack((local_corners, np.ones(4, dtype=np.float64)))
    object_blocks = []
    image_blocks = []
    for tag_id in tag_ids:
        object_blocks.append((transforms[tag_id] @ homogeneous.T).T[:, :3])
        image_blocks.append(np.asarray(detections[tag_id], dtype=np.float64).reshape(4, 2))
    object_points = np.vstack(object_blocks).astype(np.float64)
    image_points = np.vstack(image_blocks).astype(np.float64)

    inlier_count = 0
    if len(tag_ids) == 1:
        tag_id = tag_ids[0]
        generic = cv2.solvePnPGeneric(
            local_corners,
            image_blocks[0],
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        candidates: list[tuple[float, np.ndarray]] = []
        if len(generic) >= 3 and generic[0]:
            for rvec, tvec in zip(generic[1], generic[2]):
                if float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
                    continue
                rotation, _ = cv2.Rodrigues(rvec)
                T_camera_tag = np.eye(4, dtype=np.float64)
                T_camera_tag[:3, :3] = rotation
                T_camera_tag[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
                candidate = T_camera_tag @ invert_transform(transforms[tag_id])
                candidate_errors = _project_errors(
                    object_points,
                    image_points,
                    candidate,
                    camera_matrix,
                    distortion,
                )
                cost = float(np.sqrt(np.mean(np.square(candidate_errors))))
                if expected_transform is not None:
                    cost += 50.0 * float(
                        np.linalg.norm(
                            candidate[:3, 3] - expected_transform[:3, 3]
                        )
                    )
                    cost += 0.02 * _rotation_error_deg(
                        expected_transform, candidate
                    )
                candidates.append((cost, candidate))
        if candidates:
            transform = min(candidates, key=lambda item: item[0])[1]
        else:
            result = estimate_single_tag_pose(
                image_blocks[0], tag_size_m, camera_matrix, distortion
            )
            if result is None:
                return None
            T_camera_tag, _ = result
            transform = T_camera_tag @ invert_transform(transforms[tag_id])
        inlier_count = 4
    else:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_SQPNP,
            reprojectionError=max_point_error_px,
            confidence=0.999,
            iterationsCount=200,
        )
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            inlier_count = len(object_points) if ok else 0
        else:
            inlier_count = 0 if inliers is None else int(len(inliers))
        if not ok:
            return None
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                rvec,
                tvec,
            )
        except (AttributeError, cv2.error):
            pass
        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)

    errors = _project_errors(
        object_points,
        image_points,
        transform,
        camera_matrix,
        distortion,
    )
    rms = float(np.sqrt(np.mean(np.square(errors))))
    maximum = float(np.max(errors))
    camera_points = (
        transform[:3, :3] @ object_points.T
    ).T + transform[:3, 3]
    if np.any(camera_points[:, 2] <= 0.0):
        accepted = False
        reason = "behind_camera"
    elif rms > max_reprojection_error_px:
        accepted = False
        reason = "reprojection_rms"
    elif maximum > max_point_error_px:
        accepted = False
        reason = "reprojection_max"
    else:
        accepted = True
        reason = "accepted"
    return CubePose(
        T_camera_cube=transform,
        tag_ids=tag_ids,
        reprojection_rms_px=rms,
        reprojection_max_px=maximum,
        inlier_count=inlier_count,
        accepted=accepted,
        reason=reason,
    )


def _camera_parameters(
    file: h5py.File,
    frame_index: int,
    metadata: dict[str, Any],
    serial: str,
) -> tuple[np.ndarray, np.ndarray]:
    if all(name in file for name in ("fx", "fy", "cx", "cy")):
        K = np.array(
            [
                [float(file["fx"][frame_index]), 0.0, float(file["cx"][frame_index])],
                [0.0, float(file["fy"][frame_index]), float(file["cy"][frame_index])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    elif serial in metadata and "K" in metadata[serial]:
        K = np.asarray(metadata[serial]["K"], dtype=np.float64).reshape(3, 3)
    elif "K_json" in file.attrs:
        K = np.asarray(json.loads(file.attrs["K_json"]), dtype=np.float64).reshape(3, 3)
    else:
        raise RuntimeError(f"no camera intrinsics for {serial}")
    if serial in metadata and "dist" in metadata[serial]:
        distortion = np.asarray(metadata[serial]["dist"], dtype=np.float64).reshape(-1, 1)
    elif "dist_json" in file.attrs:
        distortion = np.asarray(json.loads(file.attrs["dist_json"]), dtype=np.float64).reshape(-1, 1)
    else:
        distortion = np.zeros((5, 1), dtype=np.float64)
    return K, distortion


def _discover_sources(session: Path) -> list[tuple[str, Path]]:
    sources = [
        (f"realsense/{path.stem}", path)
        for path in sorted((session / "rgbd").glob("*.h5"))
    ]
    sources.extend(
        (f"quest/{path.stem}", path)
        for path in sorted((session / "quest").glob("*.h5"))
    )
    if not sources:
        raise RuntimeError(f"no RGB HDF5 inputs found under {session}")
    return sources


def _source_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [item for item in records if item["T_world_object"] is not None]
    summary: dict[str, Any] = {
        "sampled_frames": len(records),
        "world_cube_detected_frames": sum(bool(item["world_tag_ids"]) for item in records),
        "object_cube_detected_frames": sum(bool(item["object_tag_ids"]) for item in records),
        "world_cube_pose_accepted_frames": sum(item["world_pose_accepted"] for item in records),
        "object_cube_pose_accepted_frames": sum(item["object_pose_accepted"] for item in records),
        "paired_pose_frames": len(paired),
        "paired_pose_fraction": len(paired) / max(1, len(records)),
        "relative_temporal_rejected_frames": sum(
            item["relative_pose_reason"] == "static_temporal_gate"
            for item in records
        ),
        "world_tag_visibility": dict(sorted(_histogram(item["world_tag_ids"] for item in records).items())),
        "object_tag_visibility": dict(sorted(_histogram(item["object_tag_ids"] for item in records).items())),
        "world_reprojection_rms_px": _stats(
            [item["world_reprojection_rms_px"] for item in records]
        ),
        "object_reprojection_rms_px": _stats(
            [item["object_reprojection_rms_px"] for item in records]
        ),
    }
    if not paired:
        summary["stability"] = None
        return summary
    transforms = [item["T_world_object"] for item in paired]
    center, keep = _robust_transform_center(transforms)
    translation_mm = [
        1000.0 * float(np.linalg.norm(item[:3, 3] - center[:3, 3]))
        for item in transforms
    ]
    rotation_deg = [_rotation_error_deg(center, item) for item in transforms]
    positions_mm = 1000.0 * np.stack([item[:3, 3] for item in transforms])
    edge_count = max(3, int(math.ceil(0.10 * len(transforms))))
    start_center, _ = _robust_transform_center(transforms[:edge_count])
    end_center, _ = _robust_transform_center(transforms[-edge_count:])
    summary["stability"] = {
        "robust_center_T_world_object": _transform_payload(center),
        "robust_inlier_frames": int(np.count_nonzero(keep)),
        "translation_error_mm": _stats(translation_mm),
        "rotation_error_deg": _stats(rotation_deg),
        "position_axis_std_mm": np.std(positions_mm[keep], axis=0).tolist(),
        "start_to_end_translation_mm": 1000.0
        * float(np.linalg.norm(start_center[:3, 3] - end_center[:3, 3])),
        "start_to_end_rotation_deg": _rotation_error_deg(start_center, end_center),
    }
    grouped: dict[str, Any] = {}
    for object_tag_count in sorted(
        {len(item["object_tag_ids"]) for item in paired}
    ):
        indices = [
            index
            for index, item in enumerate(paired)
            if len(item["object_tag_ids"]) == object_tag_count
        ]
        grouped[str(object_tag_count)] = {
            "frames": len(indices),
            "translation_error_mm": _stats(
                [translation_mm[index] for index in indices]
            ),
            "rotation_error_deg": _stats(
                [rotation_deg[index] for index in indices]
            ),
        }
    summary["stability_by_visible_object_tag_count"] = grouped
    summary["_center"] = center
    return summary


def _histogram(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = ",".join(str(item) for item in value) if value else "none"
        result[key] = result.get(key, 0) + 1
    return result


def _matched_pair_metrics(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    tolerance_s: float,
) -> dict[str, Any]:
    first_valid = [item for item in first if item["T_world_object"] is not None]
    second_valid = [item for item in second if item["T_world_object"] is not None]
    if not first_valid or not second_valid:
        return {"matched_frames": 0}
    second_times = np.asarray([item["t_host"] for item in second_valid], dtype=np.float64)
    translation_mm: list[float] = []
    rotation_deg: list[float] = []
    time_delta_ms: list[float] = []
    for item in first_valid:
        index = int(np.searchsorted(second_times, item["t_host"]))
        candidates = [value for value in (index - 1, index) if 0 <= value < len(second_valid)]
        if not candidates:
            continue
        best = min(candidates, key=lambda value: abs(second_times[value] - item["t_host"]))
        delta = abs(float(second_times[best] - item["t_host"]))
        if delta > tolerance_s:
            continue
        other = second_valid[best]
        translation_mm.append(
            1000.0
            * float(
                np.linalg.norm(
                    item["T_world_object"][:3, 3]
                    - other["T_world_object"][:3, 3]
                )
            )
        )
        rotation_deg.append(
            _rotation_error_deg(item["T_world_object"], other["T_world_object"])
        )
        time_delta_ms.append(1000.0 * delta)
    return {
        "matched_frames": len(translation_mm),
        "time_delta_ms": _stats(time_delta_ms),
        "translation_difference_mm": _stats(translation_mm),
        "rotation_difference_deg": _stats(rotation_deg),
    }


def _write_csv(path: Path, records_by_source: dict[str, list[dict[str, Any]]]) -> None:
    fields = [
        "source",
        "frame_index",
        "t_host",
        "world_tag_ids",
        "object_tag_ids",
        "world_pose_accepted",
        "object_pose_accepted",
        "world_pose_reason",
        "object_pose_reason",
        "relative_pose_reason",
        "world_reprojection_rms_px",
        "object_reprojection_rms_px",
        "x_m",
        "y_m",
        "z_m",
        "qx",
        "qy",
        "qz",
        "qw",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source, records in records_by_source.items():
            for item in records:
                row = {name: item.get(name, "") for name in fields}
                row["source"] = source
                row["world_tag_ids"] = ";".join(map(str, item["world_tag_ids"]))
                row["object_tag_ids"] = ";".join(map(str, item["object_tag_ids"]))
                transform = item["T_world_object"]
                if transform is not None:
                    row["x_m"], row["y_m"], row["z_m"] = transform[:3, 3]
                    row["qx"], row["qy"], row["qz"], row["qw"] = Rotation.from_matrix(
                        transform[:3, :3]
                    ).as_quat()
                writer.writerow(row)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    session = args.session.resolve()
    output = (args.output_dir or session / "dual_tag_cube_analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    world_data, world_transforms = load_model(args.world_model.resolve())
    object_data, object_transforms = load_model(args.object_model.resolve())
    world_dictionary = str(world_data["dictionary"])
    object_dictionary = str(object_data["dictionary"])
    if world_dictionary == object_dictionary:
        raise RuntimeError("world and object cube dictionaries must be different")
    world_detector = AprilTagDetector(world_dictionary)
    object_detector = AprilTagDetector(object_dictionary)
    metadata_path = session / "camera_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    records_by_source: dict[str, list[dict[str, Any]]] = {}
    for source, path in _discover_sources(session):
        serial = path.stem
        records: list[dict[str, Any]] = []
        with h5py.File(path, "r") as file:
            color = file.get("color")
            if color is None or color.ndim != 4:
                raise RuntimeError(f"invalid color dataset: {path}")
            total = int(color.shape[0])
            indices = list(range(0, total, max(1, args.stride)))
            if args.max_frames > 0:
                indices = indices[: args.max_frames]
            accepted_native_history: list[np.ndarray] = []
            previous_native_relative: np.ndarray | None = None
            for sequence_index, frame_index in enumerate(indices):
                image = np.asarray(color[frame_index], dtype=np.uint8)
                K, distortion = _camera_parameters(file, frame_index, metadata, serial)
                world_detections = world_detector.detect(image)
                object_detections = object_detector.detect(image)
                world_pose = estimate_cube_pose(
                    world_detections,
                    world_transforms,
                    float(world_data["tag_size_m"]),
                    K,
                    distortion,
                    max_reprojection_error_px=args.max_reprojection_error_px,
                    max_point_error_px=args.max_point_error_px,
                    min_tags=args.min_tags,
                )
                expected_object_pose = None
                if (
                    world_pose is not None
                    and world_pose.accepted
                    and previous_native_relative is not None
                ):
                    expected_object_pose = (
                        world_pose.T_camera_cube @ previous_native_relative
                    )
                object_pose = estimate_cube_pose(
                    object_detections,
                    object_transforms,
                    float(object_data["tag_size_m"]),
                    K,
                    distortion,
                    max_reprojection_error_px=args.max_reprojection_error_px,
                    max_point_error_px=args.max_point_error_px,
                    min_tags=args.min_tags,
                    expected_transform=expected_object_pose,
                )
                paired = (
                    world_pose is not None
                    and object_pose is not None
                    and world_pose.accepted
                    and object_pose.accepted
                )
                relative = None
                relative_reason = "cube_pose_unavailable"
                if paired:
                    native = invert_transform(world_pose.T_camera_cube) @ object_pose.T_camera_cube
                    rejected = False
                    if args.static_temporal_gate and len(accepted_native_history) >= 5:
                        history_center, _ = _robust_transform_center(
                            accepted_native_history[-60:]
                        )
                        translation_jump_mm = 1000.0 * float(
                            np.linalg.norm(
                                native[:3, 3] - history_center[:3, 3]
                            )
                        )
                        rotation_jump_deg = _rotation_error_deg(
                            native, history_center
                        )
                        rejected = (
                            translation_jump_mm > args.temporal_gate_translation_mm
                            or rotation_jump_deg > args.temporal_gate_rotation_deg
                        )
                    if rejected:
                        relative_reason = "static_temporal_gate"
                    else:
                        relative = Y_UP_TO_Y_DOWN @ native @ Y_UP_TO_Y_DOWN
                        relative_reason = "accepted"
                        accepted_native_history.append(native)
                        previous_native_relative = native
                t_host = (
                    float(file["t_host"][frame_index])
                    if "t_host" in file
                    else float(frame_index)
                )
                records.append(
                    {
                        "frame_index": int(frame_index),
                        "t_host": t_host,
                        "world_tag_ids": tuple(sorted(world_detections)),
                        "object_tag_ids": tuple(sorted(object_detections)),
                        "world_pose_accepted": bool(world_pose and world_pose.accepted),
                        "object_pose_accepted": bool(object_pose and object_pose.accepted),
                        "world_pose_reason": world_pose.reason if world_pose else "not_detected",
                        "object_pose_reason": object_pose.reason if object_pose else "not_detected",
                        "world_reprojection_rms_px": (
                            world_pose.reprojection_rms_px if world_pose else math.nan
                        ),
                        "object_reprojection_rms_px": (
                            object_pose.reprojection_rms_px if object_pose else math.nan
                        ),
                        "relative_pose_reason": relative_reason,
                        "T_world_object": relative,
                    }
                )
                if (sequence_index + 1) % 100 == 0 or sequence_index + 1 == len(indices):
                    print(f"[{source}] {sequence_index + 1}/{len(indices)}")
        records_by_source[source] = records

    source_summaries = {
        source: _source_summary(records) for source, records in records_by_source.items()
    }
    centers = {
        source: summary.pop("_center")
        for source, summary in source_summaries.items()
        if summary.get("_center") is not None
    }
    cross_sensor_centers: dict[str, Any] = {}
    cross_sensor_matched: dict[str, Any] = {}
    source_names = list(records_by_source)
    for first_index, first_name in enumerate(source_names):
        for second_name in source_names[first_index + 1 :]:
            key = f"{first_name} <-> {second_name}"
            if first_name in centers and second_name in centers:
                cross_sensor_centers[key] = {
                    "translation_difference_mm": 1000.0
                    * float(
                        np.linalg.norm(
                            centers[first_name][:3, 3] - centers[second_name][:3, 3]
                        )
                    ),
                    "rotation_difference_deg": _rotation_error_deg(
                        centers[first_name], centers[second_name]
                    ),
                }
            cross_sensor_matched[key] = _matched_pair_metrics(
                records_by_source[first_name],
                records_by_source[second_name],
                args.sync_tolerance_ms / 1000.0,
            )
    report = {
        "session": str(session),
        "world_model": str(args.world_model.resolve()),
        "world_dictionary": world_dictionary,
        "object_model": str(args.object_model.resolve()),
        "object_dictionary": object_dictionary,
        "output_transform": "T_world_object",
        "world_frame": "centre of the DICT_APRILTAG_16h5 cube",
        "object_frame": "centre of the DICT_4X4_50 cube",
        "coordinate_system": "y_down",
        "method": (
            "same-frame independent dictionary detection; multi-tag joint PnP with "
            "temporally disambiguated single-tag IPPE fallback; "
            "T_world_object=inv(T_camera_world)*T_camera_object"
        ),
        "accuracy_scope": (
            "repeatability and cross-sensor consistency only; absolute pose accuracy "
            "requires an independent ground truth measurement"
        ),
        "settings": {
            "stride": args.stride,
            "max_frames": args.max_frames,
            "min_tags": args.min_tags,
            "max_reprojection_error_px": args.max_reprojection_error_px,
            "max_point_error_px": args.max_point_error_px,
            "sync_tolerance_ms": args.sync_tolerance_ms,
            "static_temporal_gate": args.static_temporal_gate,
            "temporal_gate_translation_mm": args.temporal_gate_translation_mm,
            "temporal_gate_rotation_deg": args.temporal_gate_rotation_deg,
        },
        "sources": source_summaries,
        "cross_sensor_robust_centers": cross_sensor_centers,
        "cross_sensor_time_matched": cross_sensor_matched,
    }
    _write_csv(output / "dual_tag_cube_poses.csv", records_by_source)
    (output / "dual_tag_cube_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote: {output}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a DICT_4X4_50 tag cube in the world frame defined by a "
            "DICT_APRILTAG_16h5 tag cube across recorded RGB sensors."
        )
    )
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--world-model", type=Path, required=True)
    parser.add_argument("--object-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-tags", type=int, default=1)
    parser.add_argument("--max-reprojection-error-px", type=float, default=3.0)
    parser.add_argument("--max-point-error-px", type=float, default=8.0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=40.0)
    parser.add_argument(
        "--static-temporal-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reject relative-pose jumps in this static-scene evaluation. Disable "
            "with --no-static-temporal-gate for moving-object recordings."
        ),
    )
    parser.add_argument("--temporal-gate-translation-mm", type=float, default=30.0)
    parser.add_argument("--temporal-gate-rotation-deg", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.stride = max(1, int(args.stride))
    args.max_frames = max(0, int(args.max_frames))
    args.min_tags = max(1, int(args.min_tags))
    args.sync_tolerance_ms = max(0.0, float(args.sync_tolerance_ms))
    args.temporal_gate_translation_mm = max(
        0.0, float(args.temporal_gate_translation_mm)
    )
    args.temporal_gate_rotation_deg = max(
        0.0, float(args.temporal_gate_rotation_deg)
    )
    analyze(args)


if __name__ == "__main__":
    main()
