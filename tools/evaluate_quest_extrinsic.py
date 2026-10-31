from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from core.coordinate_frames import transform_to_y_down
from standalone.tag_cube_localizer_replay_with_cube_model import (
    CubeModelLocalizer,
    average_rotation,
    average_translation,
    invert_T,
    load_cube_model,
    make_aruco_parameters,
    matrix_to_rotvec,
)


def quaternion_matrix(value: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(value, dtype=np.float64).reshape(4)
    x, y, z, w = np.array([x, y, z, w]) / np.linalg.norm([x, y, z, w])
    return np.array(
        [
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
        ],
        dtype=np.float64,
    )


def load_known(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data["T_world_tracking"]
    result = np.eye(4, dtype=np.float64)
    if isinstance(value, dict):
        result[:3, :3] = quaternion_matrix(value["rot_xyzw"])
        result[:3, 3] = np.asarray(value["pos"], dtype=np.float64)
    else:
        result[:] = np.asarray(value, dtype=np.float64).reshape(4, 4)
    coordinate_system = data.get("coordinate_system", data.get("coord", "y_down"))
    if isinstance(value, dict):
        coordinate_system = value.get(
            "coordinate_system", value.get("coord", coordinate_system)
        )
    return transform_to_y_down(result, coordinate_system)


def transform_error(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    delta = invert_T(reference) @ estimate
    translation = float(np.linalg.norm(delta[:3, 3]))
    rotation = float(np.linalg.norm(matrix_to_rotvec(delta[:3, :3]))) * 180.0 / math.pi
    return translation, rotation


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def reprojection_errors(extrinsic: np.ndarray, observation: dict) -> np.ndarray:
    world_camera = extrinsic @ observation["T_tracking_camera"]
    camera_world = invert_T(world_camera)
    K = observation["K"]
    projected, _ = cv2.projectPoints(
        observation["object_points"],
        cv2.Rodrigues(camera_world[:3, :3])[0],
        camera_world[:3, 3],
        K,
        observation.get("distortion", np.zeros(5)),
    )
    projected = projected.reshape(-1, 2)
    return np.linalg.norm(projected - observation["image_points"], axis=1)


def bundle_adjust(initial: np.ndarray, observations: list[dict]) -> tuple[np.ndarray, float]:
    parameters = np.concatenate(
        [initial[:3, 3], Rotation.from_matrix(initial[:3, :3]).as_rotvec()]
    )

    def residuals(value: np.ndarray) -> np.ndarray:
        transform = make_transform(
            Rotation.from_rotvec(value[3:6]).as_matrix(), value[:3]
        )
        blocks = []
        for observation in observations:
            world_camera = transform @ observation["T_tracking_camera"]
            camera_world = invert_T(world_camera)
            K = observation["K"]
            predicted, _ = cv2.projectPoints(
                observation["object_points"],
                cv2.Rodrigues(camera_world[:3, :3])[0],
                camera_world[:3, 3],
                K,
                observation.get("distortion", np.zeros(5)),
            )
            predicted = predicted.reshape(-1, 2)
            blocks.append((predicted - observation["image_points"]).ravel())
        return np.concatenate(blocks)

    result = least_squares(
        residuals, parameters, loss="soft_l1", max_nfev=300,
        xtol=1e-10, ftol=1e-10,
    )
    optimized = make_transform(
        Rotation.from_rotvec(result.x[3:6]).as_matrix(), result.x[:3]
    )
    return optimized, float(np.sqrt(np.mean(np.square(result.fun))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recorded Quest tag-cube extrinsic")
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--cube-model", required=True, type=Path)
    parser.add_argument("--known", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--intrinsics-report",
        type=Path,
        help="JSON produced by calibrate_quest_intrinsics_from_cube.py",
    )
    parser.add_argument("--reinit-start", type=int)
    parser.add_argument("--reinit-end", type=int)
    args = parser.parse_args()

    session = args.session.resolve()
    known_path = (args.known or session / "tag_cube_pose.json").resolve()
    output = (args.output or session / "quest_extrinsic_evaluation").resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = load_cube_model(args.cube_model.resolve())
    known = load_known(known_path)
    h5_candidates = (session / "quest" / "quest.h5", session / "quest.h5")
    h5_path = next((path for path in h5_candidates if path.is_file()), None)
    if h5_path is None:
        searched = ", ".join(str(path) for path in h5_candidates)
        raise FileNotFoundError(f"Quest HDF5 not found; searched: {searched}")
    calibrated_K = None
    calibrated_distortion = np.zeros(5, dtype=np.float64)
    if args.intrinsics_report is not None:
        intrinsics_report = json.loads(
            args.intrinsics_report.resolve().read_text(encoding="utf-8")
        )
        calibrated_K = np.asarray(
            intrinsics_report["calibrated_K"], dtype=np.float64
        ).reshape(3, 3)
        calibrated_distortion = np.asarray(
            intrinsics_report.get(
                "distortion_k1_k2_p1_p2_k3", [0.0] * 5
            ),
            dtype=np.float64,
        ).reshape(-1)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    detector = cv2.aruco.ArucoDetector(dictionary, make_aruco_parameters())
    solver = CubeModelLocalizer(
        model, max_reproj_error_px=5.0, max_point_error_px=12.0,
        ransac_reproj_error_px=2.0, min_tags=1, min_observations=1,
    )
    rows: list[dict] = []
    estimates: list[np.ndarray] = []
    known_errors: list[float] = []
    observations: list[dict] = []

    with h5py.File(h5_path, "r") as stream:
        count = len(stream["color"])
        stream_coordinate_system = stream.attrs.get(
            "coordinate_system", stream.attrs.get("coord", "y_down")
        )
        for index in range(0, count, max(1, args.stride)):
            color = np.asarray(stream["color"][index])
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue
            corners_by_id = {
                int(marker_id[0]): np.asarray(marker_corners).reshape(4, 2)
                for marker_corners, marker_id in zip(corners, ids)
                if int(marker_id[0]) in model.T_world_tag
            }
            if not corners_by_id:
                continue
            K = (
                calibrated_K.copy()
                if calibrated_K is not None
                else np.array(
                    [[stream["fx"][index], 0, stream["cx"][index]],
                     [0, stream["fy"][index], stream["cy"][index]], [0, 0, 1]],
                    dtype=np.float64,
                )
            )
            distortion = (
                calibrated_distortion
                if calibrated_K is not None
                else np.zeros(5, dtype=np.float64)
            )
            object_points, image_points, tag_ids = solver._frame_points(corners_by_id)
            candidate = (
                solver._solve_single_tag(object_points, image_points, K, distortion)
                if len(tag_ids) == 1 else
                solver._solve_multiple_tags(object_points, image_points, K, distortion)
            )
            if candidate is None:
                continue
            T_cam_cv_world = np.eye(4)
            T_cam_cv_world[:3, :3] = cv2.Rodrigues(candidate.rvec)[0]
            T_cam_cv_world[:3, 3] = candidate.tvec.ravel()
            T_world_cam = invert_T(T_cam_cv_world)
            T_tracking_cam = np.eye(4)
            T_tracking_cam[:3, :3] = quaternion_matrix(stream["camera_rotation_xyzw"][index])
            T_tracking_cam[:3, 3] = stream["camera_position"][index]
            T_tracking_cam = transform_to_y_down(
                T_tracking_cam, stream_coordinate_system
            )
            estimate = T_world_cam @ invert_T(T_tracking_cam)
            estimates.append(estimate)
            trans_m, rot_deg = transform_error(known, estimate)

            T_cam_known = invert_T(known @ T_tracking_cam)
            rvec_known = cv2.Rodrigues(T_cam_known[:3, :3])[0]
            projected, _ = cv2.projectPoints(
                object_points, rvec_known, T_cam_known[:3, 3], K, distortion
            )
            errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
            rms_px = float(np.sqrt(np.mean(errors * errors)))
            known_errors.append(rms_px)
            observations.append({
                "frame_index": index,
                "object_points": object_points,
                "image_points": image_points,
                "K": K,
                "distortion": distortion,
                "T_tracking_camera": T_tracking_cam,
                "estimate": estimate,
            })
            rows.append({
                "frame_index": index, "tag_ids": "|".join(map(str, tag_ids)),
                "pnp_rms_px": candidate.quality.rms_px,
                "known_reprojection_rms_px": rms_px,
                "translation_error_m": trans_m, "rotation_error_deg": rot_deg,
                **{
                    f"T_world_tracking_{row}_{column}": float(estimate[row, column])
                    for row in range(4) for column in range(4)
                },
            })

    if not estimates:
        raise RuntimeError("No usable Quest tag observations found")
    rotations = [item[:3, :3] for item in estimates]
    translations = [item[:3, 3] for item in estimates]
    estimated = np.eye(4)
    estimated[:3, :3] = average_rotation(rotations)
    estimated[:3, 3] = average_translation(translations)
    seed_errors = [transform_error(estimated, item) for item in estimates]
    seed_translation = np.asarray([item[0] for item in seed_errors])
    seed_rotation = np.asarray([item[1] for item in seed_errors])
    translation_median = float(np.median(seed_translation))
    rotation_median = float(np.median(seed_rotation))
    translation_mad = float(np.median(np.abs(seed_translation - translation_median)))
    rotation_mad = float(np.median(np.abs(seed_rotation - rotation_median)))
    keep = (
        seed_translation <= translation_median + max(3.0 * 1.4826 * translation_mad, 0.005)
    ) & (
        seed_rotation <= rotation_median + max(3.0 * 1.4826 * rotation_mad, 0.5)
    )
    robust_estimates = [item for item, accepted in zip(estimates, keep) if accepted]
    robust_center = np.eye(4)
    robust_center[:3, :3] = average_rotation(
        [item[:3, :3] for item in robust_estimates]
    )
    robust_center[:3, 3] = average_translation(
        [item[:3, 3] for item in robust_estimates]
    )
    consistency_errors = [transform_error(robust_center, item) for item in estimates]
    consistency_translation = [item[0] for item in consistency_errors]
    consistency_rotation = [item[1] for item in consistency_errors]
    inlier_consistency = [
        item for item, accepted in zip(consistency_errors, keep) if accepted
    ]
    translation_xyz = np.vstack([item[:3, 3] for item in robust_estimates])
    window_size = min(50, len(estimates) // 2)
    first_center = np.eye(4)
    last_center = np.eye(4)
    first_center[:3, :3] = average_rotation(
        [item[:3, :3] for item in estimates[:window_size]]
    )
    first_center[:3, 3] = average_translation(
        [item[:3, 3] for item in estimates[:window_size]]
    )
    last_center[:3, :3] = average_rotation(
        [item[:3, :3] for item in estimates[-window_size:]]
    )
    last_center[:3, 3] = average_translation(
        [item[:3, 3] for item in estimates[-window_size:]]
    )
    drift_translation, drift_rotation = transform_error(first_center, last_center)
    final_trans_m, final_rot_deg = transform_error(known, estimated)
    trans = [row["translation_error_m"] for row in rows]
    rot = [row["rotation_error_deg"] for row in rows]
    report = {
        "session": str(session), "known_path": str(known_path),
        "coordinate_system": "y_down",
        "intrinsics": {
            "source": (
                str(args.intrinsics_report.resolve())
                if args.intrinsics_report is not None
                else "recorded_h5"
            ),
            "K": (
                calibrated_K.tolist()
                if calibrated_K is not None else None
            ),
            "distortion": calibrated_distortion.tolist(),
        },
        "frames_total": count, "frames_with_tags": len(rows),
        "known_T_world_tracking": known.tolist(),
        "tag_estimated_T_world_tracking": estimated.tolist(),
        "estimated_vs_known": {
            "translation_m": final_trans_m, "rotation_deg": final_rot_deg,
        },
        "per_frame_vs_known": {
            "translation_median_m": percentile(trans, 50),
            "translation_p95_m": percentile(trans, 95),
            "rotation_median_deg": percentile(rot, 50),
            "rotation_p95_deg": percentile(rot, 95),
        },
        "known_reprojection": {
            "rms_median_px": percentile(known_errors, 50),
            "rms_p95_px": percentile(known_errors, 95),
            "rms_mean_px": float(np.mean(known_errors)),
        },
        "per_frame_extrinsic_consistency": {
            "definition": "T_world_tracking_i = T_world_camera_from_tag_i @ inverse(T_tracking_camera_i)",
            "robust_center_T_world_tracking": robust_center.tolist(),
            "frames_total": len(estimates),
            "frames_used_for_robust_center": len(robust_estimates),
            "translation_error_median_m": percentile(consistency_translation, 50),
            "translation_error_p95_m": percentile(consistency_translation, 95),
            "translation_error_std_m": float(np.std(consistency_translation)),
            "rotation_error_median_deg": percentile(consistency_rotation, 50),
            "rotation_error_p95_deg": percentile(consistency_rotation, 95),
            "rotation_error_std_deg": float(np.std(consistency_rotation)),
            "robust_inliers": {
                "translation_error_median_m": percentile(
                    [item[0] for item in inlier_consistency], 50
                ),
                "translation_error_p95_m": percentile(
                    [item[0] for item in inlier_consistency], 95
                ),
                "rotation_error_median_deg": percentile(
                    [item[1] for item in inlier_consistency], 50
                ),
                "rotation_error_p95_deg": percentile(
                    [item[1] for item in inlier_consistency], 95
                ),
            },
            "translation_axis_std_m": np.std(translation_xyz, axis=0).tolist(),
            "temporal_drift_first_to_last_window": {
                "window_frames": window_size,
                "translation_m": drift_translation,
                "rotation_deg": drift_rotation,
            },
        },
        "note": "Repeatability against tag observations; not absolute metrology ground truth.",
    }

    if args.reinit_start is not None and args.reinit_end is not None:
        init_observations = [
            item for item in observations
            if args.reinit_start <= item["frame_index"] <= args.reinit_end
        ]
        expected = args.reinit_end - args.reinit_start + 1
        if len(init_observations) != expected:
            missing = sorted(
                set(range(args.reinit_start, args.reinit_end + 1))
                - {item["frame_index"] for item in init_observations}
            )
            raise RuntimeError(f"Re-INIT window has missing tag frames: {missing}")
        initial = np.eye(4)
        initial[:3, :3] = average_rotation(
            [item["estimate"][:3, :3] for item in init_observations]
        )
        initial[:3, 3] = average_translation(
            [item["estimate"][:3, 3] for item in init_observations]
        )
        new_extrinsic, init_ba_rms = bundle_adjust(initial, init_observations)
        eval_observations = [
            item for item in observations if item["frame_index"] > args.reinit_end
        ]
        comparison_rows = []
        for item in eval_observations:
            old_px = reprojection_errors(known, item)
            new_px = reprojection_errors(new_extrinsic, item)
            old_t, old_r = transform_error(known, item["estimate"])
            new_t, new_r = transform_error(new_extrinsic, item["estimate"])
            comparison_rows.append({
                "frame_index": item["frame_index"],
                "old_reprojection_rms_px": float(np.sqrt(np.mean(old_px * old_px))),
                "new_reprojection_rms_px": float(np.sqrt(np.mean(new_px * new_px))),
                "old_translation_error_m": old_t,
                "new_translation_error_m": new_t,
                "old_rotation_error_deg": old_r,
                "new_rotation_error_deg": new_r,
            })
        def metric(name: str, q: float = 50) -> float:
            return float(np.percentile([row[name] for row in comparison_rows], q))
        delta_t, delta_r = transform_error(known, new_extrinsic)
        report["reinitialization"] = {
            "frame_range_inclusive": [args.reinit_start, args.reinit_end],
            "frames": len(init_observations),
            "ba_rms_px": init_ba_rms,
            "new_T_world_tracking": new_extrinsic.tolist(),
            "new_vs_old": {"translation_m": delta_t, "rotation_deg": delta_r},
            "evaluation_frame_range": [args.reinit_end + 1, count - 1],
            "evaluation_frames_with_tags": len(comparison_rows),
            "old": {
                "reprojection_rms_median_px": metric("old_reprojection_rms_px"),
                "reprojection_rms_p95_px": metric("old_reprojection_rms_px", 95),
                "translation_median_m": metric("old_translation_error_m"),
                "rotation_median_deg": metric("old_rotation_error_deg"),
            },
            "new": {
                "reprojection_rms_median_px": metric("new_reprojection_rms_px"),
                "reprojection_rms_p95_px": metric("new_reprojection_rms_px", 95),
                "translation_median_m": metric("new_translation_error_m"),
                "rotation_median_deg": metric("new_rotation_error_deg"),
            },
        }
        (output / "tag_cube_pose_reinitialized.json").write_text(
            json.dumps({"T_world_tracking": new_extrinsic.tolist(),
                        "coordinate_system": "y_down",
                        "source_frames": [args.reinit_start, args.reinit_end],
                        "ba_rms_px": init_ba_rms}, indent=2), encoding="utf-8"
        )
        with (output / "old_vs_new_per_frame.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0]))
            writer.writeheader()
            writer.writerows(comparison_rows)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "per_frame.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
