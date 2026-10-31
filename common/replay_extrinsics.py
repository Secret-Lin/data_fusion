from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from adapters.quest_init import QuestTagInitializer, save_quest_extrinsic
from core.coordinate_frames import transform_to_y_down
from common.tag_dictionary import cube_dictionary
from standalone.tag_cube_localizer_replay_with_cube_model import (
    CubeModelLocalizer,
    convert_to_gray,
    detect_markers,
    extract_intrinsics,
    load_camera_metadata,
    load_cube_model,
    make_aruco_parameters,
)


class ReplayExtrinsicError(RuntimeError):
    """Raised when every replay stream cannot be placed in tag-cube world."""


@dataclass(frozen=True)
class ReplayExtrinsics:
    fixed_path: Path | None
    fixed: dict[str, np.ndarray]
    quest_path: Path | None
    quest_world_tracking: np.ndarray | None
    generated: tuple[Path, ...]


def _validate_transform(value: Any, label: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError) as exc:
        raise ReplayExtrinsicError(f"{label}: expected a numeric 4x4 transform: {exc}") from exc
    if not np.isfinite(transform).all():
        raise ReplayExtrinsicError(f"{label}: transform contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ReplayExtrinsicError(f"{label}: invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ReplayExtrinsicError(f"{label}: rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ReplayExtrinsicError(f"{label}: rotation determinant is not +1")
    return transform


def _matrix_from_pose(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, dict):
        return _validate_transform(value, label)
    position_value = value.get("pos", value.get("position"))
    quaternion_value = value.get("rot_xyzw", value.get("rotation_xyzw"))
    try:
        position = np.asarray(position_value, dtype=np.float64).reshape(3)
        x, y, z, w = np.asarray(quaternion_value, dtype=np.float64).reshape(4)
    except (TypeError, ValueError) as exc:
        raise ReplayExtrinsicError(
            f"{label}: pose object requires pos[3] and rot_xyzw[4]: {exc}"
        ) from exc
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1e-12:
        raise ReplayExtrinsicError(f"{label}: zero-length quaternion")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = position
    return _validate_transform(transform, label)


def load_fixed_extrinsics(path: Path, serials: list[str]) -> dict[str, np.ndarray]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReplayExtrinsicError(f"Fixed-camera extrinsic file is unreadable: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReplayExtrinsicError(f"Fixed-camera extrinsic root must be an object: {path}")
    coordinate_system = data.get("coordinate_system", data.get("coord", "y_down"))
    values = data.get("T_world_cam")
    if not isinstance(values, dict):
        raise ReplayExtrinsicError(f"{path}: missing T_world_cam mapping")
    output: dict[str, np.ndarray] = {}
    errors: list[str] = []
    for serial in serials:
        value = values.get(serial)
        if value is None:
            errors.append(f"{serial}: missing T_world_cam")
            continue
        try:
            transform = _validate_transform(value, f"{path}:T_world_cam[{serial}]")
            output[serial] = transform_to_y_down(transform, coordinate_system)
        except (ValueError, ReplayExtrinsicError) as exc:
            errors.append(f"{serial}: {exc}")
    if errors:
        raise ReplayExtrinsicError(
            "Fixed-camera extrinsic file is incomplete or invalid: " + "; ".join(errors)
        )
    return output


def load_quest_extrinsic(path: Path) -> np.ndarray:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReplayExtrinsicError(f"Quest extrinsic file is unreadable: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReplayExtrinsicError(f"Quest extrinsic root must be an object: {path}")
    value = data.get("T_world_tracking", data.get("T_world_quest"))
    if value is None:
        raise ReplayExtrinsicError(f"{path}: missing T_world_tracking")
    coordinate_system = data.get("coordinate_system", data.get("coord", "y_down"))
    if isinstance(value, dict):
        coordinate_system = value.get(
            "coordinate_system", value.get("coord", coordinate_system)
        )
    transform = _matrix_from_pose(value, f"{path}:T_world_tracking")
    try:
        return transform_to_y_down(transform, coordinate_system)
    except ValueError as exc:
        raise ReplayExtrinsicError(f"{path}: {exc}") from exc


def _infer_color_order(h5: h5py.File) -> str:
    value = str(h5.attrs.get("color_format", "")).strip().lower()
    return "rgb" if value.startswith("rgb") else "bgr"


def _generate_fixed_extrinsics(
    session_dir: Path,
    h5_paths: list[Path],
    cube_model_path: Path,
    output_path: Path,
) -> dict[str, np.ndarray]:
    metadata_path = session_dir / "camera_metadata.json"
    try:
        metadata = load_camera_metadata(metadata_path)
        model = load_cube_model(cube_model_path)
    except Exception as exc:
        raise ReplayExtrinsicError(f"Cannot initialize fixed-camera tag-cube calibration: {exc}") from exc
    localizer = CubeModelLocalizer(
        model=model,
        max_reproj_error_px=5.0,
        max_point_error_px=12.0,
        ransac_reproj_error_px=2.0,
        min_tags=1,
        min_observations=3,
        pose_outlier_rot_deg=5.0,
        pose_outlier_trans_m=0.03,
    )
    dictionary = cube_dictionary(model.dictionary)
    parameters = make_aruco_parameters()
    read_errors: dict[str, str] = {}
    for h5_path in h5_paths:
        serial = h5_path.stem
        try:
            with h5py.File(h5_path, "r") as h5:
                if "color" not in h5:
                    raise RuntimeError("missing color dataset")
                color = h5["color"]
                if color.ndim != 4 or color.shape[0] <= 0:
                    raise RuntimeError(f"invalid color shape {color.shape}")
                height, width = int(color.shape[1]), int(color.shape[2])
                K, distortion = extract_intrinsics(
                    metadata, serial, width, height, allow_guessed_intrinsics=False
                )
                color_order = _infer_color_order(h5)
                for frame_index in range(int(color.shape[0])):
                    localizer.add_processed_frame(serial)
                    frame = np.asarray(color[frame_index])
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    gray = convert_to_gray(frame, color_order)
                    corners, ids, _ = detect_markers(gray, dictionary, parameters)
                    if ids is None or len(ids) == 0:
                        localizer.add_no_tag_frame(serial)
                        continue
                    localizer.stats[serial].detected_any_tag += 1
                    corners_per_id = {
                        int(marker_id[0]): np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
                        for marker_corners, marker_id in zip(corners, ids)
                        if int(marker_id[0]) in model.T_world_tag
                    }
                    localizer.add_frame(
                        serial, frame_index, corners_per_id, K, distortion
                    )
        except Exception as exc:
            read_errors[serial] = str(exc)

    estimates = localizer.estimate_all()
    output: dict[str, np.ndarray] = {}
    failures: list[str] = []
    cameras: dict[str, Any] = {}
    for h5_path in h5_paths:
        serial = h5_path.stem
        estimate = estimates.get(serial)
        stats = localizer.stats[serial].asdict()
        if serial in read_errors:
            failures.append(f"{serial}: {read_errors[serial]}")
            cameras[serial] = {"valid": False, "reason": read_errors[serial], "stats": stats}
            continue
        if estimate is None:
            if stats["detected_any_tag"] == 0:
                reason = "no AprilTag from the calibrated tag cube was detected in the full sequence"
            else:
                reason = (
                    "tag detections did not produce at least 3 valid PnP observations "
                    f"(accepted={stats['accepted']}, reject_pnp={stats['reject_pnp']}, "
                    f"reject_reprojection={stats['reject_reprojection']})"
                )
            failures.append(f"{serial}: {reason}")
            cameras[serial] = {"valid": False, "reason": reason, "stats": stats}
            continue
        transform = _validate_transform(
            estimate["T_world_cam"], f"generated T_world_cam[{serial}]"
        )
        output[serial] = transform
        cameras[serial] = {
            "valid": True,
            "T_world_cam": transform.tolist(),
            "quality": estimate["quality"],
            "n_observations_total": estimate["n_observations_total"],
            "n_observations_used": estimate["n_observations_used"],
            "stats": stats,
        }
    if failures:
        raise ReplayExtrinsicError(
            "Full-sequence tag-cube calibration failed for fixed replay camera(s): "
            + "; ".join(failures)
        )
    payload = {
        "coordinate_system": "y_down",
        "convention": "column vectors; X_world = T_world_cam @ X_cam",
        "world_frame": "tag_cube",
        "generated_from": "all replay RGB frames with cube_model",
        "cube_model": str(cube_model_path),
        "T_world_cam": {key: value.tolist() for key, value in output.items()},
        "cameras": cameras,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def _generate_quest_extrinsic(
    h5_path: Path,
    cube_model_path: Path,
    output_path: Path,
    min_samples: int,
) -> np.ndarray:
    try:
        initializer = QuestTagInitializer(cube_model_path, min_samples=min_samples, strict_geometry=True)
    except Exception as exc:
        raise ReplayExtrinsicError(f"Cannot initialize Quest tag-cube calibration: {exc}") from exc
    try:
        with h5py.File(h5_path, "r") as h5:
            required = (
                "color", "fx", "fy", "cx", "cy",
                "camera_position", "camera_rotation_xyzw",
            )
            missing = [name for name in required if name not in h5]
            if missing:
                raise ReplayExtrinsicError(
                    f"{h5_path}: missing datasets required for Quest extrinsic generation: {missing}"
                )
            count = int(h5["color"].shape[0])
            coordinate_system = h5.attrs.get(
                "coordinate_system", h5.attrs.get("coord", "y_down")
            )
            color_order = _infer_color_order(h5)
            for index in range(count):
                color = np.asarray(h5["color"][index])
                if color_order == "rgb":
                    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                meta = {
                    "coordinate_system": str(coordinate_system),
                    "intrinsics": {
                        "fx": float(h5["fx"][index]),
                        "fy": float(h5["fy"][index]),
                        "cx": float(h5["cx"][index]),
                        "cy": float(h5["cy"][index]),
                    },
                    "cameraPose": {
                        "position": np.asarray(h5["camera_position"][index]).tolist(),
                        "rotation": np.asarray(h5["camera_rotation_xyzw"][index]).tolist(),
                    },
                }
                initializer.add_frame(color, meta)
    except ReplayExtrinsicError:
        raise
    except Exception as exc:
        raise ReplayExtrinsicError(f"Failed reading Quest replay {h5_path}: {exc}") from exc
    transform = initializer.finish()
    if transform is None:
        raise ReplayExtrinsicError(
            "Full-sequence Quest tag-cube calibration failed: "
            f"{len(initializer.samples)} geometric tag/PnP observations were found; "
            f"at least {initializer.min_samples} retained observations and a converged fit are required; "
            f"fit_report={initializer.fit_report}"
        )
    transform = _validate_transform(transform, "generated Quest T_world_tracking")
    save_quest_extrinsic(
        output_path,
        transform,
        len(initializer.samples),
        quality={
            "processed_frames": count,
            "accepted_tag_frames": len(initializer.samples),
            "source": "full replay sequence",
            "strict_geometry": True,
            "fit_report": initializer.fit_report,
        },
    )
    return transform


def ensure_replay_extrinsics(
    session_dir: str | Path,
    *,
    cube_model_path: str | Path,
    min_quest_samples: int = 20,
) -> ReplayExtrinsics:
    """Load or generate every replay extrinsic before any packet is published."""
    session = Path(session_dir)
    fixed_h5 = sorted((session / "rgbd").glob("*.h5")) if (session / "rgbd").is_dir() else []
    quest_h5 = sorted((session / "quest").glob("*.h5")) if (session / "quest").is_dir() else []
    model_path = Path(cube_model_path)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[1] / model_path
    generated: list[Path] = []
    fixed_path: Path | None = None
    fixed: dict[str, np.ndarray] = {}
    if fixed_h5:
        fixed_path = session / "tag_cube_extrinsics.json"
        if fixed_path.is_file():
            fixed = load_fixed_extrinsics(fixed_path, [path.stem for path in fixed_h5])
        else:
            if not model_path.is_file():
                raise ReplayExtrinsicError(
                    f"Fixed-camera extrinsics are missing ({fixed_path}) and cube model is unavailable: {model_path}"
                )
            fixed = _generate_fixed_extrinsics(session, fixed_h5, model_path, fixed_path)
            generated.append(fixed_path)

    quest_path: Path | None = None
    quest_transform: np.ndarray | None = None
    if quest_h5:
        if len(quest_h5) != 1:
            raise ReplayExtrinsicError(
                "Replay contains multiple Quest H5 files but only one shared tracking extrinsic schema; "
                f"files={[str(path) for path in quest_h5]}"
            )
        # Prefer the project-normalized Y-down file over the legacy recorder
        # output. Legacy tag_cube_pose.json files may omit their Unity Y-up
        # coordinate-system declaration.
        candidates = [session / "quest_world_extrinsic.json", session / "tag_cube_pose.json"]
        quest_path = next((path for path in candidates if path.is_file()), candidates[0])
        if quest_path.is_file():
            quest_transform = load_quest_extrinsic(quest_path)
        else:
            if not model_path.is_file():
                raise ReplayExtrinsicError(
                    f"Quest extrinsic is missing ({quest_path}) and cube model is unavailable: {model_path}"
                )
            quest_transform = _generate_quest_extrinsic(
                quest_h5[0], model_path, quest_path, max(3, int(min_quest_samples))
            )
            generated.append(quest_path)

    return ReplayExtrinsics(
        fixed_path=fixed_path,
        fixed=fixed,
        quest_path=quest_path,
        quest_world_tracking=quest_transform,
        generated=tuple(generated),
    )
