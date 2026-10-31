from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from adapters.dope_runner import DopeDetection, DopeRunner
from adapters.foundationpose_runner import FoundationPoseRunner
from config import BackendConfig
from core.event_bus import EventBus
from core.types import DetectionResult
from managers.fusion_manager import FusionManager


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default) + "\n")


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3].T @ b[:3, :3]
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def _pose_dispersion(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0, "translation_rms_m": float("nan"), "rotation_rms_deg": float("nan")}
    poses = np.asarray([row["pose_world"] for row in rows], dtype=np.float64)
    center_t = np.median(poses[:, :3, 3], axis=0)
    trans = np.linalg.norm(poses[:, :3, 3] - center_t, axis=1)
    mean_rotation = Rotation.from_matrix(poses[:, :3, :3]).mean()
    rotation = (
        mean_rotation.inv() * Rotation.from_matrix(poses[:, :3, :3])
    ).magnitude()
    return {
        "count": len(rows),
        "translation_rms_m": float(np.sqrt(np.mean(trans**2))),
        "rotation_rms_deg": float(np.degrees(np.sqrt(np.mean(rotation**2)))),
    }


def _cross_camera_consistency(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["sample_index"]), []).append(row)
    trans: list[float] = []
    rot: list[float] = []
    for values in grouped.values():
        if len(values) != 2:
            continue
        first = np.asarray(values[0]["pose_world"], dtype=np.float64)
        second = np.asarray(values[1]["pose_world"], dtype=np.float64)
        trans.append(float(np.linalg.norm(first[:3, 3] - second[:3, 3])))
        rot.append(_rotation_error_deg(first, second))
    return {
        "paired_samples": len(trans),
        "translation_rmse_m": float(np.sqrt(np.mean(np.square(trans)))) if trans else float("nan"),
        "translation_median_m": float(np.median(trans)) if trans else float("nan"),
        "rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rot)))) if rot else float("nan"),
        "rotation_median_deg": float(np.median(rot)) if rot else float("nan"),
    }


def _initial_mask(
    detection: DopeDetection, depth_m: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    if detection.projected_points is None or detection.pose_cam is None:
        raise ValueError("DOPE detection has no projected cuboid")
    points = np.asarray(detection.projected_points, dtype=np.float32)
    points /= max(float(detection.scale_factor), 1e-6)
    points[:, 0] = np.clip(points[:, 0], 0, shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, shape[0] - 1)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(points).astype(np.int32)), 255)
    expected_depth = float(detection.pose_cam[2, 3]) * 0.01
    valid_depths = depth_m[(mask > 0) & np.isfinite(depth_m) & (depth_m > 0.05)]
    if valid_depths.size:
        bins = np.arange(0.05, 5.05, 0.02)
        histogram, edges = np.histogram(valid_depths, bins=bins)
        minimum_peak = max(10, int(histogram.max()) // 10)
        candidates = np.flatnonzero(histogram >= minimum_peak)
        if candidates.size:
            centers = (edges[candidates] + edges[candidates + 1]) * 0.5
            expected_depth = float(centers[np.argmin(np.abs(centers - expected_depth))])
    depth_gate = (
        np.isfinite(depth_m)
        & (depth_m > 0.05)
        & (np.abs(depth_m - expected_depth) < 0.10)
    )
    mask = np.where((mask > 0) & depth_gate, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    if int(np.count_nonzero(mask)) < 30:
        raise RuntimeError("DOPE-derived initialization mask is too small")
    return mask


async def _fuse(
    rows: list[dict[str, Any]], method: str, calibration_path: Path
) -> list[dict[str, Any]]:
    cfg = BackendConfig()
    cfg.calibration.tag_cube.result_path = str(calibration_path)
    manager = FusionManager(EventBus(), asyncio.get_running_loop(), cfg)
    output: list[dict[str, Any]] = []
    by_sample: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(int(row["sample_index"]), []).append(row)

    for sample_index in sorted(by_sample):
        accepted: list[tuple[str, int]] = []
        sample_rows = sorted(by_sample[sample_index], key=lambda row: float(row["score"]), reverse=True)
        for row in sample_rows:
            result = await manager._on_pose_message(
                DetectionResult(
                    # The two H5 host clocks are not reliably ordered. Rows in
                    # one comparison sample were explicitly nearest-neighbour
                    # paired, so use that common sample time for graph binning.
                    t_host=float(sample_index),
                    source=str(row["source"]),
                    pose_world=np.asarray(row["pose_world"], dtype=np.float64),
                    score=float(row["score"]),
                )
            )
            if result is None:
                continue
            object_name, step = result
            manager._gtsam_instances[object_name].update()
            accepted.append((object_name, step))
        if not accepted:
            continue
        object_name, step = accepted[-1]
        pose = manager._gtsam_instances[object_name].estimate_tool_pose(step)
        if pose is None:
            continue
        output.append(
            {
                "method": method,
                "sample_index": sample_index,
                "t_host": min(float(row["t_host"]) for row in by_sample[sample_index]),
                "object_name": object_name,
                "step": step,
                "pose_world": pose,
                "accepted_observations": len(accepted),
            }
        )
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def _reuse_existing(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    calibration_path = args.session.resolve() / "tag_cube_extrinsics.json"
    raw_dope = _read_jsonl(output / "tracking_before_fusion_dope.jsonl")
    raw_foundationpose = _read_jsonl(
        output / "tracking_before_fusion_foundationpose.jsonl"
    )
    fused_dope = await _fuse(raw_dope, "dope", calibration_path)
    fused_foundationpose = await _fuse(
        raw_foundationpose, "foundationpose", calibration_path
    )
    _write_jsonl(output / "tracking_after_fusion_dope.jsonl", fused_dope)
    _write_jsonl(
        output / "tracking_after_fusion_foundationpose.jsonl", fused_foundationpose
    )
    report = {
        "session": str(args.session.resolve()),
        "object_name": args.object_name,
        "samples_requested": int(args.samples),
        "duration_s": float(args.duration_s),
        "dope": {
            "detections_before_fusion": len(raw_dope),
            "states_after_fusion": len(fused_dope),
            "cross_camera_before_fusion": _cross_camera_consistency(raw_dope),
            "dispersion_before_fusion_by_camera": {
                serial: _pose_dispersion([row for row in raw_dope if row["serial"] == serial])
                for serial in sorted({row["serial"] for row in raw_dope})
            },
            "dispersion_after_fusion": _pose_dispersion(fused_dope),
        },
        "foundationpose": {
            "detections_before_fusion": len(raw_foundationpose),
            "states_after_fusion": len(fused_foundationpose),
            "cross_camera_before_fusion": _cross_camera_consistency(raw_foundationpose),
            "dispersion_before_fusion_by_camera": {
                serial: _pose_dispersion([row for row in raw_foundationpose if row["serial"] == serial])
                for serial in sorted({row["serial"] for row in raw_foundationpose})
            },
            "dispersion_after_fusion": _pose_dispersion(fused_foundationpose),
        },
    }
    (output / "comparison_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    return report


def _best_detection(detections: list[DopeDetection], object_name: str) -> DopeDetection | None:
    valid = [
        detection
        for detection in detections
        if detection.object_name == object_name and detection.pose_cam is not None
    ]
    return max(valid, key=lambda detection: detection.score) if valid else None


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    session = args.session.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((session / "camera_metadata.json").read_text(encoding="utf-8"))
    calibration_path = session / "tag_cube_extrinsics.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    extrinsics = {
        serial: np.asarray(matrix, dtype=np.float64)
        for serial, matrix in calibration["T_world_cam"].items()
    }
    h5_paths = sorted((session / "rgbd").glob("*.h5"))
    if len(h5_paths) < 2:
        raise RuntimeError("Comparison requires at least two RealSense recordings")

    handles = {path.stem: h5py.File(path, "r") for path in h5_paths}
    try:
        common_start = max(float(handle["t_host"][0]) for handle in handles.values())
        common_end = min(float(handle["t_host"][-1]) for handle in handles.values())
        start = max(common_start, common_end - float(args.duration_s))
        sample_times = np.linspace(start, common_end, int(args.samples), endpoint=False)
        indices = {
            serial: np.searchsorted(np.asarray(handle["t_host"]), sample_times).clip(0, len(handle["t_host"]) - 1)
            for serial, handle in handles.items()
        }

        dope = DopeRunner(
            dope_root=str(args.dope_root),
            objects=[(args.object_name, str(args.dope_weights))],
            device="cuda",
            downscale_height=int(args.dope_height),
        )
        foundationpose = FoundationPoseRunner(
            args.foundationpose_root,
            args.hope_models_root,
            register_iterations=int(args.register_iterations),
            track_iterations=int(args.track_iterations),
        )

        raw_dope: list[dict[str, Any]] = []
        raw_foundationpose: list[dict[str, Any]] = []
        initialized: set[str] = set()
        for sample_index, target_time in enumerate(sample_times):
            for serial, handle in handles.items():
                frame_index = int(indices[serial][sample_index])
                color = np.asarray(handle["color"][frame_index])
                depth_m = np.asarray(handle["depth"][frame_index], dtype=np.float32) * float(
                    metadata[serial]["depth_scale_m"]
                )
                t_host = float(handle["t_host"][frame_index])
                detection = _best_detection(
                    dope.infer(color, metadata[serial]), args.object_name
                )
                if detection is not None:
                    pose_cam_dope = np.asarray(
                        detection.pose_cam, dtype=np.float64
                    ).copy()
                    pose_cam_dope[:3, 3] *= 0.01
                    raw_dope.append(
                        {
                            "method": "dope",
                            "sample_index": sample_index,
                            "serial": serial,
                            "frame_index": frame_index,
                            "t_host": t_host,
                            "source": f"realsense/{serial}/stream/dope/{args.object_name}",
                            "score": float(detection.score),
                            "pose_world": extrinsics[serial] @ pose_cam_dope,
                        }
                    )

                mask = None
                if serial not in initialized:
                    if detection is None:
                        continue
                    try:
                        mask = _initial_mask(detection, depth_m, color.shape[:2])
                    except RuntimeError as exc:
                        print(
                            f"skip FoundationPose init {serial} frame {frame_index}: {exc}",
                            flush=True,
                        )
                        continue
                    cv2.imwrite(str(output / f"initial_mask_{serial}.png"), mask)
                estimate = foundationpose.process(
                    source=f"realsense/{serial}/stream",
                    object_name=args.object_name,
                    color_bgr=color,
                    depth_m=depth_m,
                    camera_matrix=np.asarray(metadata[serial]["K"]),
                    mask=mask,
                )
                if estimate is None:
                    continue
                initialized.add(serial)
                raw_foundationpose.append(
                    {
                        "method": "foundationpose",
                        "mode": estimate.mode,
                        "sample_index": sample_index,
                        "serial": serial,
                        "frame_index": frame_index,
                        "t_host": t_host,
                        "source": f"realsense/{serial}/stream/foundationpose/{args.object_name}",
                        "score": float(estimate.score),
                        "pose_world": extrinsics[serial] @ estimate.pose_cam,
                    }
                )
            print(f"sample {sample_index + 1}/{len(sample_times)}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    fused_dope = await _fuse(raw_dope, "dope", calibration_path)
    fused_foundationpose = await _fuse(
        raw_foundationpose, "foundationpose", calibration_path
    )
    _write_jsonl(output / "tracking_before_fusion_dope.jsonl", raw_dope)
    _write_jsonl(output / "tracking_after_fusion_dope.jsonl", fused_dope)
    _write_jsonl(
        output / "tracking_before_fusion_foundationpose.jsonl", raw_foundationpose
    )
    _write_jsonl(
        output / "tracking_after_fusion_foundationpose.jsonl", fused_foundationpose
    )

    report = {
        "session": str(session),
        "object_name": args.object_name,
        "samples_requested": int(args.samples),
        "duration_s": float(args.duration_s),
        "dope": {
            "detections_before_fusion": len(raw_dope),
            "states_after_fusion": len(fused_dope),
            "cross_camera_before_fusion": _cross_camera_consistency(raw_dope),
            "dispersion_before_fusion_by_camera": {
                serial: _pose_dispersion([row for row in raw_dope if row["serial"] == serial])
                for serial in sorted({row["serial"] for row in raw_dope})
            },
            "dispersion_after_fusion": _pose_dispersion(fused_dope),
        },
        "foundationpose": {
            "detections_before_fusion": len(raw_foundationpose),
            "states_after_fusion": len(fused_foundationpose),
            "cross_camera_before_fusion": _cross_camera_consistency(raw_foundationpose),
            "dispersion_before_fusion_by_camera": {
                serial: _pose_dispersion(
                    [row for row in raw_foundationpose if row["serial"] == serial]
                )
                for serial in sorted({row["serial"] for row in raw_foundationpose})
            },
            "dispersion_after_fusion": _pose_dispersion(fused_foundationpose),
        },
    }
    (output / "comparison_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    return report


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--object", dest="object_name", default="TomatoSauce")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--dope-height", type=int, default=480)
    parser.add_argument("--register-iterations", type=int, default=3)
    parser.add_argument("--track-iterations", type=int, default=1)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dope-root", type=Path, default=Path(r"thirdparty/DOPE"))
    parser.add_argument("--dope-weights", type=Path, default=Path(r"thirdparty/DOPE/weights/TomatoSauce.pth"))
    parser.add_argument("--foundationpose-root", type=Path, default=repo / "thirdparty" / "FoundationPose")
    parser.add_argument("--hope-models-root", type=Path, default=repo / "assets" / "models" / "hope_original" / "cans")
    args = parser.parse_args()
    operation = _reuse_existing(args) if args.reuse_existing else _run(args)
    print(json.dumps(asyncio.run(operation), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
