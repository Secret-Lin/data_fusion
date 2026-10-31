from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from config import BackendConfig
from core.event_bus import EventBus
from core.types import DetectionResult
from experiments.h2o.dataset import H2OSequence, H2OWorldPoseSample
from experiments.h2o.geometry import rotation_distance_deg
from managers.baseline_manager import BaselineManager
from managers.fusion_manager import FusionManager


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _rotation_error_deg(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    return rotation_distance_deg(T_est[:3, :3], T_gt[:3, :3])


def _pose_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    trans = np.asarray([row["translation_error_m"] for row in rows])
    rot = np.asarray([row["rotation_error_deg"] for row in rows])
    return {
        "n": int(len(rows)),
        "translation_mean_m": float(np.mean(trans)),
        "translation_p50_m": float(np.percentile(trans, 50)),
        "translation_p95_m": float(np.percentile(trans, 95)),
        "rotation_mean_deg": float(np.mean(rot)),
        "rotation_p50_deg": float(np.percentile(rot, 50)),
        "rotation_p95_deg": float(np.percentile(rot, 95)),
        "failure_rate_5cm_10deg": float(np.mean((trans > 0.05) | (rot > 10.0))),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


async def _collect(queue: asyncio.Queue, output: list[Any]) -> None:
    while True:
        output.append(await queue.get())


async def _run(
    *,
    sequence: H2OSequence,
    output_dir: Path,
    max_frames: int,
    fps: float,
    seed: int,
    synthetic_noise: bool = False,
) -> dict[str, Any]:
    if synthetic_noise:
        raise ValueError("Artificial H2O noise has been removed; use unmodified annotations")
    cfg = BackendConfig()
    cfg.baseline.dope_only = False
    cfg.baseline.max_age_s = max(0.5 / fps, 0.05)
    cfg.baseline.single_camera_source = (
        f"h2o/{sequence.cameras[0]}/foundationpose"
    )
    # H2O observations are already transformed into the H2O world frame.
    # Keep this independent benchmark isolated from laboratory calibration.
    cfg.calibration.tag_cube.result_path = ""

    bus = EventBus()
    fusion = FusionManager(bus, asyncio.get_running_loop(), cfg)
    fusion.configure_runtime(offline_replay=True)
    baseline = BaselineManager(bus, cfg)

    fused_queue = bus.subscribe(cfg.fusion.topic_fused)
    best_queue = bus.subscribe(cfg.baseline.topic_best)
    avg_queue = bus.subscribe(cfg.baseline.topic_avg)
    single_queue = bus.subscribe(cfg.baseline.topic_single)
    fused_messages: list[Any] = []
    best_messages: list[Any] = []
    avg_messages: list[Any] = []
    single_messages: list[Any] = []

    tasks = [
        asyncio.create_task(fusion.run()),
        asyncio.create_task(baseline.run()),
        asyncio.create_task(_collect(fused_queue, fused_messages)),
        asyncio.create_task(_collect(best_queue, best_messages)),
        asyncio.create_task(_collect(avg_queue, avg_messages)),
        asyncio.create_task(_collect(single_queue, single_messages)),
    ]

    frame_ids = sequence.common_frame_ids()
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    if not frame_ids:
        raise RuntimeError("The selected H2O sequence has no complete frames")

    gt_by_frame: dict[int, np.ndarray] = {}
    object_name_by_frame: dict[int, str] = {}
    sync_index_by_frame: dict[int, int] = {}
    observation_records: list[dict[str, Any]] = []
    try:
        for frame_index, frame_id in enumerate(frame_ids):
            samples: list[H2OWorldPoseSample] = [
                sequence.load(camera_id, frame_id) for camera_id in sequence.cameras
            ]
            annotated = [sample for sample in samples if sample.object_id != 0]
            if not annotated:
                continue

            object_ids = {sample.object_id for sample in annotated}
            if len(object_ids) != 1:
                raise RuntimeError(
                    f"Frame {frame_id} has inconsistent object ids: {sorted(object_ids)}"
                )

            # H2O annotations are ground truth. The first selected view is the
            # scoring reference. This is annotation consistency, NOT tracking accuracy.
            gt = annotated[0].T_world_object
            gt_by_frame[frame_id] = gt
            object_name_by_frame[frame_id] = annotated[0].object_name
            sync_index_by_frame[frame_id] = frame_index
            t_host = (frame_id - frame_ids[0]) / fps

            for sample in annotated:
                observed = sample.T_world_object
                detector = "foundationpose"
                source = f"h2o/{sample.camera_id}/{detector}/{sample.object_name}"
                detection = DetectionResult(
                    t_host=t_host,
                    source=source,
                    pose_world=observed,
                    score=1.0,
                    detector=detector,
                    object_name=sample.object_name,
                    sync_group=frame_index,
                )
                observation_records.append(
                    {
                        "frame_id": frame_id,
                        "sync_group": frame_index,
                        "t_host": t_host,
                        "source": source,
                        "object_name": sample.object_name,
                        "pose_world": observed.tolist(),
                        "score": detection.score,
                        "observation_kind": "unmodified_h2o_annotation",
                    }
                )
                await bus.publish("tracking/detections", detection)
            await asyncio.sleep(0)

        # Allow both managers to drain reliable queues and publish final states.
        for _ in range(200):
            await asyncio.sleep(0.01)
            if fusion._q_tracking.empty() and baseline._q_tracking.empty():
                await asyncio.sleep(0.05)
                break
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "observations.jsonl", observation_records)

    gt_records = [
        {
            "frame_id": frame_id,
            "sync_group": sync_index_by_frame[frame_id],
            "t_host": (frame_id - frame_ids[0]) / fps,
            "object_name": object_name_by_frame[frame_id],
            "pose_world": gt.tolist(),
        }
        for frame_id, gt in gt_by_frame.items()
    ]
    _write_jsonl(output_dir / "ground_truth.jsonl", gt_records)

    def latest_by_step(messages: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        selected: dict[int, dict[str, Any]] = {}
        for message in messages:
            if "sync_group" in message and message["sync_group"] is not None:
                selected[int(message["sync_group"])] = message
        return selected

    # Resolve capture times through actual frame ids, including gaps.
    time_index = {frame - frame_ids[0]: index for frame, index in sync_index_by_frame.items()}

    def latest_by_time(messages: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        selected: dict[int, dict[str, Any]] = {}
        for message in messages:
            offset = int(round(float(message["t_host"]) * fps))
            if offset in time_index:
                selected[time_index[offset]] = message
        return selected

    estimates = {
        "fusion_full": latest_by_step(fused_messages),
        "baseline_best": latest_by_time(best_messages),
        "baseline_avg": latest_by_time(avg_messages),
        "baseline_single": latest_by_time(single_messages),
    }
    frame_lookup = {
        sync_index: frame_id
        for frame_id, sync_index in sync_index_by_frame.items()
    }
    metrics: dict[str, Any] = {}
    for method, by_index in estimates.items():
        rows: list[dict[str, Any]] = []
        serialized: list[dict[str, Any]] = []
        for index, message in sorted(by_index.items()):
            frame_id = frame_lookup.get(index)
            if frame_id is None:
                continue
            T_est = np.asarray(message["T"], dtype=np.float64).reshape(4, 4)
            T_gt = gt_by_frame[frame_id]
            row = {
                "frame_id": frame_id,
                "sync_group": index,
                "t_host": (frame_id - frame_ids[0]) / fps,
                "translation_error_m": float(
                    np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
                ),
                "rotation_error_deg": _rotation_error_deg(T_est, T_gt),
            }
            rows.append(row)
            serialized.append({**row, "pose_world": T_est.tolist(), "raw": message})
        _write_jsonl(output_dir / f"{method}.jsonl", serialized)
        metrics[method] = _pose_summary(rows)

    for camera_id in sequence.cameras:
        rows = []
        serialized = []
        camera_marker = f"h2o/{camera_id}/"
        for observation in observation_records:
            if camera_marker not in str(observation["source"]):
                continue
            frame_id = int(observation["frame_id"])
            T_est = np.asarray(observation["pose_world"], dtype=np.float64)
            T_gt = gt_by_frame[frame_id]
            row = {
                "frame_id": frame_id,
                "sync_group": int(observation["sync_group"]),
                "t_host": float(observation["t_host"]),
                "translation_error_m": float(
                    np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
                ),
                "rotation_error_deg": _rotation_error_deg(T_est, T_gt),
            }
            rows.append(row)
            serialized.append({**row, "pose_world": T_est.tolist()})
        method = f"sensor_{camera_id}"
        _write_jsonl(output_dir / f"{method}.jsonl", serialized)
        metrics[method] = _pose_summary(rows)

    report = {
        "benchmark": "h2o_annotation_fusion",
        "scope": (
            "Unmodified H2O annotation consistency through fusion, NOT independent "
            "tracking or absolute accuracy. No noise or dropout; RGB-D frontends are not evaluated."
        ),
        "sequence": sequence.validate(),
        "cameras": list(sequence.cameras),
        "sensor_count": len(sequence.cameras),
        "fps": fps,
        "seed": seed,
        "synthetic_noise": synthetic_noise,
        "frames_requested": len(frame_ids),
        "frames_with_object": len(gt_by_frame),
        "metrics": metrics,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the independent H2O annotation-to-fusion benchmark"
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="H2O take containing cam0..cam4, e.g. subject1/h1/0",
    )
    parser.add_argument(
        "--cameras",
        default="all",
        help="Use all valid cameras, or provide a comma-separated sensor set",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0, help="Compatibility metadata only; no random noise is generated")
    parser.add_argument(
        "--synthetic-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Removed: --synthetic-noise now errors; --no-synthetic-noise is accepted",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty; choose a new results_N: {output}")
    if str(args.cameras).strip().lower() == "all":
        cameras = H2OSequence.discover_cameras(args.sequence)
    else:
        cameras = [value.strip() for value in args.cameras.split(",") if value.strip()]
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    sequence = H2OSequence(args.sequence, cameras)
    report = asyncio.run(
        _run(
            sequence=sequence,
            output_dir=output,
            max_frames=int(args.max_frames),
            fps=float(args.fps),
            seed=int(args.seed),
            synthetic_noise=bool(args.synthetic_noise),
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
