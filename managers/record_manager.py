import asyncio
import json
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.structured_log import slog
from common.run_manifest import build_run_manifest, write_manifest

from config import BackendConfig
from core.event_bus import EventBus
from core.types import DetectionResult


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, DetectionResult):
        d = asdict(obj)
        d["pose_world"] = obj.pose_world.tolist()
        return d
    return obj


class RecordManager:
    """Record pre-fusion (tracking) and post-fusion outputs for offline analysis."""

    def __init__(
        self,
        bus: EventBus,
        cfg: BackendConfig,
        *,
        out_dir: str | Path = "records",
        run_name: str | None = None,
        record_vis: bool | None = None,
        vis_fps: float | None = None,
        vis_fps_by_source: dict[str, float] | None = None,
        vis_topic: str = "visualization/frames",
        record_frontend_vis: bool | None = None,
        frontend_vis_fps: float | None = None,
        runtime_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.bus = bus
        self.cfg = cfg

        output_root = Path(out_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or self._next_result_name(output_root)
        self.out_dir = output_root / self.run_name
        self.out_dir.mkdir(parents=False, exist_ok=False)
        workspace = Path(__file__).resolve().parents[1]
        artifact_paths = [
            cfg.foundationpose.initial_mask_path,
            cfg.frontend.model_path,
            cfg.calibration.tag_cube.result_path,
            *((runtime_metadata or {}).get("artifact_paths", [])),
        ]
        hope_root = Path(cfg.foundationpose.hope_models_root)
        if hope_root.is_dir():
            artifact_paths.extend(hope_root.glob("*/google_16k/textured_simple.obj"))
        self._manifest_path = self.out_dir / "run_manifest.json"
        self._manifest = build_run_manifest(
            cfg=cfg,
            workspace=workspace,
            artifact_paths=artifact_paths,
            runtime=runtime_metadata,
        )
        write_manifest(self._manifest_path, self._manifest)

        self._q_tracking = bus.subscribe("tracking/detections")
        self._q_fused = bus.subscribe(cfg.fusion.topic_fused)
        self._q_baseline_best = bus.subscribe(cfg.baseline.topic_best)
        self._q_baseline_avg = bus.subscribe(cfg.baseline.topic_avg)
        self._q_baseline_single = bus.subscribe(cfg.baseline.topic_single)
        self._q_calib = bus.subscribe("calibration/extrinsics")
        effective_record_vis = record_vis
        if effective_record_vis is None:
            effective_record_vis = bool(record_frontend_vis)
        self._record_vis = bool(effective_record_vis)

        effective_vis_fps = vis_fps
        if effective_vis_fps is None:
            effective_vis_fps = frontend_vis_fps
        if effective_vis_fps is None:
            effective_vis_fps = 30.0
        self._vis_fps = max(float(effective_vis_fps), 1.0)
        self._vis_fps_by_source = {
            str(source): max(float(fps), 1.0)
            for source, fps in (vis_fps_by_source or {}).items()
            if np.isfinite(float(fps)) and float(fps) > 0.0
        }

        self._vis_topic = str(vis_topic)
        self._q_vis = bus.subscribe(self._vis_topic) if self._record_vis else None

        self._fp_tracking = (self.out_dir / "tracking_detections.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        )
        self._fp_fused = (self.out_dir / "pose_fused.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        )
        self._fp_baseline_best = (self.out_dir / "pose_baseline_best.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        )
        self._fp_baseline_avg = (self.out_dir / "pose_baseline_avg.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        )
        self._fp_baseline_single = (self.out_dir / "pose_baseline_single.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        )
        self._fp_calib = (self.out_dir / "calibration_extrinsics.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        )

        # Background writer: keep asyncio loop non-blocking.
        self._write_queue: queue.Queue = queue.Queue(maxsize=10000)
        self._stop_evt = threading.Event()
        self._writer = threading.Thread(
            target=self._writer_loop, name="RecordManagerWriter", daemon=True
        )
        self._dropped = 0
        self._dropped_total = 0
        self._vis_writers: dict[str, Any] = {}
        self._vis_paths: dict[str, Path] = {}

        self._flush_interval_s = 1.0

    @staticmethod
    def _next_result_name(output_root: Path) -> str:
        used: set[int] = set()
        for path in output_root.iterdir():
            if not path.is_dir() or not path.name.startswith("results_"):
                continue
            suffix = path.name.removeprefix("results_")
            if suffix.isdigit():
                used.add(int(suffix))
        index = max(used, default=-1) + 1
        return f"results_{index}"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        out = []
        for ch in str(name):
            if ch.isalnum() or ch in {"-", "_"}:
                out.append(ch)
            else:
                out.append("_")
        s = "".join(out).strip("_")
        return s or "camera"

    def _write_vis_frame(self, rec: dict[str, Any]) -> None:
        if not self._record_vis:
            return

        frame = rec.get("image_bgr")
        source = str(rec.get("source", "camera"))
        stream = str(rec.get("stream", "vis"))
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return

        key = str(source)
        writer = self._vis_writers.get(key)
        if writer is None:
            safe = self._sanitize_name(source)
            out_path = self.out_dir / f"vis_{safe}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            source_fps = self._vis_fps_by_source.get(source, self._vis_fps)
            writer = cv2.VideoWriter(str(out_path), fourcc, source_fps, (w, h))
            if not writer.isOpened():
                try:
                    writer.release()
                except Exception:
                    pass
                return
            self._vis_writers[key] = writer
            self._vis_paths[key] = out_path

        try:
            frame_out = np.array(frame, copy=True)
            label = f"{source} | {stream}"
            cv2.putText(
                frame_out,
                label,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame_out)
        except Exception:
            pass

    def _release_vis_writers(self) -> None:
        for writer in self._vis_writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self._vis_writers.clear()

    def _enqueue(self, target: str, rec: dict) -> None:
        try:
            self._write_queue.put_nowait((target, rec))
        except queue.Full:
            # Lowest-latency policy: drop records if writer can't keep up.
            self._dropped += 1
            self._dropped_total += 1

    def pending_item_count(self) -> int:
        total = int(self._write_queue.qsize())
        total += int(self._q_tracking.qsize())
        total += int(self._q_fused.qsize())
        total += int(self._q_baseline_best.qsize())
        total += int(self._q_baseline_avg.qsize())
        total += int(self._q_baseline_single.qsize())
        total += int(self._q_calib.qsize())
        if self._q_vis is not None:
            total += int(self._q_vis.qsize())
        return total

    def _writer_loop(self) -> None:
        last_flush = time.time()
        while (not self._stop_evt.is_set()) or (not self._write_queue.empty()):
            try:
                target, rec = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                target = None
                rec = None

            if target is not None:
                if target == "vis_frame":
                    self._write_vis_frame(rec)
                else:
                    line = json.dumps(rec, ensure_ascii=True) + "\n"
                    if target == "tracking":
                        self._fp_tracking.write(line)
                    elif target == "fused":
                        self._fp_fused.write(line)
                    elif target == "baseline_best":
                        self._fp_baseline_best.write(line)
                    elif target == "baseline_avg":
                        self._fp_baseline_avg.write(line)
                    elif target == "baseline_single":
                        self._fp_baseline_single.write(line)
                    elif target == "calib":
                        self._fp_calib.write(line)

            now = time.time()
            if now - last_flush >= self._flush_interval_s:
                try:
                    self._fp_tracking.flush()
                    self._fp_fused.flush()
                    self._fp_baseline_best.flush()
                    self._fp_baseline_avg.flush()
                    self._fp_baseline_single.flush()
                    self._fp_calib.flush()
                except Exception:
                    pass
                last_flush = now

        # Final flush on stop
        try:
            self._fp_tracking.flush()
            self._fp_fused.flush()
            self._fp_baseline_best.flush()
            self._fp_baseline_avg.flush()
            self._fp_baseline_single.flush()
            self._fp_calib.flush()
        except Exception:
            pass
        self._release_vis_writers()

    async def run(self) -> None:
        slog(
            component=self.__class__.__name__,
            event="run",
            out_dir=str(self.out_dir),
            topic_tracking="tracking/detections",
            topic_fused=self.cfg.fusion.topic_fused,
            topic_baseline_best=self.cfg.baseline.topic_best,
            topic_baseline_avg=self.cfg.baseline.topic_avg,
            topic_baseline_single=self.cfg.baseline.topic_single,
            record_vis=self._record_vis,
            vis_fps=self._vis_fps,
            vis_fps_by_source=self._vis_fps_by_source,
            vis_topic=(self._vis_topic if self._record_vis else None),
        )

        self._writer.start()

        try:
            while True:
                get_tasks = [
                    asyncio.create_task(self._q_tracking.get()),
                    asyncio.create_task(self._q_fused.get()),
                    asyncio.create_task(self._q_baseline_best.get()),
                    asyncio.create_task(self._q_baseline_avg.get()),
                    asyncio.create_task(self._q_baseline_single.get()),
                    asyncio.create_task(self._q_calib.get()),
                ]
                if self._q_vis is not None:
                    get_tasks.append(asyncio.create_task(self._q_vis.get()))
                try:
                    done, _pending = await asyncio.wait(
                        get_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    for t in get_tasks:
                        if not t.done():
                            t.cancel()

                for t in done:
                    msg = t.result()
                    if msg is None:
                        continue

                    if isinstance(msg, DetectionResult):
                        rec = {
                            "kind": "tracking/detections",
                            "t_host": float(msg.t_host),
                            "t_capture": (
                                None
                                if msg.capture_time is None
                                else float(msg.capture_time)
                            ),
                            "source": str(msg.source),
                            "score": float(msg.score),
                            "detector": msg.detector,
                            "object_name": msg.object_name,
                            "instance_id": msg.instance_id,
                            "sync_group": msg.sync_group,
                            "reproj_error": (
                                None
                                if msg.reproj_error is None
                                else float(msg.reproj_error)
                            ),
                            "covariance": (
                                None
                                if msg.covariance is None
                                else msg.covariance.tolist()
                            ),
                            "pose_world": msg.pose_world.tolist(),
                            "pose_sensor": (
                                None
                                if msg.pose_sensor is None
                                else msg.pose_sensor.tolist()
                            ),
                            "sensor_type": msg.sensor_type,
                            "known_extrinsic": (
                                None
                                if msg.known_extrinsic is None
                                else msg.known_extrinsic.tolist()
                            ),
                            "quality_metrics": _to_jsonable(msg.quality_metrics),
                            "imu_delta_pose": (
                                None
                                if msg.imu_delta_pose is None
                                else msg.imu_delta_pose.tolist()
                            ),
                            "imu_delta_covariance": (
                                None
                                if msg.imu_delta_covariance is None
                                else msg.imu_delta_covariance.tolist()
                            ),
                        }
                        self._enqueue("tracking", rec)
                    elif isinstance(msg, dict) and "T_world_camera" in msg:
                        rec = {
                            "kind": "calibration/extrinsics",
                            "t_host": float(msg.get("t_host", time.time())),
                            "msg": _to_jsonable(msg),
                        }
                        self._enqueue("calib", rec)
                    elif isinstance(msg, dict) and msg.get("source") == "baseline/best":
                        rec = {
                            "kind": self.cfg.baseline.topic_best,
                            "t_host": float(msg.get("t_host", time.time())),
                            "msg": _to_jsonable(msg),
                        }
                        self._enqueue("baseline_best", rec)
                    elif isinstance(msg, dict) and msg.get("source") == "baseline/avg":
                        rec = {
                            "kind": self.cfg.baseline.topic_avg,
                            "t_host": float(msg.get("t_host", time.time())),
                            "msg": _to_jsonable(msg),
                        }
                        self._enqueue("baseline_avg", rec)
                    elif (
                        isinstance(msg, dict) and msg.get("source") == "baseline/single"
                    ):
                        rec = {
                            "kind": self.cfg.baseline.topic_single,
                            "t_host": float(msg.get("t_host", time.time())),
                            "msg": _to_jsonable(msg),
                        }
                        self._enqueue("baseline_single", rec)
                    elif isinstance(msg, dict) and isinstance(
                        msg.get("image_bgr"), np.ndarray
                    ):
                        rec = {
                            "t_host": float(msg.get("t_host", time.time())),
                            "source": str(msg.get("source", "camera")),
                            "image_bgr": msg.get("image_bgr"),
                        }
                        rec["stream"] = str(msg.get("stream", "vis"))
                        self._enqueue("vis_frame", rec)
                    else:
                        # pose.fused is a dict in FusionManager
                        rec = {
                            "kind": self.cfg.fusion.topic_fused,
                            "t_host": float(time.time()),
                            "msg": _to_jsonable(msg),
                        }
                        self._enqueue("fused", rec)

                if self._dropped:
                    slog(
                        component=self.__class__.__name__,
                        event="dropped_records",
                        dropped=self._dropped,
                    )
                    self._dropped = 0
        finally:
            self._stop_evt.set()
            try:
                self._writer.join()
            except Exception:
                pass
            try:
                self._fp_tracking.close()
            except Exception:
                pass
            try:
                self._fp_fused.close()
            except Exception:
                pass
            try:
                self._fp_baseline_best.close()
            except Exception:
                pass
            try:
                self._fp_baseline_avg.close()
            except Exception:
                pass
            try:
                self._fp_baseline_single.close()
            except Exception:
                pass
            try:
                self._fp_calib.close()
            except Exception:
                pass
            self._manifest["status"] = "completed"
            self._manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._manifest["recording"] = {
                "dropped_records": self._dropped_total,
                "event_bus": self.bus.metrics(),
                "visualization_files": [str(path.name) for path in self._vis_paths.values()],
                "visualization_fps_by_source": self._vis_fps_by_source,
            }
            write_manifest(self._manifest_path, self._manifest)
