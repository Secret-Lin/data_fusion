from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.structured_log import slog

from config import BackendConfig
from core.event_bus import EventBus
from core.types import DetectionResult, TimeStampedPacket
from experiments.frontends import build_frontend
from experiments.observations import (
    frontend_detection_to_observation,
    observation_to_detection_result,
)
from experiments.stress import (
    StressConfig,
    delay_for_camera,
    inject_pose_noise,
    should_activate_camera,
    should_drop_observation,
)


class FrontendObservationManager:
    """
    Front-end detection + 3D observation generation for RGB-D cameras.

    Input: realsense/{serial}/stream packets (color + depth + camera metadata)
    Output: DetectionResult observations to cfg.frontend.topic_observations
    """

    def __init__(self, bus: EventBus, cfg: BackendConfig, input_mode: str = "live"):
        self.bus = bus
        self.cfg = cfg
        self.frontend_cfg = cfg.frontend
        self.exp_cfg = cfg.experiment
        self._input_mode = str(input_mode or "live").strip().lower()
        self._drop_old_live = self._input_mode == "live"

        self._frontends: dict[str, Any] = {}
        self._queues = self._subscribe_realsense_topics()
        self._T_world_cam = self._load_extrinsics()

        self._t0 = time.monotonic()
        self._pending: list[tuple[float, DetectionResult]] = []

        self._perf_interval = 20
        self._perf_count = 0
        self._perf_infer_ms_sum = 0.0
        self._perf_total_ms_sum = 0.0
        self._perf_sched_lag_ms_sum = 0.0
        self._perf_t0_wall: float | None = None
        self._perf_t0_host: float | None = None
        self._perf_drop_count = 0

        self._stress = StressConfig(
            mode=str(self.exp_cfg.mode),
            delay_s_by_camera=dict(self.exp_cfg.delay_s_by_camera),
            drop_prob_by_camera=dict(self.exp_cfg.drop_prob_by_camera),
            noise_trans_m=float(self.exp_cfg.noise_trans_m),
            noise_rot_deg=float(self.exp_cfg.noise_rot_deg),
            camera_start_time_s=dict(self.exp_cfg.camera_start_time_s),
        )

        self._publish_vis = False
        self._vis_topic = "visualization/frames"
        self._vis_stream = "seg_vis"
        self._inflight_packets = 0

    def set_visualization_stream(
        self,
        *,
        enabled: bool,
        topic: str = "visualization/frames",
        stream: str = "seg_vis",
    ) -> None:
        self._publish_vis = bool(enabled)
        self._vis_topic = str(topic)
        self._vis_stream = str(stream)

    def set_segmentation_visualization(
        self, *, enabled: bool, topic: str = "frontend/seg_vis"
    ) -> None:
        self.set_visualization_stream(enabled=enabled, topic=topic, stream="seg_vis")

    def pending_input_count(self) -> int:
        total = int(self._inflight_packets) + len(self._pending)
        for q in self._queues.values():
            total += int(q.qsize())
        return total

    @staticmethod
    def _color_for_label(label: str) -> tuple[int, int, int]:
        base = abs(hash(str(label)))
        return (
            int(64 + (base % 160)),
            int(64 + ((base >> 8) % 160)),
            int(64 + ((base >> 16) % 160)),
        )

    def _build_seg_vis(self, color_bgr: np.ndarray, dets: list[Any]) -> np.ndarray:
        vis = np.array(color_bgr, copy=True)
        for d in dets:
            mask = getattr(d, "mask", None)
            if not isinstance(mask, np.ndarray) or mask.shape[:2] != vis.shape[:2]:
                continue

            color = self._color_for_label(getattr(d, "class_name", "obj"))
            overlay = np.array(vis, copy=True)
            m = mask.astype(bool)
            overlay[m] = color
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0.0)

            bbox = getattr(d, "bbox_xyxy", None)
            if isinstance(bbox, tuple) and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            score = float(getattr(d, "score", 0.0))
            cls_name = str(getattr(d, "class_name", "obj"))
            label = f"{cls_name} {score:.2f}"
            anchor = (10, 30)
            if isinstance(bbox, tuple) and len(bbox) == 4:
                anchor = (int(bbox[0]), max(18, int(bbox[1]) - 6))
            cv2.putText(
                vis,
                label,
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return vis

    def _frontend_for_camera(self, camera_id: str):
        key = str(camera_id)
        frontend = self._frontends.get(key)
        if frontend is None:
            frontend = build_frontend(self.frontend_cfg)
            self._frontends[key] = frontend
        return frontend

    def _subscribe_realsense_topics(self) -> dict[str, asyncio.Queue]:
        topics: dict[str, asyncio.Queue] = {}
        cams = list(getattr(self.cfg, "realsense_cameras", []) or [])
        if not cams:
            topics["realsense_stream"] = self.bus.subscribe("realsense_stream")
            return topics

        for cam in cams:
            cam_id = cam.serial or cam.name or "realsense"
            topic = f"realsense/{cam_id}/stream"
            topics[topic] = self.bus.subscribe(topic)
        return topics

    def _load_extrinsics(self) -> dict[str, np.ndarray]:
        from core.coordinate_frames import transform_to_y_down

        out: dict[str, np.ndarray] = {}
        cal_cfg = getattr(self.cfg, "calibration", None)
        if cal_cfg is None:
            return out

        path = Path(getattr(cal_cfg.tag_cube, "result_path", ""))
        if not path.exists():
            return out
        try:
            data = json.loads(path.read_text())
        except Exception:
            return out

        coordinate_system = data.get(
            "coordinate_system", data.get("coord", "y_down")
        )
        for serial, T in (data.get("T_world_cam", {}) or {}).items():
            if T is None:
                continue
            try:
                out[str(serial)] = transform_to_y_down(
                    np.array(T, dtype=np.float64).reshape(4, 4),
                    coordinate_system,
                )
            except Exception:
                continue
        return out

    @staticmethod
    def _camera_id_from_source(source: str) -> str:
        # source: realsense/{serial}
        parts = str(source).split("/")
        if len(parts) >= 2:
            return parts[1]
        return str(source)

    async def _flush_delayed(self) -> None:
        if not self._pending:
            return
        now = time.monotonic()
        keep: list[tuple[float, DetectionResult]] = []
        for t_rel, det in self._pending:
            if t_rel <= now:
                await self.bus.publish(self.frontend_cfg.topic_observations, det)
            else:
                keep.append((t_rel, det))
        self._pending = keep

    async def _handle_packet(self, pkt: TimeStampedPacket) -> None:
        t_stage_start = time.perf_counter()
        t_now_wall = time.time()

        if self._perf_t0_wall is None or self._perf_t0_host is None:
            self._perf_t0_wall = t_now_wall
            self._perf_t0_host = float(pkt.t_host)

        expected_wall = self._perf_t0_wall + (float(pkt.t_host) - self._perf_t0_host)
        sched_lag_ms = (t_now_wall - expected_wall) * 1000.0

        if not isinstance(pkt.payload, dict):
            return
        frames = pkt.payload.get("frames")
        camera_meta = pkt.payload.get("camera")
        if not isinstance(frames, dict):
            return

        color = frames.get("color")
        depth = frames.get("depth")
        if color is None:
            return

        camera_id = self._camera_id_from_source(str(pkt.source))
        t_run = time.monotonic() - self._t0

        if not should_activate_camera(camera_id, t_run, self._stress):
            return
        if should_drop_observation(camera_id, self._stress):
            return

        frontend = self._frontend_for_camera(camera_id)
        T_world_cam = self._T_world_cam.get(camera_id)
        if T_world_cam is None:
            slog(component=self.__class__.__name__, event="missing_required_extrinsic", source=pkt.source, mode="fixed")
            return
        t_infer_start = time.perf_counter()
        dets = frontend.infer(
            camera_id=camera_id,
            t_host=float(pkt.t_host),
            color_bgr=color,
            depth_u16=depth,
            camera_meta=camera_meta,
            T_world_cam=T_world_cam,
        )
        infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

        # publish debug masks/bboxes
        for d in dets:
            dbg = {
                "t_host": float(d.t_host),
                "camera_id": str(d.camera_id),
                "class_name": str(d.class_name),
                "score": float(d.score),
                "bbox": d.bbox_xyxy,
                "num_points": int(d.points_world.shape[0]),
            }
            await self.bus.publish(self.frontend_cfg.debug_topic, dbg)
            await self.bus.publish(
                "tracking/segmentation_masks",
                {
                    "t_host": float(d.t_host),
                    "source": str(pkt.source),
                    "class_name": str(d.class_name),
                    "score": float(d.score),
                    "mask": d.mask,
                },
            )

        for d in dets:
            obs = frontend_detection_to_observation(d)
            det_msg = observation_to_detection_result(obs, source_suffix="seg3d")
            if det_msg is None:
                continue
            det_msg.sync_group = pkt.sync_group
            det_msg.detector = "seg3d"

            # noise injection for degraded experiments
            if self._stress.noise_trans_m > 0.0 or self._stress.noise_rot_deg > 0.0:
                Tn = inject_pose_noise(
                    np.array(det_msg.pose_world, dtype=np.float64), self._stress
                )
                det_msg = DetectionResult(
                    t_host=det_msg.t_host,
                    source=det_msg.source,
                    pose_world=Tn,
                    score=det_msg.score,
                    reproj_error=det_msg.reproj_error,
                    detector=det_msg.detector,
                    object_name=det_msg.object_name,
                    sync_group=det_msg.sync_group,
                )

            # delay injection
            dly = delay_for_camera(camera_id, self._stress)
            if dly > 0.0:
                self._pending.append((time.monotonic() + dly, det_msg))
            else:
                await self.bus.publish(self.frontend_cfg.topic_observations, det_msg)

        if self._publish_vis:
            vis = self._build_seg_vis(color, dets)
            await self.bus.publish(
                self._vis_topic,
                {
                    "t_host": float(pkt.t_host),
                    "source": str(pkt.source),
                    "stream": self._vis_stream,
                    "image_bgr": vis,
                },
            )

        total_ms = (time.perf_counter() - t_stage_start) * 1000.0
        self._perf_count += 1
        self._perf_infer_ms_sum += infer_ms
        self._perf_total_ms_sum += total_ms
        self._perf_sched_lag_ms_sum += sched_lag_ms

        if self._perf_count % self._perf_interval == 0:
            n = float(self._perf_interval)
            slog(
                component=self.__class__.__name__,
                event="perf",
                samples=int(self._perf_interval),
                infer_ms_avg=self._perf_infer_ms_sum / n,
                stage_total_ms_avg=self._perf_total_ms_sum / n,
                input_sched_lag_ms_avg=self._perf_sched_lag_ms_sum / n,
                dropped_frames=int(self._perf_drop_count),
            )
            self._perf_infer_ms_sum = 0.0
            self._perf_total_ms_sum = 0.0
            self._perf_sched_lag_ms_sum = 0.0
            self._perf_drop_count = 0

    async def _run_queue_worker(self, queue: asyncio.Queue) -> None:
        while True:
            pkt = await queue.get()
            if self._drop_old_live:
                drained = 0
                while True:
                    try:
                        pkt = queue.get_nowait()
                        drained += 1
                    except asyncio.QueueEmpty:
                        break
                self._perf_drop_count += drained
            if pkt is None:
                continue
            self._inflight_packets += 1
            try:
                await self._handle_packet(pkt)
            finally:
                self._inflight_packets = max(0, self._inflight_packets - 1)

    async def run(self) -> None:
        slog(
            component=self.__class__.__name__,
            event="run",
            detector=self.frontend_cfg.detector,
            topic_out=self.frontend_cfg.topic_observations,
            topic_debug=self.frontend_cfg.debug_topic,
            topic_vis=(self._vis_topic if self._publish_vis else None),
            stream_vis=(self._vis_stream if self._publish_vis else None),
            mode=self.exp_cfg.mode,
            input_mode=self._input_mode,
            drop_old_frames_live=bool(self._drop_old_live),
        )

        workers = [
            asyncio.create_task(self._run_queue_worker(q))
            for q in self._queues.values()
        ]

        try:
            while True:
                await asyncio.sleep(0.02)
                await self._flush_delayed()
        finally:
            for t in workers:
                t.cancel()
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)

            rem = self._perf_count % self._perf_interval
            if rem > 0:
                n = float(rem)
                slog(
                    component=self.__class__.__name__,
                    event="perf_final",
                    samples=int(rem),
                    infer_ms_avg=self._perf_infer_ms_sum / n,
                    stage_total_ms_avg=self._perf_total_ms_sum / n,
                    input_sched_lag_ms_avg=self._perf_sched_lag_ms_sum / n,
                    dropped_frames=int(self._perf_drop_count),
                )
