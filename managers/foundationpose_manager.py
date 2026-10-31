from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from adapters.foundationpose_runner import FoundationPoseEstimate, FoundationPoseRunner
from common.structured_log import slog
from config import FoundationPoseConfig
from core.coordinate_frames import transform_to_y_down
from core.event_bus import EventBus
from core.types import DetectionResult, TimeStampedPacket, rgbd_frame_from_packet


class FoundationPoseManager:
    """EventBus bridge for HOPE CAD-based RGB-D registration and tracking."""

    def __init__(
        self,
        bus: EventBus,
        cfg: FoundationPoseConfig,
        *,
        realsense_topics: list[str] | None = None,
        runner: FoundationPoseRunner | None = None,
        recovery_runner: Any | None = None,
        calibration_result_path: str | Path | None = None,
        mobile_world_extrinsic_path: str | Path | None = None,
        input_mode: str = "live",
    ) -> None:
        self.bus = bus
        self.cfg = cfg
        self.runner = runner or FoundationPoseRunner(
            cfg.foundationpose_root,
            cfg.hope_models_root,
            hope_scale_to_m=cfg.hope_scale_to_m,
            register_iterations=cfg.register_iterations,
            track_iterations=cfg.track_iterations,
            score_tracking=cfg.score_tracking,
            warp_cache_path=cfg.warp_cache_path or None,
        )
        if recovery_runner is not None:
            self.recovery_runner = recovery_runner
        elif cfg.recovery_enable:
            from adapters.cnos_runner import CnosRunner

            self.recovery_runner = CnosRunner(
                cnos_root=cfg.cnos_root,
                template_dir=cfg.cnos_template_dir,
                sam_model_type=cfg.cnos_sam_model_type,
                sam_checkpoint=cfg.cnos_sam_checkpoint,
                dino_model_name=cfg.cnos_dino_model_name,
                dino_root=cfg.cnos_dino_root,
                dino_checkpoint=cfg.cnos_dino_checkpoint,
                device=cfg.cnos_device,
                image_width=cfg.cnos_image_width,
                confidence_threshold=cfg.recovery_conf_threshold,
                min_score_margin=cfg.recovery_min_score_margin,
                min_mask_fraction=cfg.recovery_min_mask_fraction,
                max_proposals=cfg.cnos_max_proposals,
            )
        else:
            self.recovery_runner = None
        topics = realsense_topics or ["realsense_stream"]
        # Replay is fully lossless. Live is bounded but also reliable: it applies
        # backpressure so FoundationPose still evaluates every accepted frame.
        queue_size = 0 if input_mode == "replay" else 8
        self._frame_queues = [bus.subscribe(topic, maxsize=queue_size) for topic in topics]
        self._command_queue = bus.subscribe(cfg.init_topic)
        self._mask_queue = bus.subscribe(cfg.mask_topic)
        self._ready_queue: asyncio.Queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_masks: dict[tuple[str, str], np.ndarray] = {}
        self._active: set[tuple[str, str]] = set()
        self._last_pose_world: dict[tuple[str, str], np.ndarray] = {}
        self._recovery_candidates: dict[
            tuple[str, str], tuple[tuple[int, int, int, int], int]
        ] = {}
        self._last_recovery_time: dict[str, float] = {}
        self._track_confidence: dict[tuple[str, str], float] = {}
        self._last_depth_fit_time: dict[tuple[str, str], float] = {}
        self._recovery_misses: dict[tuple[str, str], int] = {}
        self._T_world_camera: dict[str, np.ndarray] = {}
        self._T_world_tracking = self._load_mobile_world_extrinsic(
            mobile_world_extrinsic_path
        )
        self._reader_tasks: list[asyncio.Task] = []
        self._depth_buffers: dict[tuple[str, tuple[int, int]], np.ndarray] = {}
        self._processed_frames = 0
        self._inflight_frames = 0
        self._total_process_seconds = 0.0
        self._publish_visualization = False
        self._visualization_topic = "visualization/frames"
        self._load_extrinsics(calibration_result_path)
        if cfg.initial_mask_path and cfg.objects:
            self._load_initial_masks(Path(cfg.initial_mask_path), topics)

    @staticmethod
    def _source_from_topic(topic: str) -> str:
        value = str(topic).strip()
        return value[:-7] if value.endswith("/stream") else value

    @staticmethod
    def _safe_source_name(source: str) -> str:
        return "_".join(part for part in str(source).split("/") if part)

    def _load_initial_masks(self, path: Path, topics: list[str]) -> None:
        if path.is_file():
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Initial mask not readable: {path}")
            self._pending_masks[(self.cfg.source or "*", self.cfg.objects[0])] = mask
            return
        if not path.is_dir():
            raise FileNotFoundError(f"Initial mask path not found: {path}")

        sources = [self._source_from_topic(topic) for topic in topics]
        missing: list[str] = []
        for source in sources:
            safe_source = self._safe_source_name(source)
            for object_name in self.cfg.objects:
                candidates = (
                    path / f"{safe_source}__{object_name}.png",
                    path / f"{safe_source}_{object_name}.png",
                    path / f"{safe_source}.png",
                )
                mask_path = next((item for item in candidates if item.is_file()), None)
                if mask_path is None:
                    missing.append(f"{source}/{object_name}")
                    continue
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"Initial mask not readable: {mask_path}")
                self._pending_masks[(source, object_name)] = mask
        if missing:
            if self.recovery_runner is not None:
                slog(
                    component=self.__class__.__name__,
                    event="initial_masks_deferred_to_recovery",
                    missing=missing,
                    path=str(path),
                )
                return
            raise FileNotFoundError(
                f"Missing per-source FoundationPose masks in {path}: {', '.join(missing)}"
            )

    def set_visualization_stream(
        self,
        *,
        enabled: bool,
        topic: str = "visualization/frames",
    ) -> None:
        self._publish_visualization = bool(enabled)
        self._visualization_topic = str(topic)

    def pending_input_count(self) -> int:
        return (
            self._inflight_frames
            + self._ready_queue.qsize()
            + sum(q.qsize() for q in self._frame_queues)
        )

    async def run(self) -> None:
        async def forward(kind: str, queue: asyncio.Queue) -> None:
            while True:
                await self._ready_queue.put((kind, await queue.get()))

        await asyncio.get_running_loop().run_in_executor(
            self._executor, self.runner.prepare, self.cfg.objects
        )
        if self.recovery_runner is not None:
            await asyncio.get_running_loop().run_in_executor(
                self._executor, self.recovery_runner.prepare, self.cfg.objects
            )
        self._reader_tasks = [
            asyncio.create_task(forward("frame", queue)) for queue in self._frame_queues
        ]
        self._reader_tasks.append(
            asyncio.create_task(forward("command", self._command_queue))
        )
        self._reader_tasks.append(asyncio.create_task(forward("mask", self._mask_queue)))
        slog(
            component=self.__class__.__name__,
            event="run",
            objects=self.cfg.objects,
            available_objects=self.runner.available_objects(),
            recovery_enabled=self.recovery_runner is not None,
            init_topic=self.cfg.init_topic,
        )
        try:
            while True:
                kind, message = await self._ready_queue.get()
                if kind in {"command", "mask"}:
                    try:
                        self._handle_command(message, automatic=kind == "mask")
                    except Exception as exc:
                        slog(
                            component=self.__class__.__name__,
                            event="command_error",
                            error=repr(exc),
                        )
                else:
                    self._inflight_frames += 1
                    try:
                        await self._handle_frame(message)
                    except Exception as exc:
                        slog(
                            component=self.__class__.__name__,
                            event="frame_error",
                            source=str(getattr(message, "source", "")),
                            error=repr(exc),
                        )
                    finally:
                        self._inflight_frames -= 1
        finally:
            for task in self._reader_tasks:
                task.cancel()
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _handle_command(self, message: Any, *, automatic: bool = False) -> None:
        if not isinstance(message, dict):
            raise TypeError("FoundationPose init command must be a dict")
        object_name = str(message.get("object_name", "")).strip()
        if automatic:
            class_name = str(message.get("class_name", object_name)).strip()
            object_name = self.cfg.class_to_object.get(class_name, class_name)
        source = str(message.get("source", "*")).strip() or "*"
        if not object_name:
            raise ValueError("FoundationPose init command requires object_name")
        if bool(message.get("reset", False)):
            self.runner.reset(None if source == "*" else source, object_name)
            reset_keys = (
                [key for key in self._active if key[1] == object_name]
                if source == "*"
                else [(source, object_name)]
            )
            for key in reset_keys:
                self._active.discard(key)
                self._last_pose_world.pop(key, None)
                self._recovery_candidates.pop(key, None)
                self._track_confidence.pop(key, None)
                self._recovery_misses.pop(key, None)
                self._last_depth_fit_time.pop(key, None)
        if automatic and (
            (source, object_name) in self._active
            or (self.cfg.objects and object_name not in self.cfg.objects)
        ):
            return
        mask = message.get("mask")
        mask_path = message.get("mask_path")
        if mask is None and mask_path:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            if not bool(message.get("reset", False)):
                raise ValueError("FoundationPose init command requires mask or mask_path")
            return
        self._pending_masks[(source, object_name)] = np.asarray(mask)

    async def _handle_frame(self, pkt: TimeStampedPacket) -> None:
        if not isinstance(pkt, TimeStampedPacket):
            return
        if self.cfg.source and pkt.source != self.cfg.source:
            return
        try:
            frame = rgbd_frame_from_packet(pkt)
        except (TypeError, ValueError, KeyError):
            return
        world_camera = self._resolve_world_camera(pkt)
        if world_camera is None:
            slog(component=self.__class__.__name__, event="missing_required_extrinsic", source=pkt.source)
            return
        color, depth_raw, K = frame.color_bgr, frame.depth_raw, frame.camera.K
        buffer_key = (pkt.source, tuple(depth_raw.shape))
        depth_m = self._depth_buffers.get(buffer_key)
        if depth_m is None:
            depth_m = np.empty(depth_raw.shape, dtype=np.float32)
            self._depth_buffers[buffer_key] = depth_m
        np.multiply(depth_raw, frame.camera.depth_scale, out=depth_m, casting="unsafe")

        started = time.perf_counter()
        recovery_masks = await self._recovery_masks_for_frame(
            pkt, color, K, world_camera
        )
        visualization_estimates: list[
            tuple[FoundationPoseEstimate, np.ndarray, np.ndarray | None]
        ] = []
        for object_name in self.cfg.objects:
            key = (pkt.source, object_name)
            mask = recovery_masks.get(object_name)
            mask_is_recovery = mask is not None
            if mask is None:
                mask = self._pending_masks.pop(key, None)
            if mask is None:
                mask = self._pending_masks.pop(("*", object_name), None)
            if mask is None and key not in self._active:
                # A confirmed-lost object has no valid FoundationPose state.
                # Keep it silent until CNOS supplies a confirmed new mask.
                continue
            estimate = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                lambda obj=object_name, init_mask=mask: self.runner.process(
                    source=pkt.source,
                    object_name=obj,
                    color_bgr=np.asarray(color),
                    depth_m=depth_m,
                    camera_matrix=np.asarray(K),
                    mask=init_mask,
                ),
            )
            if estimate is None:
                continue
            self._active.add(key)
            if mask is not None:
                self._track_confidence[key] = 1.0
                self._recovery_misses[key] = 0
                if (
                    not mask_is_recovery
                    and hasattr(self.recovery_runner, "add_reference")
                ):
                    await asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        self.recovery_runner.add_reference,
                        np.asarray(color),
                        object_name,
                        np.asarray(mask),
                    )
            confidence = float(
                self._track_confidence.get(key, estimate.score)
            )
            estimate = FoundationPoseEstimate(
                object_name=estimate.object_name,
                pose_cam=estimate.pose_cam,
                score=confidence,
                mode=estimate.mode,
                scoring=estimate.scoring,
            )
            pose_world = world_camera @ estimate.pose_cam
            self._last_pose_world[key] = pose_world.copy()
            quality_metrics: dict[str, Any] = {}
            if bool(getattr(self.cfg, "depth_fit_quality_enable", False)):
                previous_fit = self._last_depth_fit_time.get(key)
                interval = max(0.0, float(getattr(self.cfg, "depth_fit_quality_interval_s", 0.0)))
                due = previous_fit is None or float(pkt.t_host) - previous_fit >= interval
                if due and hasattr(self.runner, "measure_depth_fit"):
                    try:
                        fit = await asyncio.get_running_loop().run_in_executor(
                            self._executor,
                            lambda: self.runner.measure_depth_fit(
                                source=pkt.source,
                                object_name=object_name,
                                depth_m=np.asarray(depth_m),
                                camera_matrix=np.asarray(K),
                                pose_cam=np.asarray(estimate.pose_cam),
                            ),
                        )
                        self._last_depth_fit_time[key] = float(pkt.t_host)
                        if isinstance(fit, dict):
                            quality_metrics.update(fit)
                    except Exception as exc:
                        slog(
                            component=self.__class__.__name__,
                            event="depth_fit_quality_error",
                            source=pkt.source,
                            object_name=object_name,
                            error=repr(exc),
                        )
            await self._publish(pkt, estimate, world_camera, quality_metrics=quality_metrics)
            visualization_estimates.append((estimate, pose_world, mask))
        if self._publish_visualization and visualization_estimates:
            await self._publish_visualization_frame(
                pkt,
                color,
                K,
                visualization_estimates,
            )
        elapsed = time.perf_counter() - started
        self._processed_frames += 1
        self._total_process_seconds += elapsed
        if self._processed_frames % 100 == 0:
            slog(component=self.__class__.__name__, event="performance", frames=self._processed_frames, mean_ms=1000.0 * self._total_process_seconds / self._processed_frames, pending=self.pending_input_count())

    async def _recovery_masks_for_frame(
        self,
        pkt: TimeStampedPacket,
        color_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        world_camera: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if self.recovery_runner is None:
            return {}
        source = str(pkt.source)
        needs_bootstrap = any(
            (source, object_name) not in self._active
            and (source, object_name) not in self._pending_masks
            and ("*", object_name) not in self._pending_masks
            for object_name in self.cfg.objects
        )
        previous = self._last_recovery_time.get(source)
        interval = max(0.0, float(self.cfg.recovery_interval_s))
        if previous is None:
            self._last_recovery_time[source] = float(pkt.t_host)
            if not needs_bootstrap:
                return {}
            due = True
        else:
            due = float(pkt.t_host) - float(previous) >= interval
        if not due:
            return {}
        self._last_recovery_time[source] = float(pkt.t_host)
        detections = await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self.recovery_runner.detect,
            np.asarray(color_bgr),
            self.cfg.objects,
        )
        detected = {item.object_name: item for item in detections}
        expected_bboxes = {
            object_name: self._expected_bbox(
                (source, object_name),
                camera_matrix,
                world_camera,
                color_bgr.shape[:2],
            )
            for object_name in self.cfg.objects
            if (source, object_name) in self._active
        }
        masks: dict[str, np.ndarray] = {}
        for object_name in self.cfg.objects:
            key = (source, object_name)
            item = detected.get(object_name)
            if item is None:
                self._recovery_candidates.pop(key, None)
                await self._record_recovery_miss(
                    pkt, key, reason="not_detected"
                )
                continue
            mask = np.asarray(item.mask)
            x1, y1, x2, y2 = item.bbox_xyxy
            bbox_area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
            fill = float(np.count_nonzero(mask)) / float(bbox_area)
            if fill < float(self.cfg.recovery_min_bbox_fill):
                self._recovery_candidates.pop(key, None)
                await self._record_recovery_miss(
                    pkt, key, reason="low_mask_fill"
                )
                continue

            occupied_by = self._occupied_by_other_object(
                source,
                object_name,
                item.bbox_xyxy,
                expected_bboxes,
            )
            if occupied_by is not None:
                self._recovery_candidates.pop(key, None)
                await self._record_recovery_miss(
                    pkt,
                    key,
                    reason="occupied_by_other_object",
                    extra={
                        "candidate_bbox": list(item.bbox_xyxy),
                        "occupied_by": occupied_by,
                        "cnos_score": float(item.score),
                        "score_margin": float(item.score_margin),
                    },
                )
                continue

            if key not in self._active:
                previous_candidate = self._recovery_candidates.get(key)
                consistent = (
                    previous_candidate is not None
                    and self._bbox_iou(item.bbox_xyxy, previous_candidate[0])
                    >= float(self.cfg.recovery_candidate_iou)
                )
                confirmations = previous_candidate[1] + 1 if consistent else 1
                self._recovery_candidates[key] = (
                    tuple(int(value) for value in item.bbox_xyxy),
                    confirmations,
                )
                required = max(1, int(self.cfg.recovery_confirmations))
                if confirmations < required:
                    mode = "recovery_candidate"
                else:
                    masks[object_name] = mask
                    self._recovery_candidates.pop(key, None)
                    self._track_confidence[key] = 1.0
                    self._recovery_misses[key] = 0
                    mode = "recovery_mask"
            else:
                expected_bbox = expected_bboxes.get(object_name)
                if (
                    expected_bbox is not None
                    and self._bbox_iou(item.bbox_xyxy, expected_bbox)
                    >= float(self.cfg.recovery_tracking_iou)
                ):
                    # CNOS agrees with the current FoundationPose track. Keep
                    # the cheaper official track_one result and avoid injecting
                    # mask noise through unnecessary register() calls.
                    self._recovery_candidates.pop(key, None)
                    self._recovery_misses[key] = 0
                    self._track_confidence[key] = 1.0
                    await self.bus.publish(
                        self.cfg.status_topic,
                        {
                            "t_host": float(pkt.t_host),
                            "source": source,
                            "object_name": object_name,
                            "mode": "recovery_validated",
                            "score": float(item.score),
                            "score_margin": float(item.score_margin),
                            "tracking_confidence": 1.0,
                            "bbox": list(item.bbox_xyxy),
                        },
                    )
                    continue

                previous = self._recovery_candidates.get(key)
                consistent = (
                    previous is not None
                    and self._bbox_iou(item.bbox_xyxy, previous[0])
                    >= float(self.cfg.recovery_candidate_iou)
                )
                confirmations = previous[1] + 1 if consistent else 1
                self._recovery_candidates[key] = (
                    tuple(int(value) for value in item.bbox_xyxy),
                    confirmations,
                )
                required = max(1, int(self.cfg.recovery_confirmations))
                if confirmations < required:
                    mode = "recovery_candidate"
                    await self._record_recovery_miss(
                        pkt,
                        key,
                        reason="track_disagrees_with_cnos",
                        publish=False,
                    )
                else:
                    masks[object_name] = mask
                    self._recovery_candidates.pop(key, None)
                    self._track_confidence[key] = 1.0
                    self._recovery_misses[key] = 0
                    mode = "recovery_mask"

            await self.bus.publish(
                self.cfg.status_topic,
                {
                    "t_host": float(pkt.t_host),
                    "source": source,
                    "object_name": object_name,
                    "mode": mode,
                    "score": float(item.score),
                    "score_margin": float(item.score_margin),
                    "tracking_confidence": float(
                        self._track_confidence.get(key, 0.0)
                    ),
                    "bbox": list(item.bbox_xyxy),
                    "bbox_fill": fill,
                    "confirmations": confirmations,
                },
            )
        return masks

    async def _record_recovery_miss(
        self,
        pkt: TimeStampedPacket,
        key: tuple[str, str],
        *,
        reason: str,
        extra: dict[str, Any] | None = None,
        publish: bool = True,
    ) -> None:
        if key not in self._active:
            return
        misses = self._recovery_misses.get(key, 0) + 1
        self._recovery_misses[key] = misses
        decay = float(
            np.clip(self.cfg.recovery_miss_confidence_decay, 0.0, 1.0)
        )
        confidence = float(self._track_confidence.get(key, 1.0)) * decay
        self._track_confidence[key] = confidence
        required = max(1, int(self.cfg.recovery_misses_before_lost))
        lost = misses >= required
        if lost:
            self._active.discard(key)
            self._last_pose_world.pop(key, None)
            self._recovery_candidates.pop(key, None)
            self._track_confidence[key] = 0.0
            await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self.runner.reset,
                key[0],
                key[1],
            )
            confidence = 0.0
        if publish:
            payload: dict[str, Any] = {
                "t_host": float(pkt.t_host),
                "source": key[0],
                "object_name": key[1],
                "mode": "lost" if lost else "tracking_unverified",
                "reason": reason,
                "misses": misses,
                "tracking_confidence": confidence,
            }
            payload.update(extra or {})
            await self.bus.publish(self.cfg.status_topic, payload)

    def _occupied_by_other_object(
        self,
        source: str,
        object_name: str,
        candidate_bbox: tuple[int, int, int, int],
        expected_bboxes: dict[str, tuple[int, int, int, int] | None],
    ) -> str | None:
        minimum_confidence = float(self.cfg.recovery_reservation_confidence)
        minimum_overlap = float(self.cfg.recovery_cross_object_overlap)
        for other_name, other_bbox in expected_bboxes.items():
            if other_name == object_name or other_bbox is None:
                continue
            other_key = (source, other_name)
            if self._track_confidence.get(other_key, 1.0) < minimum_confidence:
                continue
            if (
                self._bbox_intersection_over_min(candidate_bbox, other_bbox)
                >= minimum_overlap
            ):
                return other_name
        return None

    def _expected_bbox(
        self,
        key: tuple[str, str],
        camera_matrix: np.ndarray,
        world_camera: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        pose_world = self._last_pose_world.get(key)
        if pose_world is None:
            return None
        pose_cam = np.linalg.inv(np.asarray(world_camera, dtype=np.float64)) @ pose_world
        bounds = np.asarray(self.runner.model_bounds(key[1]), dtype=np.float64)
        low, high = bounds.reshape(2, 3)
        corners = np.array(
            [
                [x, y, z]
                for x in (low[0], high[0])
                for y in (low[1], high[1])
                for z in (low[2], high[2])
            ],
            dtype=np.float64,
        )
        camera_points = (pose_cam[:3, :3] @ corners.T).T + pose_cam[:3, 3]
        valid = camera_points[:, 2] > 1e-4
        if np.count_nonzero(valid) < 4:
            return None
        K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        x = K[0, 0] * camera_points[valid, 0] / camera_points[valid, 2] + K[0, 2]
        y = K[1, 1] * camera_points[valid, 1] / camera_points[valid, 2] + K[1, 2]
        height, width = image_shape
        x1 = int(np.clip(np.floor(np.min(x)), 0, width - 1))
        y1 = int(np.clip(np.floor(np.min(y)), 0, height - 1))
        x2 = int(np.clip(np.ceil(np.max(x)), 0, width - 1))
        y2 = int(np.clip(np.ceil(np.max(y)), 0, height - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _bbox_iou(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        first_area = max(0, first[2] - first[0] + 1) * max(
            0, first[3] - first[1] + 1
        )
        second_area = max(0, second[2] - second[0] + 1) * max(
            0, second[3] - second[1] + 1
        )
        union = first_area + second_area - intersection
        return float(intersection) / float(union) if union > 0 else 0.0

    @staticmethod
    def _bbox_intersection_over_min(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        first_area = max(0, first[2] - first[0] + 1) * max(
            0, first[3] - first[1] + 1
        )
        second_area = max(0, second[2] - second[0] + 1) * max(
            0, second[3] - second[1] + 1
        )
        return float(intersection) / float(max(1, min(first_area, second_area)))

    async def _publish(
        self,
        pkt: TimeStampedPacket,
        estimate: FoundationPoseEstimate,
        world_camera: np.ndarray,
        *,
        quality_metrics: dict[str, Any] | None = None,
    ) -> None:
        pose_world = world_camera @ estimate.pose_cam
        det = DetectionResult(
            t_host=float(pkt.t_host),
            source=f"{pkt.source}/foundationpose/{estimate.object_name}",
            pose_world=pose_world,
            score=float(estimate.score),
            detector="foundationpose",
            object_name=estimate.object_name,
            sync_group=pkt.sync_group,
            quality_metrics=quality_metrics or None,
        )
        await self.bus.publish(self.cfg.output_topic, det)
        await self.bus.publish(
            self.cfg.status_topic,
            {
                "t_host": float(pkt.t_host),
                "source": pkt.source,
                "object_name": estimate.object_name,
                "mode": estimate.mode,
                "pose_cam": estimate.pose_cam,
                "quality_metrics": quality_metrics or None,
                "scoring": asdict(estimate.scoring) if estimate.scoring else None,
            },
        )

    async def _publish_visualization_frame(
        self,
        pkt: TimeStampedPacket,
        color_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        estimates: list[
            tuple[FoundationPoseEstimate, np.ndarray, np.ndarray | None]
        ],
    ) -> None:
        image = np.asarray(color_bgr).copy()
        K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        palette = ((0, 255, 255), (255, 80, 255), (80, 220, 80), (255, 180, 40))
        edges = (
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        )
        for object_index, (estimate, pose_world, init_mask) in enumerate(estimates):
            object_color = palette[object_index % len(palette)]
            if (
                init_mask is not None
                and np.asarray(init_mask).shape[:2] == image.shape[:2]
            ):
                selected = np.asarray(init_mask) > 0
                overlay_color = np.asarray(object_color, dtype=np.float32)
                image[selected] = (
                    image[selected].astype(np.float32) * 0.55
                    + overlay_color * 0.45
                ).astype(np.uint8)

            low, high = self.runner.model_bounds(estimate.object_name)
            corners = np.array(
                [
                    [x, y, z]
                    for x in (low[0], high[0])
                    for y in (low[1], high[1])
                    for z in (low[2], high[2])
                ],
                dtype=np.float64,
            )
            pose = np.asarray(estimate.pose_cam, dtype=np.float64)
            points_camera = (pose[:3, :3] @ corners.T).T + pose[:3, 3]
            valid = points_camera[:, 2] > 1e-5
            pixels = np.zeros((8, 2), dtype=np.float64)
            pixels[valid, 0] = (
                K[0, 0] * points_camera[valid, 0] / points_camera[valid, 2]
                + K[0, 2]
            )
            pixels[valid, 1] = (
                K[1, 1] * points_camera[valid, 1] / points_camera[valid, 2]
                + K[1, 2]
            )
            for first, second in edges:
                if valid[first] and valid[second]:
                    cv2.line(
                        image,
                        tuple(np.round(pixels[first]).astype(int)),
                        tuple(np.round(pixels[second]).astype(int)),
                        object_color,
                        3,
                        cv2.LINE_AA,
                    )

            axis_length = max(float(np.max(high - low)) * 0.75, 0.03)
            axes = np.array(
                [
                    [0, 0, 0],
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, axis_length],
                ],
                dtype=np.float64,
            )
            axes_camera = (pose[:3, :3] @ axes.T).T + pose[:3, 3]
            axes_valid = axes_camera[:, 2] > 1e-5
            axes_px = np.zeros((4, 2), dtype=np.float64)
            axes_px[axes_valid, 0] = (
                K[0, 0] * axes_camera[axes_valid, 0] / axes_camera[axes_valid, 2]
                + K[0, 2]
            )
            axes_px[axes_valid, 1] = (
                K[1, 1] * axes_camera[axes_valid, 1] / axes_camera[axes_valid, 2]
                + K[1, 2]
            )
            for axis_index, axis_color in enumerate(
                ((0, 0, 255), (0, 255, 0), (255, 0, 0)), start=1
            ):
                if axes_valid[0] and axes_valid[axis_index]:
                    cv2.line(
                        image,
                        tuple(np.round(axes_px[0]).astype(int)),
                        tuple(np.round(axes_px[axis_index]).astype(int)),
                        axis_color,
                        4,
                        cv2.LINE_AA,
                    )

            world_xyz = np.asarray(pose_world, dtype=np.float64)[:3, 3]
            label = (
                f"{estimate.object_name} [{estimate.mode}] "
                f"world=({world_xyz[0]:+.3f},{world_xyz[1]:+.3f},"
                f"{world_xyz[2]:+.3f})m"
            )
            cv2.putText(
                image,
                label,
                (12, 34 + object_index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                object_color,
                2,
                cv2.LINE_AA,
            )
        await self.bus.publish(
            self._visualization_topic,
            {
                "t_host": float(pkt.t_host),
                "source": pkt.source,
                "stream": "foundationpose/all",
                "objects": [item[0].object_name for item in estimates],
                "image_bgr": image,
            },
        )

    def _lookup_extrinsic(self, source: str) -> np.ndarray | None:
        serial = next(
            (part for part in str(source).split("/") if part.isdigit() and len(part) >= 6),
            None,
        )
        for key in (source, serial, f"realsense/{serial}" if serial else None):
            if key and key in self._T_world_camera:
                return self._T_world_camera[key]
        return None

    def _resolve_world_camera(self, pkt: TimeStampedPacket) -> np.ndarray | None:
        fixed = self._lookup_extrinsic(pkt.source)
        if fixed is not None:
            return fixed
        if not str(pkt.source).startswith("quest") or not isinstance(pkt.payload, dict):
            return None

        world_tracking = pkt.payload.get("T_world_tracking")
        if world_tracking is None:
            world_tracking = self._T_world_tracking
        tracking_camera = pkt.payload.get(
            "T_tracking_cam", pkt.payload.get("T_world_cam")
        )
        if world_tracking is None or tracking_camera is None:
            return None
        try:
            world_tracking = np.asarray(world_tracking, dtype=np.float64).reshape(4, 4)
            tracking_camera = np.asarray(tracking_camera, dtype=np.float64).reshape(4, 4)
        except (TypeError, ValueError):
            return None
        # Input adapters guarantee the project-wide Y-down convention.
        result = world_tracking @ tracking_camera
        return result if np.isfinite(result).all() else None

    @staticmethod
    def _load_mobile_world_extrinsic(path: str | Path | None) -> np.ndarray | None:
        if not path or not Path(path).is_file():
            return None
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        coordinate_system = data.get(
            "coordinate_system", data.get("coord", "y_down")
        )
        value = data.get("T_world_tracking", data.get("T_world_quest"))
        if isinstance(value, dict) and "pos" in value and "rot_xyzw" in value:
            position = np.asarray(value["pos"], dtype=np.float64).reshape(3)
            x, y, z, w = np.asarray(value["rot_xyzw"], dtype=np.float64).reshape(4)
            rotation = np.array(
                [
                    [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                    [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                    [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
                ],
                dtype=np.float64,
            )
            result = np.eye(4, dtype=np.float64)
            result[:3, :3] = rotation
            result[:3, 3] = position
            coordinate_system = value.get(
                "coordinate_system", value.get("coord", coordinate_system)
            )
            return transform_to_y_down(result, coordinate_system)
        try:
            result = np.asarray(value, dtype=np.float64).reshape(4, 4)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(result).all():
            return None
        return transform_to_y_down(result, coordinate_system)

    def _load_extrinsics(self, path: str | Path | None) -> None:
        if not path or not Path(path).is_file():
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        coordinate_system = data.get(
            "coordinate_system", data.get("coord", "y_down")
        )
        for key, value in (data.get("T_world_cam") or {}).items():
            self._T_world_camera[str(key)] = transform_to_y_down(
                np.asarray(value, dtype=np.float64).reshape(4, 4),
                coordinate_system,
            )
