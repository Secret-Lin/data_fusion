# managers/sensor_manager.py

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Optional

from common.structured_log import slog
from common.replay_extrinsics import load_fixed_extrinsics

from core.event_bus import EventBus
from core.frame_sync import FrameGroupAssigner
from core.types import TimeStampedPacket

from adapters.hl2_adapter import HL2MultiSensorAdapter
from adapters.realsense_adapter import RealSenseAdapter
from adapters.quest_adapter import QuestRGBDAdapter

from config import BackendConfig, CameraConfig


class SensorManager:
    def __init__(
        self,
        bus: EventBus,
        loop,
        cfg: BackendConfig,
        *,
        enable_hl2: bool = True,
        enable_realsense: bool = True,
        enable_quest: bool = False,
        hl2_enable_ahat: bool = True,
        hl2_enable_pv: bool = True,
        hl2_enable_imu: bool = True,
        realsense_show_preview: bool = False,
        realsense_cameras: list[CameraConfig] | None = None,
    ):
        self.bus = bus
        self.loop = loop
        self.cfg = cfg

        self.enable_hl2 = enable_hl2
        self.enable_realsense = enable_realsense
        self.enable_quest = enable_quest

        self.hl2_enable_ahat = hl2_enable_ahat
        self.hl2_enable_pv = hl2_enable_pv
        self.hl2_enable_imu = hl2_enable_imu

        self.realsense_show_preview = realsense_show_preview
        self.realsense_cameras = realsense_cameras

        # ---------------- HL2 ----------------
        self.hl2_queue: asyncio.Queue | None = None
        self.hl2_adapter: HL2MultiSensorAdapter | None = None
        if self.enable_hl2:
            self.hl2_queue = asyncio.Queue()
            self.hl2_adapter = HL2MultiSensorAdapter(
                loop=loop,
                out_queue=self.hl2_queue,
                host=cfg.hl2.host,
                calibration_path=cfg.hl2.calibration_path,
                enable_ahat=self.hl2_enable_ahat,
                enable_pv=self.hl2_enable_pv,
                enable_imu=self.hl2_enable_imu,
                source_name="hl2",
            )

        # ---------------- RealSense (single or multiple) ----------------
        self.realsense_queues: dict[str, asyncio.Queue] = {}
        self.realsense_adapters: dict[str, RealSenseAdapter] = {}
        self.realsense_topics: dict[str, str] = {}
        self._fixed_world_camera = {}

        if self.enable_realsense:
            cameras = self.realsense_cameras
            if cameras is None:
                cameras = getattr(cfg, "realsense_cameras", None)
            if not cameras:
                cameras = [cfg.camera]

            missing_serials = [cam.name or "<unnamed>" for cam in cameras if not cam.serial]
            if missing_serials:
                raise RuntimeError(
                    "Real-time fixed RealSense cameras require explicit serial numbers "
                    "so each stream can be matched to T_world_cam; missing for: "
                    + ", ".join(missing_serials)
                )
            required_serials = [cam.serial for cam in cameras]
            extrinsic_path = Path(str(cfg.calibration.tag_cube.result_path or ""))
            if not str(cfg.calibration.tag_cube.result_path or "").strip():
                raise RuntimeError(
                    "Real-time RealSense input requires a fixed-camera extrinsic file; "
                    "cfg.calibration.tag_cube.result_path is empty"
                )
            self._fixed_world_camera = load_fixed_extrinsics(
                extrinsic_path, required_serials
            )

            for cam in cameras:
                cam_id = cam.serial or cam.name or "realsense"
                source = f"realsense/{cam_id}"
                topic = f"realsense/{cam_id}/stream"

                # Bounded reliable queue applies backpressure instead of silently losing frames.
                q = asyncio.Queue(maxsize=8)
                adapter = RealSenseAdapter(
                    loop,
                    q,
                    cam,
                    source_name=source,
                    show_preview=self.realsense_show_preview,
                    preview_window_name=source,
                )
                self.realsense_queues[source] = q
                self.realsense_adapters[source] = adapter
                self.realsense_topics[source] = topic
        self._realsense_sync = FrameGroupAssigner(tolerance_seconds=0.02)
        self.quest_queue: asyncio.Queue | None = None
        self.quest_adapter: QuestRGBDAdapter | None = None
        if self.enable_quest:
            self.quest_queue = asyncio.Queue(maxsize=max(1, int(cfg.quest.queue_size)))
            self.quest_adapter = QuestRGBDAdapter(loop, self.quest_queue, cfg.quest)
        # 如果你已经有 OTS 适配器，也可以在这里初始化
        # self.ots_queue = asyncio.Queue()
        # self.ots_adapter = OTSAdapter(...)

    def start(self):
        slog(
            component=self.__class__.__name__,
            event="start",
            enable_hl2=self.enable_hl2,
            enable_realsense=self.enable_realsense,
            hl2_enable_ahat=self.hl2_enable_ahat,
            hl2_enable_pv=self.hl2_enable_pv,
            hl2_enable_imu=self.hl2_enable_imu,
            realsense_show_preview=self.realsense_show_preview,
            enable_quest=self.enable_quest,
        )
        # 启动 HL2
        if self.enable_hl2 and self.hl2_adapter is not None and self.hl2_queue is not None:
            self.hl2_adapter.start()
            self.loop.create_task(self._consume_hl2_queue(self.hl2_queue))

        # 启动 RealSense
        if self.enable_realsense:
            # Keep a legacy alias for downstream consumers (first camera only).
            legacy_source = next(iter(self.realsense_adapters.keys()), None)
            for source, adapter in list(self.realsense_adapters.items()):
                q = self.realsense_queues[source]
                topic = self.realsense_topics[source]
                alias_topic = "realsense_stream" if source == legacy_source and topic != "realsense_stream" else None
                try:
                    adapter.start()
                except Exception as e:
                    slog(
                        component=self.__class__.__name__,
                        event="realsense_start_failed",
                        source=source,
                        reason=repr(e),
                    )
                    # Drop this camera and continue with others.
                    self.realsense_adapters.pop(source, None)
                    self.realsense_queues.pop(source, None)
                    self.realsense_topics.pop(source, None)
                    continue

                self.loop.create_task(self._consume_realsense_queue(source, q, topic=topic, alias_topic=alias_topic))

        if self.quest_adapter is not None and self.quest_queue is not None:
            self.quest_adapter.start()
            self.loop.create_task(self._consume_quest_queue(self.quest_queue))

        # 如果有 OTS：
        # self.ots_adapter.start()
        # self.loop.create_task(self._consume_queue(self.ots_queue, topic="ots/stream"))

    async def _consume_realsense_queue(
        self,
        source: str,
        q: asyncio.Queue,
        *,
        topic: str,
        alias_topic: str | None = None,
    ):
        while True:
            pkt: TimeStampedPacket = await q.get()
            if pkt is None:
                continue
            # Cross-device grouping must use the shared PC monotonic clock;
            # RealSense device time and Quest time are different clock domains.
            sync_group = self._realsense_sync.assign(source, float(pkt.t_arrival))
            camera_id = source.split("/", 1)[1] if "/" in source else source
            payload = dict(pkt.payload) if isinstance(pkt.payload, dict) else {}
            payload["T_world_cam"] = self._fixed_world_camera[camera_id]
            payload["coordinate_system"] = "y_down"
            pkt = replace(pkt, payload=payload, sync_group=sync_group)
            await self.bus.publish(topic, pkt)
            if alias_topic is not None:
                await self.bus.publish(alias_topic, pkt)

    async def _consume_hl2_queue(self, q: asyncio.Queue):
        while True:
            pkt: TimeStampedPacket = await q.get()
            if pkt is None:
                continue
            stream = None
            if isinstance(pkt.payload, dict):
                stream = pkt.payload.get("stream")

            if stream == "ahat":
                topic = "hl2/ahat"
            elif stream == "pv":
                topic = "hl2/pv"
            elif stream == "imu_accel":
                topic = "hl2/imu/accel"
            elif stream == "imu_gyro":
                topic = "hl2/imu/gyro"
            elif stream == "imu_mag":
                topic = "hl2/imu/mag"
            elif isinstance(stream, str) and stream.endswith("_error"):
                topic = "hl2/error"
            else:
                topic = "hl2/stream"

            await self.bus.publish(topic, pkt)

    async def _consume_quest_queue(self, q: asyncio.Queue) -> None:
        while True:
            pkt: TimeStampedPacket = await q.get()
            if pkt is None:
                continue
            sync_group = self._realsense_sync.assign(pkt.source, float(pkt.t_arrival))
            pkt = replace(pkt, sync_group=sync_group)
            await self.bus.publish(f"{pkt.source}/stream", pkt)
