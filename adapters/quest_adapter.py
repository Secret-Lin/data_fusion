from __future__ import annotations

import asyncio
import json
import socket
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from common.structured_log import slog
from config import QuestConfig
from core.coordinate_frames import transform_to_y_down
from core.types import TimeStampedPacket
from adapters.quest_init import QuestTagInitializer, save_quest_extrinsic
from standalone import record_realsense_and_quest_rgbd_dpt2 as quest_protocol


class QuestRGBDAdapter:
    """TCP receiver compatible with the validated QCAM + DPT1/DPT2 recorder protocol."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        out_queue: asyncio.Queue,
        cfg: QuestConfig,
    ) -> None:
        self.loop = loop
        self.out_queue = out_queue
        self.cfg = cfg
        self.source_name = cfg.source_name
        self._state = "idle"
        self._streaming = False
        self._initializer: QuestTagInitializer | None = None
        self._world_tracking = self._load_existing_extrinsic(cfg.world_extrinsic_path)
        if self._world_tracking is not None:
            self._state = "ready"
        if not cfg.require_sensor_command and self._world_tracking is not None:
            self._state = "sensor"
            self._streaming = True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._server: socket.socket | None = None
        self._connection: socket.socket | None = None
        self._alignment_cfg = SimpleNamespace(
            depth_valid_min_m=cfg.depth_valid_min_m,
            depth_valid_max_m=cfg.depth_valid_max_m,
            alignment_splat_radius=cfg.alignment_splat_radius,
            fill_depth_holes=cfg.fill_depth_holes,
            depth_hole_fill_kernel=cfg.depth_hole_fill_kernel,
            alignment_rotate_180=cfg.alignment_rotate_180,
            alignment_flip_horizontal=cfg.alignment_flip_horizontal,
            alignment_flip_vertical=cfg.alignment_flip_vertical,
            allow_legacy_resize_fallback=cfg.allow_legacy_resize_fallback,
        )

    @property
    def streaming(self) -> bool:
        return self._streaming

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, name="QuestRGBDAdapter", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        for sock in (self._connection, self._server):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _handle_command(self, connection: socket.socket, command: dict[str, Any]) -> None:
        name = str(command.get("cmd", ""))
        if name == "sensor_toggle":
            requested = bool(command.get("active", False))
            self._streaming = requested and self._world_tracking is not None
            self._state = "sensor" if self._streaming else ("ready" if self._world_tracking is not None else "idle")
            slog(component=self.__class__.__name__, event="sensor_toggle", active=self._streaming)
        elif name == "init_start":
            self._streaming = False
            if not self.cfg.init_cube_model_path:
                quest_protocol.quest_send_json_packet(connection, {"type": "init_progress", "progress": -1})
                return
            self._world_tracking = None
            self._initializer = QuestTagInitializer(
                self.cfg.init_cube_model_path,
                self.cfg.init_min_samples,
                self.cfg.init_reference_extrinsics_path,
            )
            self._state = "init"
            quest_protocol.quest_send_json_packet(connection, {"type": "init_progress", "progress": 0})
        elif name == "init_end":
            result = None if self._initializer is None else self._initializer.finish()
            if result is None:
                self._state = "idle"
                quest_protocol.quest_send_json_packet(connection, {"type": "init_progress", "progress": -1})
            else:
                self._world_tracking = result
                self._state = "ready"
                save_quest_extrinsic(
                    self.cfg.init_output_path,
                    result,
                    len(self._initializer.samples),
                    quality={"reference_source": self._initializer.reference_source},
                )
                quest_protocol.quest_send_json_packet(connection, {"type": "init_progress", "progress": 100})
            self._initializer = None

    @staticmethod
    def _load_existing_extrinsic(path: str) -> np.ndarray | None:
        if not path:
            return None
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            coordinate_system = data.get(
                "coordinate_system", data.get("coord", "y_down")
            )
            value = data.get("T_world_tracking", data.get("T_world_quest"))
            if isinstance(value, list):
                transform = np.asarray(value, dtype=np.float64).reshape(4, 4)
                return transform_to_y_down(transform, coordinate_system)
            if isinstance(value, dict):
                position = np.asarray(value.get("pos", value.get("position")), np.float64)
                x, y, z, w = np.asarray(value.get("rot_xyzw", value.get("rotation_xyzw")), np.float64)
                result = np.eye(4)
                result[:3, :3] = np.array([
                    [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                    [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                    [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
                ])
                result[:3, 3] = position
                value_coordinate_system = value.get(
                    "coordinate_system", value.get("coord", coordinate_system)
                )
                return transform_to_y_down(result, value_coordinate_system)
        except Exception:
            return None
        return None

    @staticmethod
    def _decode_color_meta(payload: bytes) -> tuple[dict[str, Any], np.ndarray]:
        meta = quest_protocol.quest_parse_header(payload[: quest_protocol.QUEST_HEADER_SIZE])
        start = quest_protocol.QUEST_HEADER_SIZE
        end = start + int(meta["jpgLength"])
        if end > len(payload):
            raise ValueError("Quest JPG length exceeds payload")
        color = cv2.imdecode(np.frombuffer(payload[start:end], np.uint8), cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError("Quest JPG decode failed")
        return meta, color

    def _decode_frame(self, payload: bytes, arrival_time: float) -> TimeStampedPacket:
        meta, color = self._decode_color_meta(payload)
        jpg_start = quest_protocol.QUEST_HEADER_SIZE
        jpg_end = jpg_start + int(meta["jpgLength"])
        if jpg_end > len(payload):
            raise ValueError("Quest JPG length exceeds payload")
        depth_info = None
        if len(payload) > jpg_end:
            depth_info = quest_protocol.quest_try_parse_depth_section(
                payload,
                jpg_end,
                frame_id=int(meta["frameId"]),
                valid_min_m=self.cfg.depth_valid_min_m,
                valid_max_m=self.cfg.depth_valid_max_m,
                log_every_n_frames=0,
            )
        depth_m, depth_mode = quest_protocol.quest_align_depth_to_rgb(
            depth_info, meta, self._alignment_cfg
        )
        depth_mm = quest_protocol.quest_depth_m_to_uint16_mm(
            depth_m,
            self.cfg.depth_valid_min_m,
            self.cfg.depth_valid_max_m,
            shape_if_none=color.shape[:2],
        )
        intrinsics = meta["intrinsics"]
        position = np.asarray(meta["cameraPose"]["position"], dtype=np.float64)
        x, y, z, w = np.asarray(meta["cameraPose"]["rotation"], dtype=np.float64)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.array(
            [[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
             [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
             [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]],
            dtype=np.float64,
        )
        transform[:3, 3] = position
        transform = transform_to_y_down(transform, "y_up")
        if self._world_tracking is None:
            raise RuntimeError(
                "Quest frame cannot enter the project without an external pose: "
                "run INIT successfully or configure quest.world_extrinsic_path"
            )
        world_camera = self._world_tracking @ transform
        frame_index = int(meta["frameId"])
        return TimeStampedPacket(
            t_host=arrival_time,
            source=self.source_name,
            payload={
                "t_device": float(meta["timestampMs"]) * 1e-3,
                "camera": {
                    "K": [[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]],
                    "dist": [0.0] * 5,
                    "depth_scale_m": 0.001,
                    "depth_aligned_to": "color",
                },
                "frames": {"color": color, "depth": depth_mm},
                "T_tracking_cam": transform,
                "T_world_cam": world_camera,
                "frame_index": frame_index,
                "depth_mode": depth_mode,
                "T_world_tracking": self._world_tracking,
                "coordinate_system": "y_down",
            },
            t_capture=arrival_time,
            t_arrival=arrival_time,
            frame_index=frame_index,
        )

    def _publish_blocking(self, packet: TimeStampedPacket) -> None:
        future = asyncio.run_coroutine_threadsafe(self.out_queue.put(packet), self.loop)
        future.result()

    def _serve_connection(self, connection: socket.socket) -> None:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.settimeout(1.0)
        self._connection = connection
        while not self._stop.is_set():
            try:
                payload_len = struct.unpack("!i", quest_protocol.quest_recv_exact(connection, 4))[0]
                if payload_len < 4 or payload_len > quest_protocol.QUEST_MAX_FRAME_PAYLOAD_BYTES:
                    raise ValueError(f"invalid Quest payload length: {payload_len}")
                msg_type = struct.unpack("!i", quest_protocol.quest_recv_exact(connection, 4))[0]
                payload = quest_protocol.quest_recv_exact(connection, payload_len - 4)
                if msg_type == 2:
                    self._handle_command(connection, json.loads(payload.decode("utf-8")))
                    continue
                if msg_type != 1:
                    raise ValueError(f"unknown Quest message type: {msg_type}")
                if self._state == "init" and self._initializer is not None:
                    meta, color = self._decode_color_meta(payload)
                    meta["coordinate_system"] = "y_up"
                    progress = self._initializer.add_frame(color, meta)
                    quest_protocol.quest_send_json_packet(
                        connection, {"type": "init_progress", "progress": progress}
                    )
                elif self._streaming:
                    self._publish_blocking(self._decode_frame(payload, time.monotonic()))
            except socket.timeout:
                continue

    def _worker(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                self._server = server
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self.cfg.host, self.cfg.port))
                server.listen(1)
                server.settimeout(0.5)
                slog(component=self.__class__.__name__, event="listening", host=self.cfg.host, port=self.cfg.port)
                while not self._stop.is_set():
                    try:
                        connection, address = server.accept()
                    except socket.timeout:
                        continue
                    try:
                        slog(component=self.__class__.__name__, event="connected", peer=str(address))
                        with connection:
                            self._serve_connection(connection)
                    except (ConnectionError, OSError, ValueError, json.JSONDecodeError) as exc:
                        if not self._stop.is_set():
                            slog(component=self.__class__.__name__, event="connection_error", error=repr(exc))
                    finally:
                        self._connection = None
        except OSError as exc:
            if not self._stop.is_set():
                slog(component=self.__class__.__name__, event="server_error", error=repr(exc))
