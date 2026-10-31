# adapters/realsense_adapter.py
# pyright: reportMissingTypeArgument=false

import asyncio
import time
from typing import Optional, Dict, Any, cast

import numpy as np

from common.structured_log import slog

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pyrealsense2 as _rs
except ImportError:
    _rs = None

rs = cast(Any, _rs)

from core.types import TimeStampedPacket


class RealSenseAdapter:
    def __init__(
        self,
        loop,
        queue: asyncio.Queue,
        cfg,
        source_name: str = "realsense",
        *,
        show_preview: bool = False,
        preview_window_name: str = "realsense",
    ):
        """
        cfg: BackendConfig.camera
        """
        self.loop = loop
        self.queue = queue
        self.cfg = cfg
        self.source_name = source_name

        self._show_preview = show_preview
        self._preview_window_name = preview_window_name
        self._preview_enabled = show_preview

        self._preview_task: Optional[asyncio.Task] = None
        self._preview_image: Optional[np.ndarray] = None

        self._no_frame_count = 0
        self._last_no_frame_log_t = 0.0
        self._restart_in_progress = False

        self._wait_timeout_ms = 200

        self._running = False
        self._task: Optional[asyncio.Task] = None

        self._pipeline = None
        self._config = None
        self._align = None

        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        self._cached_camera_metadata: Optional[Dict[str, Any]] = None
        self._depth_scale_m: float = 0.001
        self._last_align_error_log_t = 0.0

    def start(self):
        if rs is None:
            raise ImportError("pyrealsense2 is not installed in this env")

        slog(
            component=self.__class__.__name__,
            event="start",
            source=self.source_name,
            width=self.cfg.width,
            height=self.cfg.height,
            fps=self.cfg.fps,
            serial=self.cfg.serial,
            use_color=self.cfg.use_color,
            use_depth=self.cfg.use_depth,
            show_preview=self._show_preview,
        )

        if self._show_preview and cv2 is None:
            slog(
                component=self.__class__.__name__,
                event="preview_disabled",
                source=self.source_name,
                reason="opencv_not_installed",
            )
            self._preview_enabled = False

        # 初始化 RealSense pipeline
        self._pipeline = rs.pipeline()
        self._config = rs.config()

        if self.cfg.serial:
            self._config.enable_device(self.cfg.serial)

        if self.cfg.use_color:
            self._config.enable_stream(
                rs.stream.color,
                self.cfg.width,
                self.cfg.height,
                rs.format.bgr8,
                self.cfg.fps,
            )
        if self.cfg.use_depth:
            self._config.enable_stream(
                rs.stream.depth,
                self.cfg.width,
                self.cfg.height,
                rs.format.z16,
                self.cfg.fps,
            )

        profile = self._pipeline.start(self._config)
        self._configure_runtime_profile(profile)
        self._running = True
        self._task = self.loop.create_task(self._run_loop())

        if self._preview_enabled and cv2 is not None:
            slog(
                component=self.__class__.__name__,
                event="preview_start",
                source=self.source_name,
                window=self._preview_window_name,
            )
            self._preview_task = self.loop.create_task(self._preview_loop())

    def _build_rs_config(self):
        assert rs is not None
        cfg = rs.config()
        if self.cfg.serial:
            cfg.enable_device(self.cfg.serial)

        if self.cfg.use_color:
            cfg.enable_stream(
                rs.stream.color,
                self.cfg.width,
                self.cfg.height,
                rs.format.bgr8,
                self.cfg.fps,
            )

        if self.cfg.use_depth:
            cfg.enable_stream(
                rs.stream.depth,
                self.cfg.width,
                self.cfg.height,
                rs.format.z16,
                self.cfg.fps,
            )

        return cfg

    @staticmethod
    def _normalize_preset_name(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    @staticmethod
    def _enum_preset_map(enum_obj: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for attr in dir(enum_obj):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(enum_obj, attr)
            except Exception:
                continue
            if callable(val):
                continue
            out["".join(ch for ch in attr.lower() if ch.isalnum())] = val
        return out

    def _resolve_visual_preset(self, device: Any, preset_name: str) -> tuple[Any | None, str, str]:
        assert rs is not None

        requested = self._normalize_preset_name(preset_name)
        if requested in {"", "default"}:
            return None, "default", requested

        product_line = ""
        device_name = ""
        try:
            product_line = str(device.get_info(rs.camera_info.product_line))
        except Exception:
            pass
        try:
            device_name = str(device.get_info(rs.camera_info.name))
        except Exception:
            pass

        is_l500 = "L500" in product_line.upper() or "L515" in device_name.upper()
        if is_l500 and hasattr(rs, "l500_visual_preset"):
            enum_obj = rs.l500_visual_preset
            family = "l500"
            aliases = {
                "highaccuracy": "maxrange",
                "highdensity": "shortrange",
                "noambientlight": "noambientlight",
                "lowambientlight": "lowambientlight",
                "maxrange": "maxrange",
                "shortrange": "shortrange",
                "automatic": "automatic",
            }
        else:
            enum_obj = getattr(rs, "rs400_visual_preset", None)
            family = "rs400"
            aliases = {
                "highaccuracy": "highaccuracy",
                "highdensity": "highdensity",
                "mediumdensity": "mediumdensity",
                "default": "default",
                "custom": "custom",
                "hand": "hand",
            }

        if enum_obj is None:
            return None, family, requested

        preset_map = self._enum_preset_map(enum_obj)
        key = aliases.get(requested, requested)
        return preset_map.get(key), family, key

    def _cache_intrinsics(self, profile: Any) -> None:
        assert rs is not None

        # Cache intrinsics for pose estimation (e.g., ArUco and RGB-D VO).
        try:
            if self.cfg.use_color:
                stream = profile.get_stream(rs.stream.color)
                vsp = stream.as_video_stream_profile()
            else:
                stream = profile.get_stream(rs.stream.depth)
                vsp = stream.as_video_stream_profile()

            intr = vsp.get_intrinsics()
            K = np.array(
                [
                    [intr.fx, 0.0, intr.ppx],
                    [0.0, intr.fy, intr.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            dist = np.array(getattr(intr, "coeffs", [0, 0, 0, 0, 0])[:5], dtype=np.float64)
            self._camera_matrix = K
            self._dist_coeffs = dist
            self._cached_camera_metadata = None

            slog(
                component=self.__class__.__name__,
                event="intrinsics",
                source=self.source_name,
                fx=float(intr.fx),
                fy=float(intr.fy),
                ppx=float(intr.ppx),
                ppy=float(intr.ppy),
            )
        except Exception as e:
            slog(
                component=self.__class__.__name__,
                event="intrinsics_error",
                source=self.source_name,
                reason=repr(e),
            )

    def _configure_runtime_profile(self, profile: Any) -> None:
        assert rs is not None

        self._cached_camera_metadata = None
        self._align = None

        device = profile.get_device()
        depth_sensor = None
        try:
            depth_sensor = device.first_depth_sensor()
        except Exception:
            depth_sensor = None

        if depth_sensor is not None:
            # Hardware sync configuration
            if hasattr(self.cfg, "sync_mode") and self.cfg.sync_mode in [1, 2]:
                if depth_sensor.supports(rs.option.inter_cam_sync_mode):
                    depth_sensor.set_option(rs.option.inter_cam_sync_mode, self.cfg.sync_mode)
                    slog(
                        component=self.__class__.__name__,
                        event="sync_configured",
                        source=self.source_name,
                        mode="Master" if self.cfg.sync_mode == 1 else "Slave",
                    )

            # Visual preset configuration (RS400 + L500/L515).
            preset_name = str(getattr(self.cfg, "visual_preset", "Default") or "Default")
            if self._normalize_preset_name(preset_name) != "default":
                if depth_sensor.supports(rs.option.visual_preset):
                    preset_val, family, resolved = self._resolve_visual_preset(device, preset_name)
                    if preset_val is not None:
                        depth_sensor.set_option(rs.option.visual_preset, preset_val)
                        slog(
                            component=self.__class__.__name__,
                            event="preset_configured",
                            source=self.source_name,
                            preset=preset_name,
                            preset_family=family,
                            resolved_preset=resolved,
                        )
                    else:
                        slog(
                            component=self.__class__.__name__,
                            event="preset_unsupported",
                            source=self.source_name,
                            preset=preset_name,
                            preset_family=family,
                        )

            if self.cfg.use_depth:
                try:
                    self._depth_scale_m = float(depth_sensor.get_depth_scale())
                except Exception:
                    self._depth_scale_m = 0.001

        if self.cfg.use_color and self.cfg.use_depth:
            self._align = rs.align(rs.stream.color)
            slog(
                component=self.__class__.__name__,
                event="align_enabled",
                source=self.source_name,
                target="color",
            )

        self._cache_intrinsics(profile)

    def _restart_pipeline_blocking(self) -> None:
        assert rs is not None
        if self._pipeline is None:
            self._pipeline = rs.pipeline()

        try:
            self._pipeline.stop()
        except Exception:
            pass

        self._config = self._build_rs_config()
        profile = self._pipeline.start(self._config)
        self._configure_runtime_profile(profile)

    def _update_preview_image(self, frames: Dict[str, Any]) -> None:
        if not self._preview_enabled or cv2 is None:
            return

        color = frames.get("color")
        depth = frames.get("depth")

        preview = None
        if color is not None and depth is not None:
            depth_u16 = depth
            depth_mm = depth_u16.astype(np.float32)
            depth_norm = np.clip(depth_mm / 4000.0, 0.0, 1.0)
            depth_u8 = (depth_norm * 255.0).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
            if depth_vis.shape[:2] != color.shape[:2]:
                depth_vis = cv2.resize(depth_vis, (color.shape[1], color.shape[0]))
            preview = np.hstack([color, depth_vis])
        elif color is not None:
            preview = color
        elif depth is not None:
            depth_u16 = depth
            depth_mm = depth_u16.astype(np.float32)
            depth_norm = np.clip(depth_mm / 4000.0, 0.0, 1.0)
            depth_u8 = (depth_norm * 255.0).astype(np.uint8)
            preview = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

        if preview is None:
            return

        self._preview_image = preview


    async def _preview_loop(self) -> None:
        if cv2 is None:
            return

        while self._running and self._preview_enabled:
            img = self._preview_image
            if img is not None:
                try:
                    cv2.imshow(self._preview_window_name, img)
                    key = cv2.waitKey(1)
                    if key in (ord('q'), ord('Q')):
                        slog(
                            component=self.__class__.__name__,
                            event="preview_disabled",
                            source=self.source_name,
                            reason="user_quit",
                        )
                        self._preview_enabled = False
                        break
                except Exception as e:
                    slog(
                        component=self.__class__.__name__,
                        event="preview_disabled",
                        source=self.source_name,
                        reason=repr(e),
                    )
                    self._preview_enabled = False
                    break

            await asyncio.sleep(1.0 / 30.0)

    async def _run_loop(self):
        # RealSense SDK 是阻塞式的，这里用 run_in_executor 包起来
        loop = asyncio.get_running_loop()
        while self._running:
            if self._pipeline is None:
                await asyncio.sleep(0.05)
                continue

            # Prefer short-timeout wait_for_frames; some systems starve with poll_for_frames.
            try:
                frames = await loop.run_in_executor(
                    None,
                    self._pipeline.wait_for_frames,
                    self._wait_timeout_ms,
                )
            except Exception:
                frames = None

            if frames is None:
                self._no_frame_count += 1
                now = time.monotonic()
                if now - self._last_no_frame_log_t > 2.0:
                    slog(
                        component=self.__class__.__name__,
                        event="no_frames",
                        source=self.source_name,
                        count=self._no_frame_count,
                    )
                    self._last_no_frame_log_t = now

                # If we starve for too long, attempt a restart.
                if self._no_frame_count >= 300 and not self._restart_in_progress:
                    self._restart_in_progress = True
                    slog(
                        component=self.__class__.__name__,
                        event="pipeline_restart_start",
                        source=self.source_name,
                        count=self._no_frame_count,
                    )
                    try:
                        await loop.run_in_executor(None, self._restart_pipeline_blocking)
                        self._no_frame_count = 0
                        slog(
                            component=self.__class__.__name__,
                            event="pipeline_restart_ok",
                            source=self.source_name,
                        )
                    except Exception as e:
                        slog(
                            component=self.__class__.__name__,
                            event="pipeline_restart_failed",
                            source=self.source_name,
                            reason=repr(e),
                        )
                        await asyncio.sleep(1.0)
                    finally:
                        self._restart_in_progress = False

                await asyncio.sleep(0.01)
                continue

            self._no_frame_count = 0

            if self._align is not None:
                try:
                    frames = self._align.process(frames)
                except Exception as e:
                    now = time.monotonic()
                    if now - self._last_align_error_log_t > 2.0:
                        slog(
                            component=self.__class__.__name__,
                            event="align_error",
                            source=self.source_name,
                            reason=repr(e),
                        )
                        self._last_align_error_log_t = now

            data: Dict[str, Any] = {}
            ts = frames.get_timestamp() * 1e-3  # ms -> s

            if self.cfg.use_color:
                color_frame = frames.get_color_frame()
                if color_frame:
                    # IMPORTANT: copy() to detach from pyrealsense frame buffer.
                    # Holding a view can prevent frames from being released and starve the pipeline.
                    color_image = np.asanyarray(color_frame.get_data()).copy()
                    data["color"] = color_image

            if self.cfg.use_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data()).copy()
                    data["depth"] = depth_image

            if self._preview_enabled:
                self._update_preview_image(data)

            # Only compute K/dist tolist() once and cache it.
            if self._cached_camera_metadata is None:
                self._cached_camera_metadata = {
                    "K": self._camera_matrix.tolist() if self._camera_matrix is not None else None,
                    "dist": self._dist_coeffs.tolist() if self._dist_coeffs is not None else None,
                    "depth_scale_m": float(self._depth_scale_m),
                    "depth_aligned_to": "color" if self._align is not None else None,
                }

            arrival_time = time.monotonic()
            frame_index = int(frames.get_frame_number())
            pkt = TimeStampedPacket(
                t_host=arrival_time,
                source=self.source_name,
                payload={
                    "t_device": ts,
                    "camera": self._cached_camera_metadata,
                    "frames": data,
                    "frame_index": frame_index,
                },
                t_capture=ts,
                t_arrival=arrival_time,
                frame_index=frame_index,
            )
            await self.queue.put(pkt)

    async def stop(self):
        slog(
            component=self.__class__.__name__,
            event="stop",
            source=self.source_name,
        )
        self._running = False

        if self._preview_task is not None:
            self._preview_task.cancel()
            try:
                await self._preview_task
            except asyncio.CancelledError:
                pass

        if self._task is not None:
            await self._task
        if self._pipeline is not None:
            self._pipeline.stop()

        if self._preview_enabled and cv2 is not None:
            try:
                cv2.destroyWindow(self._preview_window_name)
            except Exception:
                pass

        slog(
            component=self.__class__.__name__,
            event="preview_stop",
            source=self.source_name,
            window=self._preview_window_name,
        )
