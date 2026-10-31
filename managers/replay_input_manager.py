from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import cv2

from common.structured_log import slog
from common.replay_extrinsics import ensure_replay_extrinsics

from config import BackendConfig
from core.coordinate_frames import normalize_coordinate_system, transform_to_y_down
from core.event_bus import EventBus
from core.types import TimeStampedPacket


class ReplayInputManager:
    """Publish prerecorded per-camera H5 RGBD files as RealSense-like packets."""

    # Number of frames to pre-read in a single H5 batch slice.
    # Batch reading is ~5-10x faster than per-frame indexing.
    _H5_BATCH_SIZE = 100

    def __init__(
        self,
        bus: EventBus,
        cfg: BackendConfig,
        *,
        session_dir: str | Path,
        loop_playback: bool = False,
        fps_override: float | None = None,
        sync_tolerance_s: float | None = None,
    ):
        self.bus = bus
        self.cfg = cfg
        self.session_dir = Path(session_dir)
        self.loop_playback = bool(loop_playback)
        self.fps_override = fps_override
        configured_tolerance = float(getattr(cfg.fusion, "sync_window_s", 0.015))
        self.sync_tolerance_s = max(
            0.0,
            float(sync_tolerance_s) if sync_tolerance_s is not None else min(configured_tolerance, 0.015),
        )

        self._h5_paths = self._discover_h5_files(self.session_dir)
        self._quest_h5_paths = self._discover_quest_h5_files(self.session_dir)
        cube_model_path = (
            cfg.quest.init_cube_model_path
            or cfg.calibration.tag_cube.cube_model_path
        )
        resolved_extrinsics = ensure_replay_extrinsics(
            self.session_dir,
            cube_model_path=cube_model_path,
            min_quest_samples=cfg.quest.init_min_samples,
        )
        self._fixed_world_camera = resolved_extrinsics.fixed
        self._quest_world_tracking = resolved_extrinsics.quest_world_tracking
        if resolved_extrinsics.fixed_path is not None:
            cfg.calibration.tag_cube.result_path = str(resolved_extrinsics.fixed_path)
        if resolved_extrinsics.quest_path is not None:
            cfg.quest.world_extrinsic_path = str(resolved_extrinsics.quest_path)
        for generated_path in resolved_extrinsics.generated:
            slog(
                component=self.__class__.__name__,
                event="replay_extrinsic_generated",
                path=str(generated_path),
                coordinate_system="y_down",
            )
        self._camera_meta = self._load_camera_meta(self.session_dir)
        self._h5_state = self._open_h5_state()

        # Per-camera batch buffer for H5 pre-reading.
        # Key: serial, Value: {t_host, color, depth, start_fi, count}
        self._h5_batch: dict[str, dict] = {}

        # --- Quest replay ---
        self._quest_h5_state = self._open_quest_h5_state()
        self.recorded_fps_by_source = self._recorded_fps_by_source()

    @staticmethod
    def _dataset_fps(dataset: Any) -> float | None:
        if dataset is None or int(dataset.shape[0]) < 2:
            return None
        timestamps = np.asarray(dataset, dtype=np.float64).reshape(-1)
        timestamps = timestamps[np.isfinite(timestamps)]
        if timestamps.size < 2:
            return None
        duration = float(timestamps[-1] - timestamps[0])
        if duration <= 1e-6:
            return None
        # Average captured rate preserves the original replay duration even
        # when there are occasional dropped or jittered source frames.
        fps = float(timestamps.size - 1) / duration
        return fps if np.isfinite(fps) and fps > 0.0 else None

    def _recorded_fps_by_source(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for serial, state in self._h5_state.items():
            fps = self._dataset_fps(state.get("t_host"))
            if fps is not None:
                output[f"realsense/{serial}"] = fps
        for source, state in self._quest_h5_state.items():
            fps = self._dataset_fps(state.get("t_host"))
            if fps is not None:
                output[source] = fps
        return output

    @staticmethod
    def _discover_h5_files(session_dir: Path) -> dict[str, Path]:
        rgbd_root = session_dir / "rgbd"
        if not rgbd_root.exists() or not rgbd_root.is_dir():
            return {}

        files = sorted(rgbd_root.glob("*.h5"))
        return {p.stem: p for p in files}

    # --------------- quest discovery ---------------
    @staticmethod
    def _discover_quest_h5_files(session_dir: Path) -> dict[str, Path]:
        quest_root = session_dir / "quest"
        if not quest_root.exists() or not quest_root.is_dir():
            return {}

        files = sorted(quest_root.glob("*.h5"))
        return {f"quest/{p.stem}": p for p in files}

    def _open_quest_h5_state(self) -> dict[str, dict[str, Any]]:
        state: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        for name, p in self._quest_h5_paths.items():
            try:
                f = h5py.File(str(p), "r")
                if "color" not in f:
                    f.close()
                    errors.append(f"{p}: missing color dataset")
                    slog(
                        component=self.__class__.__name__,
                        event="quest_h5_missing_color",
                        name=name,
                        path=str(p),
                    )
                    continue

                color_ds = f["color"]
                if color_ds.ndim != 4:
                    f.close()
                    errors.append(f"{p}: invalid color shape {color_ds.shape}")
                    continue
                n = int(color_ds.shape[0])
                if n <= 0:
                    f.close()
                    errors.append(f"{p}: color dataset is empty")
                    continue

                required_pose = ("camera_position", "camera_rotation_xyzw")
                missing_pose = [dataset for dataset in required_pose if dataset not in f]
                if missing_pose:
                    f.close()
                    errors.append(
                        f"{p}: missing per-frame Quest tracking pose datasets {missing_pose}; "
                        "T_world_tracking alone cannot locate a moving camera"
                    )
                    continue

                # Detect color format from H5 root attributes
                color_format = str(f.attrs.get("color_format", "")).lower()
                coordinate_system = f.attrs.get(
                    "coordinate_system", f.attrs.get("coord", "y_down")
                )
                normalize_coordinate_system(coordinate_system)

                # Optional datasets
                depth_ds = f["depth"] if "depth" in f else None
                t_host_ds = f["t_host"] if "t_host" in f else None
                frame_index_ds = f["frame_index"] if "frame_index" in f else None
                fx_ds = f["fx"] if "fx" in f else None
                fy_ds = f["fy"] if "fy" in f else None
                cx_ds = f["cx"] if "cx" in f else None
                cy_ds = f["cy"] if "cy" in f else None
                cam_pos_ds = f["camera_position"] if "camera_position" in f else None
                cam_rot_ds = f["camera_rotation_xyzw"] if "camera_rotation_xyzw" in f else None
                if cam_pos_ds.shape != (n, 3) or cam_rot_ds.shape != (n, 4):
                    f.close()
                    errors.append(
                        f"{p}: invalid Quest pose shapes; expected camera_position={(n, 3)} "
                        f"and camera_rotation_xyzw={(n, 4)}, got "
                        f"{cam_pos_ds.shape} and {cam_rot_ds.shape}"
                    )
                    continue

                state[name] = {
                    "file": f,
                    "color": color_ds,
                    "color_format": color_format,
                    "coordinate_system": str(coordinate_system),
                    "depth": depth_ds,
                    "t_host": t_host_ds,
                    "frame_index": frame_index_ds,
                    "fx": fx_ds,
                    "fy": fy_ds,
                    "cx": cx_ds,
                    "cy": cy_ds,
                    "camera_position": cam_pos_ds,
                    "camera_rotation_xyzw": cam_rot_ds,
                    "n_frames": n,
                    "frame_i": 0,
                }
            except Exception as exc:
                errors.append(f"{p}: {exc}")
                slog(
                    component=self.__class__.__name__,
                    event="quest_h5_open_failed",
                    name=name,
                    path=str(p),
                    reason=repr(exc),
                )
        if errors:
            for item in state.values():
                item["file"].close()
            raise RuntimeError("Quest replay validation failed: " + "; ".join(errors))
        return state

    def _next_quest_h5_frame(
        self, name: str
    ) -> tuple[int, float, np.ndarray, np.ndarray, dict[str, Any]] | None:
        """Return (frame_index, t_host, color_bgr, depth, extra_info) or None.
        color_bgr is already converted to BGR per H5 color_format attribute.
        """
        st = self._quest_h5_state.get(name)
        if st is None:
            return None

        fi = int(st["frame_i"])
        n = int(st["n_frames"])
        if fi >= n:
            return None

        t = float(st["t_host"][fi]) if st["t_host"] is not None else float(fi)
        c = np.asarray(st["color"][fi])
        if c.dtype != np.uint8:
            c = np.clip(c, 0, 255).astype(np.uint8)

        # Convert to BGR based on detected H5 color_format
        cf = st.get("color_format", "")
        if cf == "rgb8":
            c = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
        elif cf in {"bgr8", "bgr8_jpeg_decoded"}:
            pass  # already BGR
        else:
            pass  # no metadata, assume already BGR
        d = np.asarray(st["depth"][fi]) if st["depth"] is not None else np.zeros_like(c[:, :, 0], dtype=np.uint16)

        # Per-frame intrinsics
        fx_val = float(st["fx"][fi]) if st["fx"] is not None else None
        fy_val = float(st["fy"][fi]) if st["fy"] is not None else None
        cx_val = float(st["cx"][fi]) if st["cx"] is not None else None
        cy_val = float(st["cy"][fi]) if st["cy"] is not None else None

        # Per-frame camera pose (quest SLAM world)
        T_tracking_cam = np.eye(4, dtype=np.float64)
        pos = np.asarray(st["camera_position"][fi], dtype=np.float64)
        quat_xyzw = np.asarray(st["camera_rotation_xyzw"][fi], dtype=np.float64)
        if pos.shape == (3,) and quat_xyzw.shape == (4,):
            # xyzw quaternion → rotation matrix (no scipy dependency)
            x, y, z, w = (
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
                float(quat_xyzw[3]),
            )
            norm = float(np.linalg.norm(quat_xyzw))
            if norm <= 1e-12:
                raise RuntimeError(f"{name} frame {fi}: zero-length camera quaternion")
            x, y, z, w = x / norm, y / norm, z / norm, w / norm
            R = np.array(
                [
                    [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                    [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                    [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
                ],
                dtype=np.float64,
            )
            T_tracking_cam[:3, :3] = R
            T_tracking_cam[:3, 3] = pos
            T_tracking_cam = transform_to_y_down(
                T_tracking_cam, st.get("coordinate_system", "y_down")
            )
        else:
            raise RuntimeError(
                f"{name} frame {fi}: invalid camera pose shapes "
                f"position={pos.shape}, quaternion={quat_xyzw.shape}"
            )

        extra = {
            "fx": fx_val,
            "fy": fy_val,
            "cx": cx_val,
            "cy": cy_val,
            "T_tracking_cam": T_tracking_cam,
        }
        frame_index = int(st["frame_index"][fi]) if st["frame_index"] is not None else fi
        st["frame_i"] = fi + 1
        return frame_index, t, c, d, extra

    def _reset_quest_h5_state(self) -> None:
        for st in self._quest_h5_state.values():
            st["frame_i"] = 0

    def _close_quest_h5_files(self) -> None:
        for st in self._quest_h5_state.values():
            f = st.get("file")
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass

    @staticmethod
    def _load_camera_meta(session_dir: Path) -> dict[str, dict[str, Any]]:
        p = session_dir / "camera_metadata.json"
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                out[str(k)] = v
        return out

    def _default_camera_meta(w: int, h: int) -> dict[str, Any]:
        # Approximate D435-like intrinsics from image size.
        hfov_deg = 69.0
        fx = float(w) / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
        fy = fx
        cx = (float(w) - 1.0) * 0.5
        cy = (float(h) - 1.0) * 0.5
        return {
            "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "depth_scale_m": 0.001,
            "depth_aligned_to": "color",
        }

    def _open_h5_state(self) -> dict[str, dict[str, Any]]:
        state: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        for serial, p in self._h5_paths.items():
            try:
                f = h5py.File(str(p), "r")
                if "color" not in f or "depth" not in f:
                    f.close()
                    errors.append(f"{p}: missing color or depth dataset")
                    slog(
                        component=self.__class__.__name__,
                        event="h5_missing_dataset",
                        serial=serial,
                        path=str(p),
                    )
                    continue

                color_ds = f["color"]
                depth_ds = f["depth"]
                t_host_ds = f["t_host"] if "t_host" in f else None
                frame_index_ds = f["frame_index"] if "frame_index" in f else None

                if color_ds.ndim != 4 or depth_ds.ndim != 3:
                    f.close()
                    errors.append(
                        f"{p}: invalid RGBD shapes color={color_ds.shape}, "
                        f"depth={depth_ds.shape}"
                    )
                    slog(
                        component=self.__class__.__name__,
                        event="h5_bad_shape",
                        serial=serial,
                        path=str(p),
                        color_shape=tuple(color_ds.shape),
                        depth_shape=tuple(depth_ds.shape),
                    )
                    continue

                n = int(color_ds.shape[0])
                if n <= 0 or int(depth_ds.shape[0]) != n:
                    f.close()
                    errors.append(
                        f"{p}: mismatched or empty RGBD lengths "
                        f"color={color_ds.shape[0]}, depth={depth_ds.shape[0]}"
                    )
                    slog(
                        component=self.__class__.__name__,
                        event="h5_bad_length",
                        serial=serial,
                        path=str(p),
                        n_color=int(color_ds.shape[0]),
                        n_depth=int(depth_ds.shape[0]),
                    )
                    continue

                if t_host_ds is not None and int(t_host_ds.shape[0]) != n:
                    t_host_ds = None
                if frame_index_ds is not None and int(frame_index_ds.shape[0]) != n:
                    frame_index_ds = None

                state[serial] = {
                    "file": f,
                    "color": color_ds,
                    "color_format": str(f.attrs.get("color_format", "")).lower(),
                    "depth": depth_ds,
                    "t_host": t_host_ds,
                    "frame_index": frame_index_ds,
                    "n_frames": n,
                    "frame_i": 0,
                }
            except Exception as exc:
                errors.append(f"{p}: {exc}")
                slog(
                    component=self.__class__.__name__,
                    event="h5_open_failed",
                    serial=serial,
                    path=str(p),
                    reason=repr(exc),
                )
        if errors:
            for item in state.values():
                item["file"].close()
            raise RuntimeError("Fixed replay validation failed: " + "; ".join(errors))
        return state

    def _reset_h5_state(self) -> None:
        for st in self._h5_state.values():
            st["frame_i"] = 0
        self._h5_batch.clear()

    def _next_h5_frame(
        self, serial: str
    ) -> tuple[int, float, np.ndarray, np.ndarray] | None:
        st = self._h5_state.get(serial)
        if st is None:
            return None

        color_ds = st["color"]
        depth_ds = st["depth"]
        t_host_ds = st["t_host"]
        fi = int(st["frame_i"])
        n = int(st["n_frames"])
        if fi >= n:
            return None

        # ── Batch pre-fetch ────────────────────────────────────────────────
        batch = self._h5_batch.get(serial)
        if batch is None or fi < batch["start_fi"] or fi >= batch["start_fi"] + batch["count"]:
            chunk = min(self._H5_BATCH_SIZE, n - fi)
            if chunk <= 0:
                return None

            t_host_arr = (
                np.asarray(t_host_ds[fi:fi + chunk], dtype=np.float64)
                if t_host_ds is not None
                else np.arange(fi, fi + chunk, dtype=np.float64)
            )
            color_arr = np.asarray(color_ds[fi:fi + chunk])
            if color_arr.dtype != np.uint8:
                color_arr = np.clip(color_arr, 0, 255).astype(np.uint8)
            depth_arr = np.asarray(depth_ds[fi:fi + chunk])
            frame_index_arr = (
                np.asarray(st["frame_index"][fi:fi + chunk], dtype=np.int64)
                if st["frame_index"] is not None
                else np.arange(fi, fi + chunk, dtype=np.int64)
            )

            self._h5_batch[serial] = {
                "t_host": t_host_arr,
                "color": color_arr,
                "depth": depth_arr,
                "frame_index": frame_index_arr,
                "start_fi": fi,
                "count": chunk,
            }
            batch = self._h5_batch[serial]

        # Serve from cached batch
        idx = fi - batch["start_fi"]
        t = float(batch["t_host"][idx])
        c = np.array(batch["color"][idx])   # copy to isolate from batch buffer
        # Convert to BGR based on detected H5 color_format
        cf = st.get("color_format", "")
        if cf == "rgb8":
            c = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
        # else: bgr8 or empty → already BGR, no conversion
        d = np.array(batch["depth"][idx])
        st["frame_i"] = fi + 1
        return int(batch["frame_index"][idx]), t, c, d

    def _close_h5_files(self) -> None:
        for st in self._h5_state.values():
            f = st.get("file")
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        self._close_quest_h5_files()

    async def run(self) -> None:
        has_h5 = bool(self._h5_state)
        has_quest = bool(self._quest_h5_state)
        if not has_h5 and not has_quest:
            raise RuntimeError(
                f"No readable replay h5 files found in session/rgbd or session/quest: {self.session_dir}"
            )

        slog(
            component=self.__class__.__name__,
            event="run",
            session_dir=str(self.session_dir),
            n_cameras_h5=len(self._h5_state),
            n_cameras_quest=len(self._quest_h5_state),
            loop=self.loop_playback,
            fps=(None if self.fps_override is None else float(self.fps_override)),
            has_camera_metadata=bool(self._camera_meta),
            sync_tolerance_s=self.sync_tolerance_s,
            timing="recorded_t_host" if self.fps_override is None else "fps_override",
        )

        try:
            sync_group = 0
            previous_group_time: float | None = None
            pending_rs: dict[str, tuple[int, float, np.ndarray, np.ndarray]] = {}
            pending_quest: dict[
                str, tuple[int, float, np.ndarray, np.ndarray, dict[str, Any]]
            ] = {}
            while True:
                for serial in sorted(self._h5_state.keys()):
                    if serial not in pending_rs:
                        item = self._next_h5_frame(serial)
                        if item is not None:
                            pending_rs[serial] = item
                for name in sorted(self._quest_h5_state.keys()):
                    if name not in pending_quest:
                        item = self._next_quest_h5_frame(name)
                        if item is not None:
                            pending_quest[name] = item

                timestamps = [item[1] for item in pending_rs.values()]
                timestamps.extend(item[1] for item in pending_quest.values())
                if not timestamps:
                    if self.loop_playback:
                        self._reset_h5_state()
                        self._reset_quest_h5_state()
                        pending_rs.clear()
                        pending_quest.clear()
                        previous_group_time = None
                        continue
                    break

                group_time = min(timestamps)
                if previous_group_time is not None:
                    if self.fps_override is not None and float(self.fps_override) > 1e-6:
                        delay = 1.0 / float(self.fps_override)
                    else:
                        delay = max(0.0, group_time - previous_group_time)
                    if delay > 0:
                        await asyncio.sleep(delay)
                previous_group_time = group_time

                selected_rs = [
                    serial
                    for serial, item in pending_rs.items()
                    if item[1] - group_time <= self.sync_tolerance_s
                ]
                selected_quest = [
                    name
                    for name, item in pending_quest.items()
                    if item[1] - group_time <= self.sync_tolerance_s
                ]

                for serial in sorted(selected_rs):
                    frame_index, t_host, color, depth = pending_rs.pop(serial)
                    camera = self._camera_meta.get(serial)
                    if not isinstance(camera, dict) or not camera:
                        h, w = color.shape[:2]
                        camera = self._default_camera_meta(int(w), int(h))

                    payload = {
                        "camera": camera,
                        "frames": {
                            "color": color,
                            "depth": depth,
                        },
                        "frame_index": frame_index,
                        "sync_group": sync_group,
                        "T_world_cam": self._fixed_world_camera[serial],
                        "coordinate_system": "y_down",
                    }
                    pkt = TimeStampedPacket(
                        t_host=float(t_host),
                        source=f"realsense/{serial}",
                        payload=payload,
                        t_capture=float(t_host),
                        t_arrival=float(asyncio.get_running_loop().time()),
                        frame_index=frame_index,
                        sync_group=sync_group,
                    )
                    await self.bus.publish(f"realsense/{serial}/stream", pkt)
                    await self.bus.publish("realsense_stream", pkt)

                for name in sorted(selected_quest):
                    frame_index, t_host, color_bgr, depth, extra = pending_quest.pop(name)
                    fx = extra.get("fx")
                    fy = extra.get("fy")
                    cx = extra.get("cx")
                    cy = extra.get("cy")
                    h, w = color_bgr.shape[:2]
                    if fx is None or fy is None:
                        cam_meta = self._default_camera_meta(int(w), int(h))
                        K = cam_meta["K"]
                    else:
                        K = [[float(fx), 0.0, float(cx or w/2.0)],
                             [0.0, float(fy), float(cy or h/2.0)],
                             [0.0, 0.0, 1.0]]

                    camera = {
                        "K": K,
                        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "depth_scale_m": 0.001,
                        "depth_aligned_to": "color",
                    }

                    payload = {
                        "camera": camera,
                        "frames": {
                            "color": color_bgr,
                            "depth": depth,
                        },
                        "T_tracking_cam": extra.get(
                            "T_tracking_cam", extra.get("T_world_cam")
                        ),
                        "T_world_cam": self._quest_world_tracking
                        @ extra["T_tracking_cam"],
                        "T_world_tracking": self._quest_world_tracking,
                        "coordinate_system": "y_down",
                        "frame_index": frame_index,
                        "sync_group": sync_group,
                    }
                    pkt = TimeStampedPacket(
                        t_host=float(t_host),
                        source=name,
                        payload=payload,
                        t_capture=float(t_host),
                        t_arrival=float(asyncio.get_running_loop().time()),
                        frame_index=frame_index,
                        sync_group=sync_group,
                    )
                    quest_topic = f"{name}/stream"
                    await self.bus.publish(quest_topic, pkt)

                sync_group += 1

        finally:
            self._close_h5_files()

        slog(component=self.__class__.__name__, event="finished")
