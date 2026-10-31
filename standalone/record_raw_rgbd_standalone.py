#!/usr/bin/env python3
"""
Multi-RealSense RGBD recorder with per-camera capture thread + per-camera writer thread,
backed by HDF5 storage.

This version is designed to achieve three goals:

A. More stable than a single shared writer:
   - Each camera has its own writer thread and its own queue.

B. Higher robustness for multi-camera high-rate recording:
   - Each camera has its own capture thread and writer thread.
   - Slow disk writes from one camera do not directly block the others.

C. Better long-term storage structure than NPZ for dataset management:
   - Each camera writes to its own HDF5 file.
   - Data are stored in chunked, appendable datasets.
   - Metadata are saved in JSON and duplicated into HDF5 attributes.

Dependencies:
  - numpy
  - opencv-python
  - pyrealsense2
  - h5py

Example:
  python multi_realsense_hdf5_recorder.py \
  --duration-sec 120 \
  --fps 30 \
  --write-batch-size 16 \
  --queue-size 128 \
  --h5-compression none

  python multi_realsense_hdf5_recorder.py \
  --duration-sec 120 \
  --fps 30 \
  --write-batch-size 16 \
  --queue-size 128 \
  --h5-compression lzf

Output structure:
  inputs/<timestamp>/
    camera_metadata.json
    rgbd/
      <serial>.h5
      <serial>.mp4         (optional)

HDF5 layout for each camera file:
  /color                 uint8   [N, H, W, 3]
  /depth                 uint16  [N, H, W]
  /t_host                float64 [N]
  /frame_index           int64   [N]
  /color_hw_timestamp_ms float64 [N]
  /depth_hw_timestamp_ms float64 [N]
  /color_frame_number    int64   [N]
  /depth_frame_number    int64   [N]
  /color_sensor_ts_ms    float64 [N]
  /depth_sensor_ts_ms    float64 [N]
  /color_backend_ts_ms   float64 [N]
  /depth_backend_ts_ms   float64 [N]
  /dropped_count         int64   [1]
  attrs:
    serial
    width
    height
    fps
    depth_scale_m
    depth_aligned_to
    K_json
    dist_json

Notes:
  - For best stability, use SSD.
  - HDF5 is usually better than NPZ for long recordings, but each camera should write to
    its own HDF5 file. Avoid multiple cameras writing to the same HDF5 file.
  - If disk cannot keep up, queue overflow will cause frame drops. This is often preferable
    to blocking capture and destabilizing all streams.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import h5py
import numpy as np
import pyrealsense2 as rs


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-RealSense HDF5 RGBD recorder")
    p.add_argument("--serial", nargs="*", default=None, help="Camera serial list")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration-sec", type=float, default=120.0)

    # HDF5 batching controls
    p.add_argument(
        "--write-batch-size",
        type=int,
        default=8,
        help="How many frames to append to HDF5 at once per camera",
    )
    p.add_argument(
        "--queue-size",
        type=int,
        default=16,
        help="Per-camera frame queue capacity before dropping frames",
    )
    p.add_argument(
        "--h5-compression",
        type=str,
        default="none",
        choices=["none", "lzf", "gzip"],
        help="HDF5 dataset compression. 'none' is fastest; 'lzf' is a good compromise.",
    )
    p.add_argument(
        "--gzip-level",
        type=int,
        default=1,
        help="GZIP compression level if --h5-compression=gzip",
    )
    p.add_argument(
        "--flush-every-batch",
        action="store_true",
        help="Call HDF5 flush() after every appended batch for stronger durability",
    )

    # Optional raw color preview video
    p.add_argument("--save-raw-video", action="store_true")
    p.add_argument("--raw-video-fps", type=float, default=None)
    p.add_argument("--raw-video-codec", type=str, default="mp4v")

    p.add_argument("--show", action="store_true")
    p.add_argument("--load-params", type=str, default=None,
                   help="Load camera params JSON (saved by tune_realsense_standalone.py)")
    return p


def enumerate_d400_serials() -> list[str]:
    ctx = rs.context()
    out: list[str] = []
    for dev in ctx.devices:
        try:
            if dev.get_info(rs.camera_info.product_line) == "D400":
                out.append(dev.get_info(rs.camera_info.serial_number))
        except Exception:
            continue
    return out


@dataclass
class CameraRuntime:
    serial: str
    width: int
    height: int
    fps: int
    run_dir: Path
    metadata: dict[str, Any]
    save_raw_video: bool
    raw_video_fps: Optional[float]
    raw_video_codec: str
    show: bool
    queue_size: int
    write_batch_size: int
    h5_compression: str
    gzip_level: int
    flush_every_batch: bool

    pipeline: Optional[rs.pipeline] = None
    align: Optional[rs.align] = None
    frame_q: queue.Queue = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    capture_done_event: threading.Event = field(default_factory=threading.Event)

    capture_thread: Optional[threading.Thread] = None
    writer_thread: Optional[threading.Thread] = None

    frames_captured: int = 0
    frames_written: int = 0
    frames_dropped_queue_full: int = 0
    capture_errors: int = 0
    writer_errors: list[str] = field(default_factory=list)


class H5AppendWriter:
    def __init__(
        self,
        path: Path,
        serial: str,
        width: int,
        height: int,
        fps: int,
        metadata: dict[str, Any],
        compression: str,
        gzip_level: int,
    ) -> None:
        self.path = path
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.metadata = metadata
        self.compression = compression
        self.gzip_level = gzip_level
        self.h5 = None
        self.ds_color = None
        self.ds_depth = None
        self.ds_t_host = None
        self.ds_frame_index = None
        self.ds_color_hw_timestamp_ms = None
        self.ds_depth_hw_timestamp_ms = None
        self.ds_color_frame_number = None
        self.ds_depth_frame_number = None
        self.ds_color_sensor_ts_ms = None
        self.ds_depth_sensor_ts_ms = None
        self.ds_color_backend_ts_ms = None
        self.ds_depth_backend_ts_ms = None
        self.ds_dropped_count = None

    def __enter__(self) -> "H5AppendWriter":
        self.h5 = h5py.File(str(self.path), "w")

        comp = None if self.compression == "none" else self.compression
        comp_opts = None
        if self.compression == "gzip":
            comp_opts = int(self.gzip_level)

        h = int(self.height)
        w = int(self.width)

        # Chunking is important for append performance.
        # A moderate temporal chunk is typically a good tradeoff.
        chunk_t = 8

        self.ds_color = self.h5.create_dataset(
            "color",
            shape=(0, h, w, 3),
            maxshape=(None, h, w, 3),
            dtype=np.uint8,
            chunks=(chunk_t, h, w, 3),
            compression=comp,
            compression_opts=comp_opts,
        )
        self.ds_depth = self.h5.create_dataset(
            "depth",
            shape=(0, h, w),
            maxshape=(None, h, w),
            dtype=np.uint16,
            chunks=(chunk_t, h, w),
            compression=comp,
            compression_opts=comp_opts,
        )
        self.ds_t_host = self.h5.create_dataset(
            "t_host",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(max(chunk_t, 64),),
        )
        self.ds_frame_index = self.h5.create_dataset(
            "frame_index",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(max(chunk_t, 64),),
        )
        self.ds_color_hw_timestamp_ms = self.h5.create_dataset(
            "color_hw_timestamp_ms",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(max(chunk_t, 64),),
        )
        self.ds_depth_hw_timestamp_ms = self.h5.create_dataset(
            "depth_hw_timestamp_ms",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(max(chunk_t, 64),),
        )
        self.ds_color_frame_number = self.h5.create_dataset(
            "color_frame_number",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(max(chunk_t, 64),),
        )
        self.ds_depth_frame_number = self.h5.create_dataset(
            "depth_frame_number",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(max(chunk_t, 64),),
        )
        self.ds_color_sensor_ts_ms = self.h5.create_dataset(
            "color_sensor_ts_ms",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(max(chunk_t, 64),),
        )
        self.ds_depth_sensor_ts_ms = self.h5.create_dataset(
            "depth_sensor_ts_ms",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(max(chunk_t, 64),),
        )
        self.ds_color_backend_ts_ms = self.h5.create_dataset(
            "color_backend_ts_ms",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(max(chunk_t, 64),),
        )
        self.ds_depth_backend_ts_ms = self.h5.create_dataset(
            "depth_backend_ts_ms",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(max(chunk_t, 64),),
        )
        self.ds_dropped_count = self.h5.create_dataset(
            "dropped_count",
            shape=(1,),
            maxshape=(1,),
            dtype=np.int64,
        )
        self.ds_dropped_count[0] = 0

        self.h5.attrs["serial"] = self.serial
        self.h5.attrs["width"] = int(self.width)
        self.h5.attrs["height"] = int(self.height)
        self.h5.attrs["fps"] = int(self.fps)
        self.h5.attrs["depth_scale_m"] = float(self.metadata.get("depth_scale_m", 0.0))
        self.h5.attrs["depth_aligned_to"] = str(self.metadata.get("depth_aligned_to", "color"))
        self.h5.attrs["K_json"] = json.dumps(to_jsonable(self.metadata.get("K", [])))
        self.h5.attrs["dist_json"] = json.dumps(to_jsonable(self.metadata.get("dist", [])))

        return self

    def append(
        self,
        color_batch: np.ndarray,
        depth_batch: np.ndarray,
        t_batch: np.ndarray,
        frame_index_batch: np.ndarray,
        color_hw_timestamp_ms_batch: np.ndarray,
        depth_hw_timestamp_ms_batch: np.ndarray,
        color_frame_number_batch: np.ndarray,
        depth_frame_number_batch: np.ndarray,
        color_sensor_ts_ms_batch: np.ndarray,
        depth_sensor_ts_ms_batch: np.ndarray,
        color_backend_ts_ms_batch: np.ndarray,
        depth_backend_ts_ms_batch: np.ndarray,
    ) -> None:
        assert self.h5 is not None
        assert self.ds_color is not None
        assert self.ds_depth is not None
        assert self.ds_t_host is not None
        assert self.ds_frame_index is not None
        assert self.ds_color_hw_timestamp_ms is not None
        assert self.ds_depth_hw_timestamp_ms is not None
        assert self.ds_color_frame_number is not None
        assert self.ds_depth_frame_number is not None
        assert self.ds_color_sensor_ts_ms is not None
        assert self.ds_depth_sensor_ts_ms is not None
        assert self.ds_color_backend_ts_ms is not None
        assert self.ds_depth_backend_ts_ms is not None

        n = int(color_batch.shape[0])
        old_n = int(self.ds_color.shape[0])
        new_n = old_n + n

        self.ds_color.resize((new_n, self.height, self.width, 3))
        self.ds_depth.resize((new_n, self.height, self.width))
        self.ds_t_host.resize((new_n,))
        self.ds_frame_index.resize((new_n,))
        self.ds_color_hw_timestamp_ms.resize((new_n,))
        self.ds_depth_hw_timestamp_ms.resize((new_n,))
        self.ds_color_frame_number.resize((new_n,))
        self.ds_depth_frame_number.resize((new_n,))
        self.ds_color_sensor_ts_ms.resize((new_n,))
        self.ds_depth_sensor_ts_ms.resize((new_n,))
        self.ds_color_backend_ts_ms.resize((new_n,))
        self.ds_depth_backend_ts_ms.resize((new_n,))

        self.ds_color[old_n:new_n] = color_batch
        self.ds_depth[old_n:new_n] = depth_batch
        self.ds_t_host[old_n:new_n] = t_batch
        self.ds_frame_index[old_n:new_n] = frame_index_batch
        self.ds_color_hw_timestamp_ms[old_n:new_n] = color_hw_timestamp_ms_batch
        self.ds_depth_hw_timestamp_ms[old_n:new_n] = depth_hw_timestamp_ms_batch
        self.ds_color_frame_number[old_n:new_n] = color_frame_number_batch
        self.ds_depth_frame_number[old_n:new_n] = depth_frame_number_batch
        self.ds_color_sensor_ts_ms[old_n:new_n] = color_sensor_ts_ms_batch
        self.ds_depth_sensor_ts_ms[old_n:new_n] = depth_sensor_ts_ms_batch
        self.ds_color_backend_ts_ms[old_n:new_n] = color_backend_ts_ms_batch
        self.ds_depth_backend_ts_ms[old_n:new_n] = depth_backend_ts_ms_batch

    def set_dropped_count(self, value: int) -> None:
        assert self.ds_dropped_count is not None
        self.ds_dropped_count[0] = int(value)

    def flush(self) -> None:
        assert self.h5 is not None
        self.h5.flush()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.h5 is not None:
            self.h5.flush()
            self.h5.close()


def create_camera_runtime(
    serial: str,
    width: int,
    height: int,
    fps: int,
    run_dir: Path,
    save_raw_video: bool,
    raw_video_fps: Optional[float],
    raw_video_codec: str,
    show: bool,
    queue_size: int,
    write_batch_size: int,
    h5_compression: str,
    gzip_level: int,
    flush_every_batch: bool,
    override_exposure: float | None = None,
    override_auto_exposure: bool | None = None,
    override_focus: float | None = None,
    override_auto_focus: bool | None = None,
) -> CameraRuntime:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipe.start(cfg)

    # Apply saved camera tuning params
    if override_exposure is not None or override_auto_focus is not None:
        try:
            cs = profile.get_device().first_color_sensor()
            if override_auto_exposure is not None:
                cs.set_option(rs.option.enable_auto_exposure, 1 if override_auto_exposure else 0)
            if not override_auto_exposure and override_exposure is not None:
                cs.set_option(rs.option.exposure, float(override_exposure))
            if override_auto_focus is not None:
                try:
                    cs.set_option(rs.option.enable_auto_focus, 1 if override_auto_focus else 0)
                except Exception:
                    pass
            if not override_auto_focus and override_focus is not None:
                try:
                    cs.set_option(rs.option.manual_focus, float(override_focus))
                except Exception:
                    pass
        except Exception as exc:
            print(f"[WARNING] Failed to apply saved params for {serial}: {exc}")

    align = rs.align(rs.stream.color)

    color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_prof.get_intrinsics()

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    K = np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    metadata = {
        "K": K,
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        "depth_scale_m": depth_scale,
        "depth_aligned_to": "color",
    }

    rt = CameraRuntime(
        serial=serial,
        width=width,
        height=height,
        fps=fps,
        run_dir=run_dir,
        metadata=metadata,
        save_raw_video=save_raw_video,
        raw_video_fps=raw_video_fps,
        raw_video_codec=raw_video_codec,
        show=show,
        queue_size=queue_size,
        write_batch_size=write_batch_size,
        h5_compression=h5_compression,
        gzip_level=gzip_level,
        flush_every_batch=flush_every_batch,
    )
    rt.pipeline = pipe
    rt.align = align
    rt.frame_q = queue.Queue(maxsize=max(int(queue_size), 1))
    return rt


def _safe_frame_metadata(frame: rs.frame, key: Any, default: float = np.nan) -> float:
    try:
        if frame.supports_frame_metadata(key):
            return float(frame.get_frame_metadata(key))
    except Exception:
        pass
    return float(default)


def capture_worker(rt: CameraRuntime, end_time_monotonic: float) -> None:
    assert rt.pipeline is not None
    assert rt.align is not None

    local_frame_idx = 0

    try:
        while not rt.stop_event.is_set():
            if time.monotonic() >= end_time_monotonic:
                break

            try:
                frames = rt.pipeline.wait_for_frames(timeout_ms=1000)
            except Exception:
                rt.capture_errors += 1
                continue

            if frames is None or frames.size() == 0:
                continue

            try:
                aligned = rt.align.process(frames)
            except Exception:
                aligned = frames
                rt.capture_errors += 1
            raw_color = frames.get_color_frame()
            raw_depth = frames.get_depth_frame()
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                continue

            try:
                color = np.asarray(cf.get_data()).copy()
                depth = np.asarray(df.get_data()).copy()
                t_host = time.monotonic()

                color_hw_timestamp_ms = float(raw_color.get_timestamp())
                depth_hw_timestamp_ms = float(raw_depth.get_timestamp())
                color_frame_number = int(raw_color.get_frame_number())
                depth_frame_number = int(raw_depth.get_frame_number())

                color_sensor_ts_ms = _safe_frame_metadata(raw_color, rs.frame_metadata_value.sensor_timestamp)
                depth_sensor_ts_ms = _safe_frame_metadata(raw_depth, rs.frame_metadata_value.sensor_timestamp)
                color_backend_ts_ms = _safe_frame_metadata(raw_color, rs.frame_metadata_value.backend_timestamp)
                depth_backend_ts_ms = _safe_frame_metadata(raw_depth, rs.frame_metadata_value.backend_timestamp)

                item = (
                    local_frame_idx,
                    t_host,
                    color,
                    depth,
                    color_hw_timestamp_ms,
                    depth_hw_timestamp_ms,
                    color_frame_number,
                    depth_frame_number,
                    color_sensor_ts_ms,
                    depth_sensor_ts_ms,
                    color_backend_ts_ms,
                    depth_backend_ts_ms,
                )


                # Prefer keeping the newest frames rather than blocking capture.
                # If the queue is full, drop the oldest queued frame and insert the new one.
                try:
                    rt.frame_q.put_nowait(item)
                except queue.Full:
                    try:
                        rt.frame_q.get_nowait()
                        rt.frame_q.task_done()
                    except queue.Empty:
                        pass

                    rt.frames_dropped_queue_full += 1

                    try:
                        rt.frame_q.put_nowait(item)
                    except queue.Full:
                        # Extremely rare race: if still full, drop the new frame.
                        rt.frames_dropped_queue_full += 1

                local_frame_idx += 1
                rt.frames_captured += 1

                if rt.show:
                    vis = color.copy()
                    cv2.putText(
                        vis,
                        f"{rt.serial[:8]} cap:{rt.frames_captured} wr:{rt.frames_written} drop:{rt.frames_dropped_queue_full}",
                        (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(f"REC {rt.serial}", vis)
                    cv2.waitKey(1)
            except Exception:
                rt.capture_errors += 1

    finally:
        rt.capture_done_event.set()


def writer_worker(rt: CameraRuntime) -> None:
    h5_path = rt.run_dir / "rgbd" / f"{rt.serial}.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    video_writer = None
    video_failed = False

    batch_color: list[np.ndarray] = []
    batch_depth: list[np.ndarray] = []
    batch_t: list[float] = []
    batch_idx: list[int] = []
    batch_color_hw_ts: list[float] = []
    batch_depth_hw_ts: list[float] = []
    batch_color_frame_num: list[int] = []
    batch_depth_frame_num: list[int] = []
    batch_color_sensor_ts: list[float] = []
    batch_depth_sensor_ts: list[float] = []
    batch_color_backend_ts: list[float] = []
    batch_depth_backend_ts: list[float] = []

    def flush_batch(h5w: H5AppendWriter) -> None:
        if not batch_color:
            return
        try:
            color_arr = np.stack(batch_color, axis=0)
            depth_arr = np.stack(batch_depth, axis=0)
            t_arr = np.asarray(batch_t, dtype=np.float64)
            idx_arr = np.asarray(batch_idx, dtype=np.int64)
            color_hw_ts_arr = np.asarray(batch_color_hw_ts, dtype=np.float64)
            depth_hw_ts_arr = np.asarray(batch_depth_hw_ts, dtype=np.float64)
            color_frame_num_arr = np.asarray(batch_color_frame_num, dtype=np.int64)
            depth_frame_num_arr = np.asarray(batch_depth_frame_num, dtype=np.int64)
            color_sensor_ts_arr = np.asarray(batch_color_sensor_ts, dtype=np.float64)
            depth_sensor_ts_arr = np.asarray(batch_depth_sensor_ts, dtype=np.float64)
            color_backend_ts_arr = np.asarray(batch_color_backend_ts, dtype=np.float64)
            depth_backend_ts_arr = np.asarray(batch_depth_backend_ts, dtype=np.float64)
            h5w.append(
                color_arr,
                depth_arr,
                t_arr,
                idx_arr,
                color_hw_ts_arr,
                depth_hw_ts_arr,
                color_frame_num_arr,
                depth_frame_num_arr,
                color_sensor_ts_arr,
                depth_sensor_ts_arr,
                color_backend_ts_arr,
                depth_backend_ts_arr,
            )
            rt.frames_written += int(color_arr.shape[0])
            if rt.flush_every_batch:
                h5w.set_dropped_count(rt.frames_dropped_queue_full)
                h5w.flush()
        except Exception as exc:
            rt.writer_errors.append(str(exc))
        finally:
            batch_color.clear()
            batch_depth.clear()
            batch_t.clear()
            batch_idx.clear()
            batch_color_hw_ts.clear()
            batch_depth_hw_ts.clear()
            batch_color_frame_num.clear()
            batch_depth_frame_num.clear()
            batch_color_sensor_ts.clear()
            batch_depth_sensor_ts.clear()
            batch_color_backend_ts.clear()
            batch_depth_backend_ts.clear()

    with H5AppendWriter(
        path=h5_path,
        serial=rt.serial,
        width=rt.width,
        height=rt.height,
        fps=rt.fps,
        metadata=rt.metadata,
        compression=rt.h5_compression,
        gzip_level=rt.gzip_level,
    ) as h5w:
        try:
            while True:
                if rt.stop_event.is_set() and rt.frame_q.empty() and rt.capture_done_event.is_set():
                    break

                try:
                    item = rt.frame_q.get(timeout=0.1)
                except queue.Empty:
                    if rt.capture_done_event.is_set():
                        break
                    continue

                try:
                    (
                        frame_idx,
                        t_host,
                        color,
                        depth,
                        color_hw_timestamp_ms,
                        depth_hw_timestamp_ms,
                        color_frame_number,
                        depth_frame_number,
                        color_sensor_ts_ms,
                        depth_sensor_ts_ms,
                        color_backend_ts_ms,
                        depth_backend_ts_ms,
                    ) = item
                    batch_color.append(color)
                    batch_depth.append(depth)
                    batch_t.append(t_host)
                    batch_idx.append(frame_idx)
                    batch_color_hw_ts.append(color_hw_timestamp_ms)
                    batch_depth_hw_ts.append(depth_hw_timestamp_ms)
                    batch_color_frame_num.append(color_frame_number)
                    batch_depth_frame_num.append(depth_frame_number)
                    batch_color_sensor_ts.append(color_sensor_ts_ms)
                    batch_depth_sensor_ts.append(depth_sensor_ts_ms)
                    batch_color_backend_ts.append(color_backend_ts_ms)
                    batch_depth_backend_ts.append(depth_backend_ts_ms)

                    if rt.save_raw_video:
                        if video_writer is None and not video_failed:
                            hh, ww = color.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*str(rt.raw_video_codec)[:4])
                            out_path = rt.run_dir / "rgbd" / f"{rt.serial}.mp4"
                            video_writer = cv2.VideoWriter(
                                str(out_path),
                                fourcc,
                                float(rt.raw_video_fps if rt.raw_video_fps is not None else rt.fps),
                                (int(ww), int(hh)),
                            )
                            if video_writer is None or not video_writer.isOpened():
                                video_failed = True
                                video_writer = None
                        if video_writer is not None:
                            video_writer.write(color)

                    if len(batch_color) >= max(int(rt.write_batch_size), 1):
                        flush_batch(h5w)

                except Exception as exc:
                    rt.writer_errors.append(str(exc))
                finally:
                    rt.frame_q.task_done()

            flush_batch(h5w)
            h5w.set_dropped_count(rt.frames_dropped_queue_full)
            h5w.flush()

        finally:
            if video_writer is not None:
                try:
                    video_writer.release()
                except Exception:
                    pass


def main() -> None:
    args = build_parser().parse_args()

    all_serials = enumerate_d400_serials()
    if not all_serials:
        print("[!] No D400 RealSense cameras found")
        return

    # ── Load saved params (override CLI) ──
    if args.load_params:
        params_path = Path(args.load_params)
        if not params_path.exists():
            print(f"[!] Params file not found: {params_path}")
            return
        saved = json.loads(params_path.read_text(encoding="utf-8"))
        print(f"[i] Loaded params from {params_path}")
        args.width = int(saved.get("width", args.width))
        args.height = int(saved.get("height", args.height))
        args.fps = int(saved.get("fps", args.fps))
        # Store exposure/focus settings for apply after pipeline start
        args._override_exposure = saved.get("exposure")
        args._override_auto_exposure = saved.get("auto_exposure")
        args._override_focus = saved.get("focus")
        args._override_auto_focus = saved.get("auto_focus")
    else:
        args._override_exposure = None
        args._override_auto_exposure = None
        args._override_focus = None
        args._override_auto_focus = None

    selected = all_serials
    if args.serial:
        sel = set(args.serial)
        selected = [s for s in all_serials if s in sel]
    if not selected:
        print(f"[!] Requested serials not found: {args.serial}")
        return

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("inputs") / run_stamp
    (run_dir / "rgbd").mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Multi-RealSense HDF5 RGBD Recorder")
    print(f"Serials: {selected}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps} fps")
    print(f"Duration: {args.duration_sec:.1f} s")
    print(f"Write batch size: {args.write_batch_size}")
    print(f"Per-camera queue size: {args.queue_size}")
    print(f"HDF5 compression: {args.h5_compression}")
    print(f"Output: {run_dir}")
    print("=" * 72)

    runtimes: list[CameraRuntime] = []

    try:
        # Create a fully independent runtime per camera.
        for serial in selected:
            rt = create_camera_runtime(
                serial=serial,
                width=args.width,
                height=args.height,
                fps=args.fps,
                run_dir=run_dir,
                save_raw_video=args.save_raw_video,
                raw_video_fps=args.raw_video_fps,
                raw_video_codec=args.raw_video_codec,
                show=args.show,
                queue_size=args.queue_size,
                write_batch_size=args.write_batch_size,
                h5_compression=args.h5_compression,
                gzip_level=args.gzip_level,
                flush_every_batch=args.flush_every_batch,
                override_exposure=args._override_exposure,
                override_auto_exposure=args._override_auto_exposure,
                override_focus=args._override_focus,
                override_auto_focus=args._override_auto_focus,
            )
            runtimes.append(rt)

        metadata = {rt.serial: rt.metadata for rt in runtimes}
        (run_dir / "camera_metadata.json").write_text(
            json.dumps(to_jsonable(metadata), indent=2),
            encoding="utf-8",
        )

        end_time_monotonic = time.monotonic() + float(args.duration_sec)

        # Start writer first, then capture.
        for rt in runtimes:
            rt.writer_thread = threading.Thread(
                target=writer_worker,
                args=(rt,),
                daemon=True,
                name=f"writer-{rt.serial}",
            )
            rt.writer_thread.start()

        for rt in runtimes:
            rt.capture_thread = threading.Thread(
                target=capture_worker,
                args=(rt, end_time_monotonic),
                daemon=True,
                name=f"capture-{rt.serial}",
            )
            rt.capture_thread.start()

        print("[*] Recording... Press Ctrl+C to stop early.")

        while True:
            alive_capture = any(
                rt.capture_thread is not None and rt.capture_thread.is_alive()
                for rt in runtimes
            )
            alive_writer = any(
                rt.writer_thread is not None and rt.writer_thread.is_alive()
                for rt in runtimes
            )

            status = "  ".join(
                f"{rt.serial[:6]} cap:{rt.frames_captured} wr:{rt.frames_written} q:{rt.frame_q.qsize()} drop:{rt.frames_dropped_queue_full}"
                for rt in runtimes
            )
            print(f"\r    {status}", end="", flush=True)

            if not alive_capture and not alive_writer:
                break

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[*] KeyboardInterrupt received, stopping...")
        for rt in runtimes:
            rt.stop_event.set()

    finally:
        for rt in runtimes:
            rt.stop_event.set()

        for rt in runtimes:
            if rt.capture_thread is not None:
                rt.capture_thread.join(timeout=3.0)

        # Allow writers to drain per-camera queues.
        for rt in runtimes:
            if rt.writer_thread is not None:
                rt.writer_thread.join(timeout=120.0)

        for rt in runtimes:
            if rt.pipeline is not None:
                try:
                    rt.pipeline.stop()
                except Exception:
                    pass

        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print("\n[OK] Recording finished")
    print(f"[*] Output: {run_dir}")
    print("\nSummary:")
    for rt in runtimes:
        print(
            f"  - {rt.serial}: captured={rt.frames_captured}, written={rt.frames_written}, "
            f"dropped(queue_full)={rt.frames_dropped_queue_full}, capture_errors={rt.capture_errors}, "
            f"writer_errors={len(rt.writer_errors)}"
        )
        if rt.writer_errors:
            for e in rt.writer_errors[:5]:
                print(f"      writer_error: {e}")


if __name__ == "__main__":
    main()
