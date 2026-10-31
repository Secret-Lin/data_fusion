#!/usr/bin/env python3
r"""
Multi-RealSense + Quest DPT2 RGBD recorder with independent capture and writer threads,
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
  
USB setup (with Android SDK platform-tools on PATH): adb reverse tcp:5001 tcp:5001.
For wireless streaming, configure the host firewall to permit the selected TCP port.

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
    p.add_argument("--load-params", type=str, action="append", default=None,
                   help="Camera params JSON (saved by tune_realsense_standalone.py). Can be specified multiple times.")
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


# ============================================================================
# Quest RGBD TCP recorder extension — DPT2 / SDK reprojection
# ============================================================================

import socket
import struct
from datetime import datetime

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# QCAM RGB header remains unchanged.
QUEST_HEADER_SIZE = 136
QUEST_MAGIC_QCAM = 0x4D414351
QUEST_HEADER_FMT = "<IHHQqiiiiffffii" + "fffffff" + "fffffff" + "iiii"
QUEST_HEADER_STRUCT = struct.Struct(QUEST_HEADER_FMT)

# Legacy and current depth sections.
QUEST_DEPTH_MAGIC_DPT1 = 0x44505431  # ASCII DPT1
QUEST_DEPTH_MAGIC_DPT2 = 0x44505432  # ASCII DPT2

QUEST_DEPTH_FORMAT_UINT16_MM_LE = 1
QUEST_DEPTH_FORMAT_RGB_ALIGNED_Z_UINT16_MM_LE = 2
QUEST_DEPTH_FORMAT_RAW_UNORM16_LE = 3

QUEST_DEPTH_SECTION_HEADER_SIZE_DPT1 = 80
QUEST_DEPTH_SECTION_HEADER_SIZE_DPT2 = 200

QUEST_DEPTH_META_DESCRIPTOR_VALID = 1 << 0
QUEST_DEPTH_META_REPROJECTION_VALID = 1 << 1
QUEST_DEPTH_META_ZBUFFER_VALID = 1 << 2
QUEST_DEPTH_META_SOURCE_SIZED = 1 << 3
QUEST_DEPTH_META_POSE_FALLBACK = 1 << 4

QUEST_MAX_FRAME_PAYLOAD_BYTES = 64 * 1024 * 1024
QUEST_MAX_RESULT_JSON_BYTES = 512 * 1024

DEFAULT_OBJECT_SIZE_M = [0.20, 0.20, 0.20]
POSE_CLASS_SIZES_M = {
    "person": [0.45, 1.70, 0.30],
    "tv": [1.00, 0.60, 0.08],
    "cup": [0.08, 0.10, 0.08],
    "bottle": [0.08, 0.25, 0.08],
    "cell phone": [0.08, 0.16, 0.01],
    "keyboard": [0.45, 0.15, 0.03],
    "mouse": [0.07, 0.04, 0.11],
    "laptop": [0.32, 0.22, 0.02],
}

_QUEST_NDC_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def quest_recv_exact(
    sock: socket.socket,
    n: int,
    stop_event: Optional[threading.Event] = None,
) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        try:
            part = sock.recv(n - len(chunks))
        except socket.timeout:
            if stop_event is None:
                raise
            if stop_event.is_set():
                raise ConnectionAbortedError("recording stopped")
            continue
        if not part:
            raise ConnectionError("socket closed")
        chunks.extend(part)
    return bytes(chunks)


def quest_send_json_packet(sock: socket.socket, packet: dict[str, Any]) -> None:
    data = json.dumps(packet, separators=(",", ":")).encode("utf-8")
    if len(data) > QUEST_MAX_RESULT_JSON_BYTES:
        raise ValueError(f"Quest result JSON too large: {len(data)} bytes")
    sock.sendall(struct.pack("!i", len(data)))
    sock.sendall(data)


def quest_parse_header(data: bytes) -> dict[str, Any]:
    if len(data) < QUEST_HEADER_SIZE:
        raise ValueError(f"Quest header too short: {len(data)} bytes")

    values = QUEST_HEADER_STRUCT.unpack(data[:QUEST_HEADER_SIZE])
    i = 0
    magic = values[i]; i += 1
    version = values[i]; i += 1
    header_size = values[i]; i += 1
    frame_id = values[i]; i += 1
    timestamp_ms = values[i]; i += 1
    stream_w, stream_h, camera_w, camera_h = values[i:i + 4]; i += 4
    fx, fy, cx, cy = values[i:i + 4]; i += 4
    sensor_w, sensor_h = values[i:i + 2]; i += 2
    cam_pose = values[i:i + 7]; i += 7
    lens_pose = values[i:i + 7]; i += 7
    camera_position, jpg_quality, jpg_len, flags = values[i:i + 4]

    if magic != QUEST_MAGIC_QCAM:
        raise ValueError(f"bad Quest magic: {magic:#x}")
    if header_size != QUEST_HEADER_SIZE:
        raise ValueError(f"unexpected Quest header size: {header_size}")
    if jpg_len <= 0:
        raise ValueError(f"invalid Quest JPG length: {jpg_len}")

    return {
        "version": int(version),
        "frameId": int(frame_id),
        "timestampMs": int(timestamp_ms),
        "streamW": int(stream_w),
        "streamH": int(stream_h),
        "cameraW": int(camera_w),
        "cameraH": int(camera_h),
        "intrinsics": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "sensorW": int(sensor_w),
            "sensorH": int(sensor_h),
        },
        "cameraPose": {
            "position": [float(v) for v in cam_pose[0:3]],
            "rotation": [float(v) for v in cam_pose[3:7]],
        },
        "lensOffset": {
            "position": [float(v) for v in lens_pose[0:3]],
            "rotation": [float(v) for v in lens_pose[3:7]],
        },
        "cameraPosition": int(camera_position),
        "jpgQuality": int(jpg_quality),
        "jpgLength": int(jpg_len),
        "flags": int(flags),
    }


def _quest_read_be_i32(payload: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("!i", payload, offset)[0], offset + 4


def _quest_read_be_u32(payload: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("!I", payload, offset)[0], offset + 4


def _quest_read_be_f32(payload: bytes, offset: int) -> tuple[float, int]:
    return struct.unpack_from("!f", payload, offset)[0], offset + 4


def _quest_read_be_f64(payload: bytes, offset: int) -> tuple[float, int]:
    return struct.unpack_from("!d", payload, offset)[0], offset + 8


def quest_depth_from_raw_unorm(
    raw_depth_01: np.ndarray,
    zbuffer_params: tuple[float, float, float, float] | np.ndarray,
    valid_min_m: float,
    valid_max_m: float,
) -> np.ndarray:
    """
    Meta shader convention:
        ndc = raw * 2 - 1
        metric_depth = zbuffer_x / (ndc + zbuffer_y)
    """
    z = np.asarray(zbuffer_params, dtype=np.float32).reshape(4)
    ndc = raw_depth_01.astype(np.float32) * 2.0 - 1.0
    denominator = ndc + float(z[1])

    depth_m = np.zeros(raw_depth_01.shape, dtype=np.float32)
    valid = (
        np.isfinite(ndc)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-8)
    )
    depth_m[valid] = float(z[0]) / denominator[valid]

    invalid = (
        (~np.isfinite(depth_m))
        | (depth_m < float(valid_min_m))
        | (depth_m > float(valid_max_m))
    )
    depth_m[invalid] = 0.0
    return depth_m


def quest_try_parse_depth_section(
    payload: bytes,
    offset: int,
    frame_id: int,
    valid_min_m: float,
    valid_max_m: float,
    log_every_n_frames: int,
) -> Optional[dict[str, Any]]:
    """Parse legacy DPT1 or current DPT2 depth data."""
    remaining = len(payload) - offset
    if remaining <= 0:
        return None
    if remaining < 4:
        raise ValueError(f"Quest depth trailing bytes too short: {remaining}")

    start = offset
    magic, offset = _quest_read_be_u32(payload, offset)

    common: dict[str, Any] = {
        "protocol": "unknown",
        "source_width": 0,
        "source_height": 0,
        "eye_index": -1,
        "metadata_flags": 0,
        "fov_mode": 0,
        "create_time": 0.0,
        "predicted_display_time": 0.0,
        "reprojection_matrix": None,
        "zbuffer_params": None,
    }

    if magic == QUEST_DEPTH_MAGIC_DPT1:
        common["protocol"] = "DPT1"
        header_size = QUEST_DEPTH_SECTION_HEADER_SIZE_DPT1
        if remaining < header_size:
            raise ValueError(f"incomplete DPT1 header: {remaining} < {header_size}")

        depth_w, offset = _quest_read_be_i32(payload, offset)
        depth_h, offset = _quest_read_be_i32(payload, offset)
        depth_format, offset = _quest_read_be_i32(payload, offset)
        depth_len, offset = _quest_read_be_i32(payload, offset)

        min_depth, offset = _quest_read_be_f32(payload, offset)
        max_depth, offset = _quest_read_be_f32(payload, offset)
        near_z, offset = _quest_read_be_f32(payload, offset)
        far_z, offset = _quest_read_be_f32(payload, offset)

        depth_pos = struct.unpack_from("!fff", payload, offset); offset += 12
        depth_rot = struct.unpack_from("!ffff", payload, offset); offset += 16
        fov_left, fov_right, fov_top, fov_down = struct.unpack_from(
            "!ffff", payload, offset
        ); offset += 16

        common["source_width"] = int(depth_w)
        common["source_height"] = int(depth_h)

    elif magic == QUEST_DEPTH_MAGIC_DPT2:
        common["protocol"] = "DPT2"

        header_size, offset = _quest_read_be_i32(payload, offset)
        if header_size != QUEST_DEPTH_SECTION_HEADER_SIZE_DPT2:
            raise ValueError(
                f"unsupported DPT2 header size: {header_size}; "
                f"expected {QUEST_DEPTH_SECTION_HEADER_SIZE_DPT2}"
            )
        if remaining < header_size:
            raise ValueError(f"incomplete DPT2 header: {remaining} < {header_size}")

        depth_w, offset = _quest_read_be_i32(payload, offset)
        depth_h, offset = _quest_read_be_i32(payload, offset)
        source_w, offset = _quest_read_be_i32(payload, offset)
        source_h, offset = _quest_read_be_i32(payload, offset)
        depth_format, offset = _quest_read_be_i32(payload, offset)
        depth_len, offset = _quest_read_be_i32(payload, offset)
        eye_index, offset = _quest_read_be_i32(payload, offset)
        metadata_flags, offset = _quest_read_be_i32(payload, offset)
        fov_mode, offset = _quest_read_be_i32(payload, offset)

        create_time, offset = _quest_read_be_f64(payload, offset)
        predicted_display_time, offset = _quest_read_be_f64(payload, offset)

        min_depth, offset = _quest_read_be_f32(payload, offset)
        max_depth, offset = _quest_read_be_f32(payload, offset)
        near_z, offset = _quest_read_be_f32(payload, offset)
        far_z, offset = _quest_read_be_f32(payload, offset)

        depth_pos = struct.unpack_from("!fff", payload, offset); offset += 12
        depth_rot = struct.unpack_from("!ffff", payload, offset); offset += 16
        fov_left, fov_right, fov_top, fov_down = struct.unpack_from(
            "!ffff", payload, offset
        ); offset += 16

        matrix_values = struct.unpack_from("!16f", payload, offset); offset += 64
        reprojection_matrix = np.asarray(
            matrix_values, dtype=np.float32
        ).reshape(4, 4)

        zbuffer_params = struct.unpack_from("!ffff", payload, offset); offset += 16

        common.update({
            "source_width": int(source_w),
            "source_height": int(source_h),
            "eye_index": int(eye_index),
            "metadata_flags": int(metadata_flags),
            "fov_mode": int(fov_mode),
            "create_time": float(create_time),
            "predicted_display_time": float(predicted_display_time),
            "reprojection_matrix": reprojection_matrix,
            "zbuffer_params": tuple(float(v) for v in zbuffer_params),
        })
    else:
        raise ValueError(f"unknown Quest depth magic: {magic:#x}")

    if offset - start != header_size:
        raise ValueError(
            f"{common['protocol']} header parse size mismatch: "
            f"{offset - start} != {header_size}"
        )

    if depth_format not in (
        QUEST_DEPTH_FORMAT_UINT16_MM_LE,
        QUEST_DEPTH_FORMAT_RGB_ALIGNED_Z_UINT16_MM_LE,
        QUEST_DEPTH_FORMAT_RAW_UNORM16_LE,
    ):
        raise ValueError(f"unsupported Quest depth format: {depth_format}")

    if depth_w <= 0 or depth_h <= 0:
        raise ValueError(f"invalid Quest depth resolution: {depth_w}x{depth_h}")

    expected_len = int(depth_w) * int(depth_h) * 2
    if depth_len != expected_len:
        raise ValueError(
            f"Quest depth byte length mismatch: header={depth_len}, "
            f"expected={expected_len}"
        )
    if len(payload) - offset < depth_len:
        raise ValueError(
            f"incomplete Quest depth bytes: "
            f"remaining={len(payload) - offset}, expected={depth_len}"
        )

    depth_bytes = payload[offset:offset + depth_len]
    raw_u16 = np.frombuffer(
        depth_bytes, dtype="<u2"
    ).reshape((depth_h, depth_w)).copy()

    raw_depth_01 = None
    if depth_format == QUEST_DEPTH_FORMAT_RAW_UNORM16_LE:
        raw_depth_01 = raw_u16.astype(np.float32) / 65535.0
        zparams = common.get("zbuffer_params")
        flags = int(common.get("metadata_flags", 0))
        if (
            zparams is not None
            and (flags & QUEST_DEPTH_META_ZBUFFER_VALID)
            and np.all(np.isfinite(np.asarray(zparams, dtype=np.float32)))
        ):
            depth_m = quest_depth_from_raw_unorm(
                raw_depth_01,
                zparams,
                valid_min_m,
                valid_max_m,
            )
        else:
            depth_m = np.zeros(raw_depth_01.shape, dtype=np.float32)
    else:
        depth_m = raw_u16.astype(np.float32) * 0.001
        invalid = (
            (~np.isfinite(depth_m))
            | (depth_m < float(valid_min_m))
            | (depth_m > float(valid_max_m))
        )
        depth_m[invalid] = 0.0

    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    valid_metric = (
        np.isfinite(depth_m)
        & (depth_m >= float(valid_min_m))
        & (depth_m <= float(valid_max_m))
    )
    depth_mm[valid_metric] = np.clip(
        np.round(depth_m[valid_metric] * 1000.0),
        0,
        65535,
    ).astype(np.uint16)

    is_rgb_aligned = (
        depth_format == QUEST_DEPTH_FORMAT_RGB_ALIGNED_Z_UINT16_MM_LE
    )

    if depth_format == QUEST_DEPTH_FORMAT_RAW_UNORM16_LE:
        format_name = "raw_environment_depth_unorm16_le"
    elif is_rgb_aligned:
        format_name = "rgb_aligned_z_uint16_mm_le"
    else:
        format_name = "raw_openxr_uint16_mm_le"

    if (
        log_every_n_frames > 0
        and frame_id % int(log_every_n_frames) == 0
    ):
        values = depth_m[valid_metric]
        stats = "no metric depth"
        if values.size > 0:
            stats = (
                f"valid={float(valid_metric.mean()):.3f} "
                f"min/p50/max={values.min():.3f}/"
                f"{np.median(values):.3f}/{values.max():.3f}m"
            )
        print(
            f"[Quest DEPTH] frame={frame_id} {common['protocol']} "
            f"format={depth_format} output={depth_w}x{depth_h} "
            f"source={common['source_width']}x{common['source_height']} "
            f"eye={common['eye_index']} "
            f"flags=0x{int(common['metadata_flags']):X} {stats}"
        )

    return {
        "width": int(depth_w),
        "height": int(depth_h),
        "format_id": int(depth_format),
        "format": format_name,
        "is_rgb_aligned": bool(is_rgb_aligned),

        "raw_u16": raw_u16,
        "raw_depth_01": raw_depth_01,
        "depth_m": depth_m,
        "depth_mm": depth_mm,
        "valid_mask": valid_metric,

        "min_depth": float(min_depth),
        "max_depth": float(max_depth),
        "near_z": float(near_z),
        "far_z": float(far_z),

        "depth_position": tuple(float(v) for v in depth_pos),
        "depth_rotation": tuple(float(v) for v in depth_rot),
        "fov_left": float(fov_left),
        "fov_right": float(fov_right),
        "fov_top": float(fov_top),
        "fov_down": float(fov_down),

        **common,
    }


def quest_apply_orientation(
    array: Optional[np.ndarray],
    rotate_180: bool,
    flip_horizontal: bool,
    flip_vertical: bool,
) -> Optional[np.ndarray]:
    if array is None:
        return None

    result = np.asarray(array)
    if rotate_180:
        result = np.rot90(result, 2)
    if flip_horizontal:
        result = np.fliplr(result)
    if flip_vertical:
        result = np.flipud(result)
    return np.ascontiguousarray(result)


def quest_quat_xyzw_to_matrix(q: Any) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)

    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def quest_get_ndc_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    key = (int(width), int(height))
    cached = _QUEST_NDC_GRID_CACHE.get(key)
    if cached is not None:
        return cached

    xx = (
        (np.arange(width, dtype=np.float64) + 0.5)
        / float(width) * 2.0 - 1.0
    )
    yy = (
        (np.arange(height, dtype=np.float64) + 0.5)
        / float(height) * 2.0 - 1.0
    )
    ndc_x, ndc_y = np.meshgrid(xx, yy)
    _QUEST_NDC_GRID_CACHE[key] = (ndc_x, ndc_y)
    return ndc_x, ndc_y


def quest_fill_depth_holes(
    depth_m: np.ndarray,
    valid_min_m: float,
    valid_max_m: float,
    kernel_size: int,
) -> np.ndarray:
    """
    Small optional dilation fill for sparse projection holes.

    This is disabled by default because any fill modifies object boundaries.
    """
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= float(valid_min_m))
        & (depth_m <= float(valid_max_m))
    )
    if not np.any(valid):
        return depth_m

    kernel_size = max(3, int(kernel_size) | 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    filled = depth_m.copy()
    valid_u8 = valid.astype(np.uint8)

    # Dilate inverse depth so nearer surfaces win over farther surfaces.
    inverse = np.zeros_like(filled, dtype=np.float32)
    inverse[valid] = 1.0 / np.maximum(filled[valid], 1e-6)
    inverse_dilated = cv2.dilate(inverse, kernel)
    mask_dilated = cv2.dilate(valid_u8, kernel).astype(bool)

    holes = (~valid) & mask_dilated & (inverse_dilated > 0)
    filled[holes] = 1.0 / inverse_dilated[holes]
    return filled


def quest_project_points_to_rgb(
    points_rgb: np.ndarray,
    meta: dict[str, Any],
    input_valid_count: int,
    rt: "QuestRuntime",
    depth_info: dict[str, Any],
    mode: str,
) -> Optional[np.ndarray]:
    rgb_w = int(meta["streamW"])
    rgb_h = int(meta["streamH"])

    z_rgb = points_rgb[:, 2]
    in_front = (
        np.isfinite(z_rgb)
        & (z_rgb >= rt.depth_valid_min_m)
        & (z_rgb <= rt.depth_valid_max_m)
    )
    if not np.any(in_front):
        depth_info["alignment_error"] = "no depth points in front of RGB camera"
        return None

    points_rgb = points_rgb[in_front]
    z_rgb = z_rgb[in_front].astype(np.float32, copy=False)

    intr = meta["intrinsics"]
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])

    u = fx * (points_rgb[:, 0] / z_rgb) + cx

    # PassthroughCameraAccess sensor/view Y is bottom-up, while the decoded JPG
    # uses OpenCV top-down rows.
    sensor_y = fy * (points_rgb[:, 1] / z_rgb) + cy
    v = (rgb_h - 1.0) - sensor_y

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)

    inside = (
        (ui >= 0) & (ui < rgb_w)
        & (vi >= 0) & (vi < rgb_h)
    )
    if not np.any(inside):
        depth_info["alignment_error"] = "all projected depth points outside RGB image"
        return None

    ui = ui[inside]
    vi = vi[inside]
    z_rgb = z_rgb[inside]

    aligned_flat = np.full(rgb_w * rgb_h, np.inf, dtype=np.float32)
    radius = max(0, int(rt.alignment_splat_radius))
    writes = 0

    for dy in range(-radius, radius + 1):
        vv = vi + dy
        valid_v = (vv >= 0) & (vv < rgb_h)
        if not np.any(valid_v):
            continue

        for dx in range(-radius, radius + 1):
            uu = ui + dx
            keep = valid_v & (uu >= 0) & (uu < rgb_w)
            if not np.any(keep):
                continue

            flat = vv[keep] * rgb_w + uu[keep]
            np.minimum.at(aligned_flat, flat, z_rgb[keep])
            writes += int(np.count_nonzero(keep))

    aligned_flat[~np.isfinite(aligned_flat)] = 0.0
    aligned = aligned_flat.reshape((rgb_h, rgb_w))

    if rt.fill_depth_holes:
        aligned = quest_fill_depth_holes(
            aligned,
            rt.depth_valid_min_m,
            rt.depth_valid_max_m,
            rt.depth_hole_fill_kernel,
        )

    output_valid = (
        np.isfinite(aligned)
        & (aligned >= rt.depth_valid_min_m)
        & (aligned <= rt.depth_valid_max_m)
    )

    depth_info["alignment_stats"] = {
        "input_valid": int(input_valid_count),
        "projected_inside": int(np.count_nonzero(inside)),
        "output_valid": int(np.count_nonzero(output_valid)),
        "output_valid_ratio": float(np.mean(output_valid)),
        "splat_radius": int(radius),
        "splat_writes": int(writes),
        "mode": mode,
    }
    depth_info["alignment_error"] = None
    return aligned


def quest_make_aligned_depth_sdk(
    depth_info: dict[str, Any],
    meta: dict[str, Any],
    rt: "QuestRuntime",
) -> Optional[np.ndarray]:
    flags = int(depth_info.get("metadata_flags", 0))
    matrix = depth_info.get("reprojection_matrix")

    if matrix is None or not (flags & QUEST_DEPTH_META_REPROJECTION_VALID):
        return None

    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        depth_info["alignment_error"] = "invalid SDK reprojection matrix"
        return None

    try:
        inverse_reprojection = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        depth_info["alignment_error"] = "singular SDK reprojection matrix"
        return None

    metric_depth = quest_apply_orientation(
        depth_info.get("depth_m"),
        rt.alignment_rotate_180,
        rt.alignment_flip_horizontal,
        rt.alignment_flip_vertical,
    )
    if metric_depth is None:
        return None

    if depth_info.get("format_id") == QUEST_DEPTH_FORMAT_RAW_UNORM16_LE:
        raw_depth = quest_apply_orientation(
            depth_info.get("raw_depth_01"),
            rt.alignment_rotate_180,
            rt.alignment_flip_horizontal,
            rt.alignment_flip_vertical,
        )
        if raw_depth is None:
            return None
        ndc_z = raw_depth.astype(np.float64) * 2.0 - 1.0
    else:
        zparams = depth_info.get("zbuffer_params")
        if zparams is None:
            return None
        zparams = np.asarray(zparams, dtype=np.float64).reshape(4)
        if (
            not np.all(np.isfinite(zparams))
            or abs(float(zparams[0])) < 1e-10
        ):
            return None
        ndc_z = (
            float(zparams[0])
            / np.maximum(metric_depth.astype(np.float64), 1e-8)
            - float(zparams[1])
        )

    valid = (
        np.isfinite(metric_depth)
        & (metric_depth >= rt.depth_valid_min_m)
        & (metric_depth <= rt.depth_valid_max_m)
        & np.isfinite(ndc_z)
        & (ndc_z >= -1.01)
        & (ndc_z <= 1.01)
    )
    input_valid_count = int(np.count_nonzero(valid))
    if input_valid_count <= 0:
        depth_info["alignment_error"] = "no valid SDK depth texels"
        return None

    h, w = metric_depth.shape
    ndc_x, ndc_y = quest_get_ndc_grid(w, h)

    clip = np.stack(
        [
            ndc_x[valid],
            ndc_y[valid],
            ndc_z[valid],
            np.ones(input_valid_count, dtype=np.float64),
        ],
        axis=1,
    )

    world_h = (inverse_reprojection @ clip.T).T
    homogeneous_w = world_h[:, 3]
    finite_w = np.isfinite(homogeneous_w) & (np.abs(homogeneous_w) > 1e-10)
    if not np.any(finite_w):
        depth_info["alignment_error"] = "invalid homogeneous coordinates"
        return None

    points_world = (
        world_h[finite_w, :3] / homogeneous_w[finite_w, None]
    ).astype(np.float32)

    rgb_pose = meta["cameraPose"]
    rgb_position = np.asarray(
        rgb_pose["position"], dtype=np.float32
    ).reshape(3)
    rgb_rotation = quest_quat_xyzw_to_matrix(rgb_pose["rotation"])

    # Unity world -> RGB camera-local, row-vector form.
    points_rgb = (
        points_world - rgb_position.reshape(1, 3)
    ) @ rgb_rotation

    return quest_project_points_to_rgb(
        points_rgb,
        meta,
        input_valid_count,
        rt,
        depth_info,
        mode="sdk_reprojection",
    )


def quest_align_depth_to_rgb(
    depth_info: Optional[dict[str, Any]],
    meta: dict[str, Any],
    rt: "QuestRuntime",
) -> tuple[Optional[np.ndarray], str]:
    if depth_info is None:
        return None, "none"

    rgb_w = int(meta["streamW"])
    rgb_h = int(meta["streamH"])

    if depth_info.get("is_rgb_aligned", False):
        aligned = np.asarray(depth_info["depth_m"], dtype=np.float32)
        if aligned.shape != (rgb_h, rgb_w):
            aligned = cv2.resize(
                aligned,
                (rgb_w, rgb_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return aligned, "quest_rgb_aligned"

    aligned = quest_make_aligned_depth_sdk(depth_info, meta, rt)
    if aligned is not None:
        return aligned, "sdk_reprojection"

    if rt.allow_legacy_resize_fallback:
        aligned = cv2.resize(
            depth_info["depth_m"],
            (rgb_w, rgb_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        return aligned, "legacy_resize_debug"

    return None, "none"


def quest_depth_m_to_uint16_mm(
    depth_m: Optional[np.ndarray],
    valid_min_m: float,
    valid_max_m: float,
    shape_if_none: Optional[tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    if depth_m is None:
        if shape_if_none is None:
            return None
        return np.zeros(shape_if_none, dtype=np.uint16)

    depth = np.asarray(depth_m, dtype=np.float32)
    output = np.zeros(depth.shape, dtype=np.uint16)
    valid = (
        np.isfinite(depth)
        & (depth >= float(valid_min_m))
        & (depth <= float(valid_max_m))
    )
    output[valid] = np.clip(
        np.round(depth[valid] * 1000.0),
        0,
        65535,
    ).astype(np.uint16)
    return output


def quest_colorize_depth_mm(
    depth_mm: np.ndarray,
    valid_min_m: float,
    valid_max_m: float,
    label: str,
) -> np.ndarray:
    depth_m = depth_mm.astype(np.float32) * 0.001
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= float(valid_min_m))
        & (depth_m <= float(valid_max_m))
    )
    if not np.any(valid):
        return np.zeros((depth_mm.shape[0], depth_mm.shape[1], 3), dtype=np.uint8)

    values = depth_m[valid]
    lo = float(np.percentile(values, 2))
    hi = float(np.percentile(values, 98))
    hi = max(hi, lo + 0.25)

    normalized = np.zeros_like(depth_m, dtype=np.float32)
    normalized[valid] = np.clip(
        (depth_m[valid] - lo) / (hi - lo),
        0.0,
        1.0,
    )
    vis = cv2.applyColorMap(
        (normalized * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    vis[~valid] = 0
    cv2.putText(
        vis,
        f"{label} {lo:.2f}-{hi:.2f}m valid={float(valid.mean()):.3f}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def quest_sample_depth_in_bbox(
    depth_mm_color: Optional[np.ndarray],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Optional[float]:
    if depth_mm_color is None:
        return None

    h, w = depth_mm_color.shape[:2]
    x1i = int(max(0, min(w - 1, round(float(x1)))))
    y1i = int(max(0, min(h - 1, round(float(y1)))))
    x2i = int(max(0, min(w - 1, round(float(x2)))))
    y2i = int(max(0, min(h - 1, round(float(y2)))))

    if x2i <= x1i or y2i <= y1i:
        return None

    bw = x2i - x1i
    bh = y2i - y1i
    cx1 = int(x1i + bw * 0.30)
    cx2 = int(x1i + bw * 0.70)
    cy1 = int(y1i + bh * 0.30)
    cy2 = int(y1i + bh * 0.70)

    patch = depth_mm_color[cy1:cy2, cx1:cx2]
    valid = patch[(patch > 150) & (patch < 10000)]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * 0.001


def quest_bbox_center_to_position_unity_cam(
    det: dict[str, Any],
    z_m: float,
    intr: dict[str, Any],
) -> list[float]:
    u = 0.5 * (float(det["x1"]) + float(det["x2"]))
    v = 0.5 * (float(det["y1"]) + float(det["y2"]))

    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])

    x = (u - cx) / fx * z_m
    y = -(v - cy) / fy * z_m
    return [float(x), float(y), float(z_m)]


def quest_default_size_for_label(label: str) -> list[float]:
    return list(POSE_CLASS_SIZES_M.get(str(label), DEFAULT_OBJECT_SIZE_M))


def quest_run_yolo(
    model: Any,
    frame_bgr: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: Optional[str],
) -> list[dict[str, Any]]:
    if model is None:
        return []

    kwargs: dict[str, Any] = {
        "source": frame_bgr,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "verbose": False,
    }
    if device is not None and str(device).strip():
        kwargs["device"] = device

    result = model.predict(**kwargs)[0]
    detections: list[dict[str, Any]] = []

    if result.boxes is None:
        return detections

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    names = result.names
    for xyxy, confidence, cls_id in zip(boxes_xyxy, confs, classes):
        x1, y1, x2, y2 = xyxy.tolist()

        if isinstance(names, dict):
            label = str(names.get(int(cls_id), int(cls_id)))
        else:
            label = str(names[int(cls_id)])

        detections.append({
            "label": label,
            "classId": int(cls_id),
            "conf": round(float(confidence), 4),
            "x1": round(float(x1), 1),
            "y1": round(float(y1), 1),
            "x2": round(float(x2), 1),
            "y2": round(float(y2), 1),
        })

    return detections


def quest_make_fake_pose_objects(
    detections: list[dict[str, Any]],
    depth_mm_color: Optional[np.ndarray],
    meta: dict[str, Any],
    fallback_depth_m: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pose_objects: list[dict[str, Any]] = []
    depth_sources: dict[str, int] = {}
    intr = meta["intrinsics"]

    for det in detections:
        z_m = quest_sample_depth_in_bbox(
            depth_mm_color,
            det["x1"],
            det["y1"],
            det["x2"],
            det["y2"],
        )
        if z_m is None:
            z_m = fallback_depth_m
            source = "fallback"
        else:
            source = "aligned_depth"

        depth_sources[source] = depth_sources.get(source, 0) + 1
        pose_objects.append({
            "label": det["label"],
            "classId": int(det.get("classId", -1)),
            "conf": float(det.get("conf", 0.0)),
            "trackId": -1,
            "positionCam": quest_bbox_center_to_position_unity_cam(
                det, z_m, intr
            ),
            "rotationCam": [0.0, 0.0, 0.0, 1.0],
            "size": quest_default_size_for_label(det["label"]),
            "depthSource": source,
        })

    return pose_objects, depth_sources


@dataclass
class QuestRuntime:
    run_dir: Path
    host: str
    port: int
    record_every_n_frames: int
    record_only_with_depth: bool
    show: bool
    queue_size: int
    write_batch_size: int
    h5_compression: str
    gzip_level: int
    flush_every_batch: bool

    yolo_model_path: Optional[str]
    yolo_conf: float
    yolo_iou: float
    yolo_imgsz: int
    yolo_max_det: int
    yolo_device: Optional[str]
    fake_pose: bool
    fake_pose_depth_m: float

    depth_valid_min_m: float
    depth_valid_max_m: float
    alignment_splat_radius: int
    fill_depth_holes: bool
    depth_hole_fill_kernel: int
    alignment_rotate_180: bool
    alignment_flip_horizontal: bool
    alignment_flip_vertical: bool
    raw_display_flip_vertical: bool
    allow_legacy_resize_fallback: bool
    depth_log_every_n_frames: int

    frame_q: queue.Queue = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    capture_done_event: threading.Event = field(default_factory=threading.Event)
    capture_thread: Optional[threading.Thread] = None
    writer_thread: Optional[threading.Thread] = None

    frames_received: int = 0
    frames_written: int = 0
    frames_dropped_queue_full: int = 0
    capture_errors: int = 0
    writer_errors: list[str] = field(default_factory=list)
    connected: bool = False
    model: Any = None


class QuestH5AppendWriter:
    """Append Quest RGB, aligned depth, raw depth and DPT2 metadata."""

    def __init__(
        self,
        path: Path,
        compression: str,
        gzip_level: int,
    ) -> None:
        self.path = path
        self.compression = compression
        self.gzip_level = gzip_level
        self.h5: Optional[h5py.File] = None
        self.initialized = False
        self.width = 0
        self.height = 0
        self.raw_depth_w = 0
        self.raw_depth_h = 0
        self.string_dtype = h5py.string_dtype(encoding="utf-8")

    def __enter__(self) -> "QuestH5AppendWriter":
        self.h5 = h5py.File(str(self.path), "w")
        return self

    def _comp(self) -> tuple[Optional[str], Optional[int]]:
        compression = None if self.compression == "none" else self.compression
        options = int(self.gzip_level) if self.compression == "gzip" else None
        return compression, options

    def _create_scalar_dataset(
        self,
        name: str,
        dtype: Any,
        chunk_t: int,
    ) -> None:
        assert self.h5 is not None
        self.h5.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            dtype=dtype,
            chunks=(max(chunk_t, 64),),
        )

    def _init_datasets(
        self,
        first: dict[str, Any],
    ) -> None:
        assert self.h5 is not None

        color = first["color"]
        h, w = color.shape[:2]
        self.height = int(h)
        self.width = int(w)

        compression, options = self._comp()
        chunk_t = 4 if h * w >= 1280 * 960 else 8

        self.h5.create_dataset(
            "color",
            shape=(0, h, w, 3),
            maxshape=(None, h, w, 3),
            dtype=np.uint8,
            chunks=(chunk_t, h, w, 3),
            compression=compression,
            compression_opts=options,
        )
        self.h5.create_dataset(
            "depth",
            shape=(0, h, w),
            maxshape=(None, h, w),
            dtype=np.uint16,
            chunks=(chunk_t, h, w),
            compression=compression,
            compression_opts=options,
        )

        for name, dtype in [
            ("t_host", np.float64),
            ("frame_index", np.int64),
            ("quest_frame_id", np.int64),
            ("quest_timestamp_ms", np.int64),
            ("fx", np.float32),
            ("fy", np.float32),
            ("cx", np.float32),
            ("cy", np.float32),
            ("camera_position_id", np.int32),
            ("depth_format_id", np.int32),
            ("depth_eye_index", np.int32),
            ("depth_metadata_flags", np.int32),
            ("depth_source_width", np.int32),
            ("depth_source_height", np.int32),
            ("depth_output_width", np.int32),
            ("depth_output_height", np.int32),
            ("depth_create_time", np.float64),
            ("depth_predicted_display_time", np.float64),
            ("depth_alignment_valid_ratio", np.float32),
        ]:
            self._create_scalar_dataset(name, dtype, chunk_t)

        self.h5.create_dataset(
            "camera_position",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float32,
            chunks=(max(chunk_t, 64), 3),
        )
        self.h5.create_dataset(
            "camera_rotation_xyzw",
            shape=(0, 4),
            maxshape=(None, 4),
            dtype=np.float32,
            chunks=(max(chunk_t, 64), 4),
        )
        self.h5.create_dataset(
            "depth_position",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float32,
            chunks=(max(chunk_t, 64), 3),
        )
        self.h5.create_dataset(
            "depth_rotation_xyzw",
            shape=(0, 4),
            maxshape=(None, 4),
            dtype=np.float32,
            chunks=(max(chunk_t, 64), 4),
        )
        self.h5.create_dataset(
            "depth_fov_l_r_t_d",
            shape=(0, 4),
            maxshape=(None, 4),
            dtype=np.float32,
            chunks=(max(chunk_t, 64), 4),
        )
        self.h5.create_dataset(
            "depth_zbuffer_params",
            shape=(0, 4),
            maxshape=(None, 4),
            dtype=np.float32,
            chunks=(max(chunk_t, 64), 4),
        )
        self.h5.create_dataset(
            "depth_reprojection_matrix",
            shape=(0, 4, 4),
            maxshape=(None, 4, 4),
            dtype=np.float32,
            chunks=(max(chunk_t, 16), 4, 4),
        )

        for name in [
            "depth_protocol",
            "depth_format",
            "depth_mode",
            "depth_alignment_error",
            "depth_meta_json",
            "detections_json",
            "pose_objects_json",
        ]:
            self._create_scalar_dataset(name, self.string_dtype, chunk_t)

        self.h5.create_dataset(
            "dropped_count",
            shape=(1,),
            maxshape=(1,),
            dtype=np.int64,
        )
        self.h5["dropped_count"][0] = 0

        intr = first["meta"].get("intrinsics", {})
        K = [
            [float(intr.get("fx", 0.0)), 0.0, float(intr.get("cx", 0.0))],
            [0.0, float(intr.get("fy", 0.0)), float(intr.get("cy", 0.0))],
            [0.0, 0.0, 1.0],
        ]

        self.h5.attrs["serial"] = "quest"
        self.h5.attrs["width"] = int(w)
        self.h5.attrs["height"] = int(h)
        self.h5.attrs["color_format"] = "bgr8_jpeg_decoded"
        self.h5.attrs["depth_format"] = "uint16_mm_rgb_aligned_to_color"
        self.h5.attrs["depth_aligned_to"] = "color"
        self.h5.attrs["raw_depth_storage"] = (
            "depth_raw_u16 stores exact transported codes; "
            "format=3 means UNORM16, formats 1/2 mean millimeters"
        )
        self.h5.attrs["quest_depth_protocol"] = "DPT2 current; DPT1 compatible"
        self.h5.attrs["coord"] = "unity_cam"
        self.h5.attrs["K_json"] = json.dumps(K)
        self.h5.attrs["dist_json"] = json.dumps([0.0] * 5)

        self.initialized = True

    def _ensure_raw_depth_datasets(
        self,
        items: list[dict[str, Any]],
        existing_frames: int,
    ) -> None:
        assert self.h5 is not None
        if "depth_raw_u16" in self.h5:
            return

        sample = next(
            (
                item
                for item in items
                if item.get("depth_raw_u16") is not None
                and item.get("depth_low_mm") is not None
            ),
            None,
        )
        if sample is None:
            return

        raw = np.asarray(sample["depth_raw_u16"])
        dh, dw = raw.shape[:2]
        self.raw_depth_h = int(dh)
        self.raw_depth_w = int(dw)

        compression, options = self._comp()
        chunk_t = 8

        self.h5.create_dataset(
            "depth_raw_u16",
            shape=(existing_frames, dh, dw),
            maxshape=(None, dh, dw),
            dtype=np.uint16,
            chunks=(chunk_t, dh, dw),
            compression=compression,
            compression_opts=options,
            fillvalue=0,
        )
        self.h5.create_dataset(
            "depth_lowres",
            shape=(existing_frames, dh, dw),
            maxshape=(None, dh, dw),
            dtype=np.uint16,
            chunks=(chunk_t, dh, dw),
            compression=compression,
            compression_opts=options,
            fillvalue=0,
        )

    @staticmethod
    def _depth_meta_json(item: dict[str, Any]) -> str:
        info = item.get("depth_info")
        if info is None:
            return "{}"

        matrix = info.get("reprojection_matrix")
        zparams = info.get("zbuffer_params")
        payload = {
            "protocol": info.get("protocol"),
            "format_id": info.get("format_id"),
            "format": info.get("format"),
            "width": info.get("width"),
            "height": info.get("height"),
            "source_width": info.get("source_width"),
            "source_height": info.get("source_height"),
            "eye_index": info.get("eye_index"),
            "metadata_flags": info.get("metadata_flags"),
            "fov_mode": info.get("fov_mode"),
            "create_time": info.get("create_time"),
            "predicted_display_time": info.get("predicted_display_time"),
            "min_depth": info.get("min_depth"),
            "max_depth": info.get("max_depth"),
            "near_z": info.get("near_z"),
            "far_z": info.get("far_z"),
            "depth_position": info.get("depth_position"),
            "depth_rotation": info.get("depth_rotation"),
            "fov_left": info.get("fov_left"),
            "fov_right": info.get("fov_right"),
            "fov_top": info.get("fov_top"),
            "fov_down": info.get("fov_down"),
            "zbuffer_params": (
                None if zparams is None
                else np.asarray(zparams).reshape(-1).tolist()
            ),
            "reprojection_matrix": (
                None if matrix is None
                else np.asarray(matrix).reshape(4, 4).tolist()
            ),
            "alignment_stats": info.get("alignment_stats"),
            "alignment_error": info.get("alignment_error"),
        }
        return json.dumps(to_jsonable(payload), separators=(",", ":"))

    def append_batch(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        assert self.h5 is not None

        if not self.initialized:
            self._init_datasets(items[0])

        n_old = int(self.h5["color"].shape[0])
        self._ensure_raw_depth_datasets(items, n_old)

        n = len(items)
        n_new = n_old + n

        for name in ["color", "depth"]:
            dataset = self.h5[name]
            dataset.resize((n_new,) + dataset.shape[1:])

        scalar_names = [
            "t_host",
            "frame_index",
            "quest_frame_id",
            "quest_timestamp_ms",
            "fx",
            "fy",
            "cx",
            "cy",
            "camera_position_id",
            "depth_format_id",
            "depth_eye_index",
            "depth_metadata_flags",
            "depth_source_width",
            "depth_source_height",
            "depth_output_width",
            "depth_output_height",
            "depth_create_time",
            "depth_predicted_display_time",
            "depth_alignment_valid_ratio",
            "depth_protocol",
            "depth_format",
            "depth_mode",
            "depth_alignment_error",
            "depth_meta_json",
            "detections_json",
            "pose_objects_json",
        ]
        for name in scalar_names:
            self.h5[name].resize((n_new,))

        for name in [
            "camera_position",
            "camera_rotation_xyzw",
            "depth_position",
            "depth_rotation_xyzw",
            "depth_fov_l_r_t_d",
            "depth_zbuffer_params",
            "depth_reprojection_matrix",
        ]:
            dataset = self.h5[name]
            dataset.resize((n_new,) + dataset.shape[1:])

        if "depth_raw_u16" in self.h5:
            self.h5["depth_raw_u16"].resize(
                (n_new, self.raw_depth_h, self.raw_depth_w)
            )
            self.h5["depth_lowres"].resize(
                (n_new, self.raw_depth_h, self.raw_depth_w)
            )

        self.h5["color"][n_old:n_new] = np.stack(
            [item["color"] for item in items],
            axis=0,
        )
        self.h5["depth"][n_old:n_new] = np.stack(
            [item["depth_color_mm"] for item in items],
            axis=0,
        )

        if "depth_raw_u16" in self.h5:
            raw_batch = []
            metric_batch = []
            for item in items:
                raw = item.get("depth_raw_u16")
                metric = item.get("depth_low_mm")

                if raw is None:
                    raw = np.zeros(
                        (self.raw_depth_h, self.raw_depth_w),
                        dtype=np.uint16,
                    )
                elif raw.shape != (self.raw_depth_h, self.raw_depth_w):
                    raw = cv2.resize(
                        raw,
                        (self.raw_depth_w, self.raw_depth_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                if metric is None:
                    metric = np.zeros(
                        (self.raw_depth_h, self.raw_depth_w),
                        dtype=np.uint16,
                    )
                elif metric.shape != (self.raw_depth_h, self.raw_depth_w):
                    metric = cv2.resize(
                        metric,
                        (self.raw_depth_w, self.raw_depth_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                raw_batch.append(np.asarray(raw, dtype=np.uint16))
                metric_batch.append(np.asarray(metric, dtype=np.uint16))

            self.h5["depth_raw_u16"][n_old:n_new] = np.stack(raw_batch, axis=0)
            self.h5["depth_lowres"][n_old:n_new] = np.stack(metric_batch, axis=0)

        self.h5["t_host"][n_old:n_new] = np.asarray(
            [item["t_host"] for item in items],
            dtype=np.float64,
        )
        self.h5["frame_index"][n_old:n_new] = np.asarray(
            [item["frame_index"] for item in items],
            dtype=np.int64,
        )
        self.h5["quest_frame_id"][n_old:n_new] = np.asarray(
            [item["meta"]["frameId"] for item in items],
            dtype=np.int64,
        )
        self.h5["quest_timestamp_ms"][n_old:n_new] = np.asarray(
            [item["meta"].get("timestampMs", 0) for item in items],
            dtype=np.int64,
        )

        for name in ["fx", "fy", "cx", "cy"]:
            self.h5[name][n_old:n_new] = np.asarray(
                [item["meta"]["intrinsics"][name] for item in items],
                dtype=np.float32,
            )

        self.h5["camera_position_id"][n_old:n_new] = np.asarray(
            [item["meta"].get("cameraPosition", -1) for item in items],
            dtype=np.int32,
        )
        self.h5["camera_position"][n_old:n_new] = np.asarray(
            [item["meta"]["cameraPose"]["position"] for item in items],
            dtype=np.float32,
        )
        self.h5["camera_rotation_xyzw"][n_old:n_new] = np.asarray(
            [item["meta"]["cameraPose"]["rotation"] for item in items],
            dtype=np.float32,
        )

        def depth_value(item: dict[str, Any], key: str, default: Any) -> Any:
            info = item.get("depth_info")
            return default if info is None else info.get(key, default)

        self.h5["depth_format_id"][n_old:n_new] = np.asarray(
            [depth_value(item, "format_id", 0) for item in items],
            dtype=np.int32,
        )
        self.h5["depth_eye_index"][n_old:n_new] = np.asarray(
            [depth_value(item, "eye_index", -1) for item in items],
            dtype=np.int32,
        )
        self.h5["depth_metadata_flags"][n_old:n_new] = np.asarray(
            [depth_value(item, "metadata_flags", 0) for item in items],
            dtype=np.int32,
        )
        self.h5["depth_source_width"][n_old:n_new] = np.asarray(
            [depth_value(item, "source_width", 0) for item in items],
            dtype=np.int32,
        )
        self.h5["depth_source_height"][n_old:n_new] = np.asarray(
            [depth_value(item, "source_height", 0) for item in items],
            dtype=np.int32,
        )
        self.h5["depth_output_width"][n_old:n_new] = np.asarray(
            [depth_value(item, "width", 0) for item in items],
            dtype=np.int32,
        )
        self.h5["depth_output_height"][n_old:n_new] = np.asarray(
            [depth_value(item, "height", 0) for item in items],
            dtype=np.int32,
        )
        self.h5["depth_create_time"][n_old:n_new] = np.asarray(
            [depth_value(item, "create_time", 0.0) for item in items],
            dtype=np.float64,
        )
        self.h5["depth_predicted_display_time"][n_old:n_new] = np.asarray(
            [depth_value(item, "predicted_display_time", 0.0) for item in items],
            dtype=np.float64,
        )

        self.h5["depth_position"][n_old:n_new] = np.asarray(
            [depth_value(item, "depth_position", (0.0, 0.0, 0.0)) for item in items],
            dtype=np.float32,
        )
        self.h5["depth_rotation_xyzw"][n_old:n_new] = np.asarray(
            [depth_value(item, "depth_rotation", (0.0, 0.0, 0.0, 1.0)) for item in items],
            dtype=np.float32,
        )
        self.h5["depth_fov_l_r_t_d"][n_old:n_new] = np.asarray(
            [
                [
                    depth_value(item, "fov_left", 0.0),
                    depth_value(item, "fov_right", 0.0),
                    depth_value(item, "fov_top", 0.0),
                    depth_value(item, "fov_down", 0.0),
                ]
                for item in items
            ],
            dtype=np.float32,
        )
        self.h5["depth_zbuffer_params"][n_old:n_new] = np.asarray(
            [
                depth_value(item, "zbuffer_params", (0.0, 0.0, 0.0, 0.0))
                or (0.0, 0.0, 0.0, 0.0)
                for item in items
            ],
            dtype=np.float32,
        )
        self.h5["depth_reprojection_matrix"][n_old:n_new] = np.asarray(
            [
                depth_value(item, "reprojection_matrix", np.eye(4, dtype=np.float32))
                if depth_value(item, "reprojection_matrix", None) is not None
                else np.eye(4, dtype=np.float32)
                for item in items
            ],
            dtype=np.float32,
        )

        self.h5["depth_alignment_valid_ratio"][n_old:n_new] = np.asarray(
            [
                float(
                    (
                        item.get("depth_info", {}).get("alignment_stats", {})
                        or {}
                    ).get("output_valid_ratio", 0.0)
                )
                for item in items
            ],
            dtype=np.float32,
        )

        self.h5["depth_protocol"][n_old:n_new] = [
            str(depth_value(item, "protocol", "none")) for item in items
        ]
        self.h5["depth_format"][n_old:n_new] = [
            str(depth_value(item, "format", "none")) for item in items
        ]
        self.h5["depth_mode"][n_old:n_new] = [
            str(item.get("depth_mode", "none")) for item in items
        ]
        self.h5["depth_alignment_error"][n_old:n_new] = [
            str(depth_value(item, "alignment_error", "") or "") for item in items
        ]
        self.h5["depth_meta_json"][n_old:n_new] = [
            self._depth_meta_json(item) for item in items
        ]
        self.h5["detections_json"][n_old:n_new] = [
            json.dumps(item.get("detections", []), separators=(",", ":"))
            for item in items
        ]
        self.h5["pose_objects_json"][n_old:n_new] = [
            json.dumps(item.get("poseObjects", []), separators=(",", ":"))
            for item in items
        ]

    def set_dropped_count(self, value: int) -> None:
        if self.h5 is not None and "dropped_count" in self.h5:
            self.h5["dropped_count"][0] = int(value)

    def flush(self) -> None:
        if self.h5 is not None:
            self.h5.flush()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.h5 is not None:
            self.h5.flush()
            self.h5.close()


def quest_writer_worker(rt: QuestRuntime) -> None:
    h5_path = rt.run_dir / "quest" / "quest.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    batch: list[dict[str, Any]] = []

    def flush_batch(writer: QuestH5AppendWriter) -> None:
        nonlocal batch
        if not batch:
            return
        try:
            writer.append_batch(batch)
            rt.frames_written += len(batch)
            if rt.flush_every_batch:
                writer.set_dropped_count(rt.frames_dropped_queue_full)
                writer.flush()
        except Exception as exc:
            rt.writer_errors.append(str(exc))
        finally:
            batch = []

    with QuestH5AppendWriter(
        h5_path,
        rt.h5_compression,
        rt.gzip_level,
    ) as writer:
        while True:
            if (
                rt.stop_event.is_set()
                and rt.frame_q.empty()
                and rt.capture_done_event.is_set()
            ):
                break

            try:
                item = rt.frame_q.get(timeout=0.1)
            except queue.Empty:
                if rt.capture_done_event.is_set():
                    break
                continue

            try:
                batch.append(item)
                if len(batch) >= max(int(rt.write_batch_size), 1):
                    flush_batch(writer)
            except Exception as exc:
                rt.writer_errors.append(str(exc))
            finally:
                rt.frame_q.task_done()

        flush_batch(writer)
        writer.set_dropped_count(rt.frames_dropped_queue_full)
        writer.flush()


def quest_capture_worker(
    rt: QuestRuntime,
    end_time_monotonic: float,
) -> None:
    rt.frame_q = queue.Queue(maxsize=max(int(rt.queue_size), 1))

    if rt.yolo_model_path and rt.model is None:
        if YOLO is None:
            print("[Quest] ultralytics is not installed; YOLO disabled.")
        else:
            try:
                print(f"[Quest] Loading YOLO model: {rt.yolo_model_path}")
                rt.model = YOLO(rt.yolo_model_path)
                print("[Quest] YOLO loaded.")
            except Exception as exc:
                print(f"[Quest] Failed to load YOLO model: {exc}")
                rt.model = None

    local_idx = 0
    server: Optional[socket.socket] = None

    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((rt.host, int(rt.port)))
        server.listen(1)
        server.settimeout(0.5)
        print(f"[Quest] Listening on {rt.host}:{rt.port}")

        while (
            not rt.stop_event.is_set()
            and time.monotonic() < end_time_monotonic
        ):
            try:
                conn, address = server.accept()
            except socket.timeout:
                continue
            except Exception as exc:
                rt.capture_errors += 1
                print(f"[Quest] accept error: {exc}")
                continue

            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(5.0)
                rt.connected = True
                print(f"[Quest] Connected by {address}")

                while (
                    not rt.stop_event.is_set()
                    and time.monotonic() < end_time_monotonic
                ):
                    try:
                        frame_start = time.perf_counter()

                        payload_len = struct.unpack(
                            "!i",
                            quest_recv_exact(conn, 4, rt.stop_event),
                        )[0]
                        if (
                            payload_len <= QUEST_HEADER_SIZE
                            or payload_len > QUEST_MAX_FRAME_PAYLOAD_BYTES
                        ):
                            raise ValueError(
                                f"invalid Quest payload length: {payload_len}"
                            )

                        payload = quest_recv_exact(
                            conn,
                            payload_len,
                            rt.stop_event,
                        )
                        t_host = time.monotonic()

                        meta = quest_parse_header(payload[:QUEST_HEADER_SIZE])
                        jpg_start = QUEST_HEADER_SIZE
                        jpg_end = jpg_start + int(meta["jpgLength"])

                        if jpg_end > len(payload):
                            raise ValueError("Quest JPG length exceeds payload")

                        jpg_bytes = payload[jpg_start:jpg_end]
                        depth_info = None
                        if len(payload) > jpg_end:
                            depth_info = quest_try_parse_depth_section(
                                payload,
                                jpg_end,
                                frame_id=int(meta["frameId"]),
                                valid_min_m=rt.depth_valid_min_m,
                                valid_max_m=rt.depth_valid_max_m,
                                log_every_n_frames=rt.depth_log_every_n_frames,
                            )

                        color = cv2.imdecode(
                            np.frombuffer(jpg_bytes, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if color is None:
                            raise ValueError(
                                f"Quest JPG decode failed frame={meta['frameId']}"
                            )

                        color_h, color_w = color.shape[:2]
                        if (
                            color_w != int(meta["streamW"])
                            or color_h != int(meta["streamH"])
                        ):
                            meta["streamW"] = int(color_w)
                            meta["streamH"] = int(color_h)

                        align_start = time.perf_counter()
                        aligned_depth_m, depth_mode = quest_align_depth_to_rgb(
                            depth_info,
                            meta,
                            rt,
                        )
                        align_ms = (
                            time.perf_counter() - align_start
                        ) * 1000.0

                        depth_color_mm = quest_depth_m_to_uint16_mm(
                            aligned_depth_m,
                            rt.depth_valid_min_m,
                            rt.depth_valid_max_m,
                            shape_if_none=(color_h, color_w),
                        )
                        assert depth_color_mm is not None

                        depth_low_mm = (
                            None
                            if depth_info is None
                            else np.asarray(depth_info["depth_mm"], dtype=np.uint16)
                        )
                        depth_raw_u16 = (
                            None
                            if depth_info is None
                            else np.asarray(depth_info["raw_u16"], dtype=np.uint16)
                        )

                        detections = quest_run_yolo(
                            rt.model,
                            color,
                            rt.yolo_imgsz,
                            rt.yolo_conf,
                            rt.yolo_iou,
                            rt.yolo_max_det,
                            rt.yolo_device,
                        )

                        if rt.fake_pose:
                            pose_objects, depth_sources = (
                                quest_make_fake_pose_objects(
                                    detections,
                                    depth_color_mm,
                                    meta,
                                    rt.fake_pose_depth_m,
                                )
                            )
                        else:
                            pose_objects, depth_sources = [], {}

                        packet = {
                            "schemaVersion": 3,
                            "resultType": "vision",
                            "coord": "unity_cam",
                            "frameId": int(meta["frameId"]),
                            "timestampMs": int(time.time() * 1000),
                            "imageW": int(meta["streamW"]),
                            "imageH": int(meta["streamH"]),
                            "detections": detections,
                            "poseObjects": pose_objects,
                            "depth": {
                                "available": depth_info is not None,
                                "mode": depth_mode,
                                "aligned": aligned_depth_m is not None,
                            },
                        }
                        quest_send_json_packet(conn, packet)

                        rt.frames_received += 1

                        should_record = (
                            rt.record_every_n_frames <= 1
                            or local_idx % int(rt.record_every_n_frames) == 0
                        )
                        if rt.record_only_with_depth and depth_info is None:
                            should_record = False

                        if should_record:
                            item = {
                                "frame_index": local_idx,
                                "t_host": t_host,
                                "color": color.copy(),
                                "depth_color_mm": depth_color_mm.copy(),
                                "depth_low_mm": (
                                    None
                                    if depth_low_mm is None
                                    else depth_low_mm.copy()
                                ),
                                "depth_raw_u16": (
                                    None
                                    if depth_raw_u16 is None
                                    else depth_raw_u16.copy()
                                ),
                                "depth_mode": depth_mode,
                                "depth_info": depth_info,
                                "meta": meta,
                                "detections": detections,
                                "poseObjects": pose_objects,
                            }

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
                                    rt.frames_dropped_queue_full += 1

                        if rt.show:
                            preview = color.copy()
                            for detection in detections:
                                point1 = (
                                    int(detection["x1"]),
                                    int(detection["y1"]),
                                )
                                point2 = (
                                    int(detection["x2"]),
                                    int(detection["y2"]),
                                )
                                cv2.rectangle(
                                    preview,
                                    point1,
                                    point2,
                                    (0, 255, 0),
                                    2,
                                )
                                cv2.putText(
                                    preview,
                                    f"{detection['label']} "
                                    f"{detection['conf']:.2f}",
                                    (point1[0], max(20, point1[1] - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1,
                                )

                            cv2.putText(
                                preview,
                                (
                                    f"Quest rx:{rt.frames_received} "
                                    f"wr:{rt.frames_written} "
                                    f"depth:{depth_mode} "
                                    f"align:{align_ms:.1f}ms "
                                    f"poseDepth:{depth_sources}"
                                ),
                                (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.52,
                                (0, 255, 255),
                                1,
                            )
                            cv2.imshow("REC Quest", preview)

                            if depth_low_mm is not None:
                                raw_display = depth_low_mm
                                if rt.raw_display_flip_vertical:
                                    raw_display = np.flipud(raw_display)
                                cv2.imshow(
                                    "REC Quest depth raw metric",
                                    quest_colorize_depth_mm(
                                        raw_display,
                                        rt.depth_valid_min_m,
                                        rt.depth_valid_max_m,
                                        "raw-display",
                                    ),
                                )

                            cv2.imshow(
                                "REC Quest depth aligned",
                                quest_colorize_depth_mm(
                                    depth_color_mm,
                                    rt.depth_valid_min_m,
                                    rt.depth_valid_max_m,
                                    f"aligned {depth_mode}",
                                ),
                            )
                            cv2.waitKey(1)

                        if (
                            rt.depth_log_every_n_frames > 0
                            and int(meta["frameId"])
                            % rt.depth_log_every_n_frames == 0
                        ):
                            stats = (
                                {}
                                if depth_info is None
                                else depth_info.get("alignment_stats", {}) or {}
                            )
                            valid_ratio = float(
                                stats.get("output_valid_ratio", 0.0)
                            )
                            eye = (
                                -1
                                if depth_info is None
                                else int(depth_info.get("eye_index", -1))
                            )
                            flags = (
                                0
                                if depth_info is None
                                else int(depth_info.get("metadata_flags", 0))
                            )
                            elapsed_ms = (
                                time.perf_counter() - frame_start
                            ) * 1000.0
                            print(
                                f"[Quest] frame={meta['frameId']} "
                                f"RGB={color_w}x{color_h} "
                                f"RGB-eye={meta['cameraPosition']} "
                                f"depth-eye={eye} flags=0x{flags:X} "
                                f"mode={depth_mode} "
                                f"alignValid={valid_ratio:.3f} "
                                f"align={align_ms:.1f}ms "
                                f"total={elapsed_ms:.1f}ms "
                                f"q={rt.frame_q.qsize()}"
                            )

                        local_idx += 1

                    except (ConnectionError, socket.timeout, OSError) as exc:
                        print(f"[Quest] disconnected: {exc}")
                        break
                    except Exception as exc:
                        rt.capture_errors += 1
                        print(f"[Quest] frame error: {exc}")
                        continue

            rt.connected = False

    finally:
        rt.capture_done_event.set()
        rt.connected = False
        if server is not None:
            try:
                server.close()
            except Exception:
                pass


def build_combined_parser() -> argparse.ArgumentParser:
    p = build_parser()
    p.description = "Multi-RealSense + Quest RGBD HDF5 recorder"
    p.add_argument("--output-root", type=str, default="inputs", help="Root directory for all recordings")
    p.add_argument("--disable-realsense", action="store_true", help="Do not record RealSense cameras")

    p.add_argument("--enable-quest", action="store_true", help="Also listen for and record Quest RGBD stream")
    p.add_argument("--quest-host", type=str, default="0.0.0.0")
    p.add_argument("--quest-port", type=int, default=5001)
    p.add_argument("--quest-queue-size", type=int, default=64)
    p.add_argument("--quest-write-batch-size", type=int, default=8)
    p.add_argument("--quest-record-every-n-frames", type=int, default=1)
    p.add_argument("--quest-record-only-with-depth", action="store_true", help="Only write Quest frames that contain DPT1/DPT2 depth")
    p.add_argument("--quest-show", action="store_true", help="Show Quest preview windows")

    p.add_argument("--quest-model-path", type=str, default=None, help="YOLO model for Quest frames. If omitted, Quest YOLO is disabled.")
    p.add_argument("--quest-yolo-conf", type=float, default=0.35)
    p.add_argument("--quest-yolo-iou", type=float, default=0.45)
    p.add_argument("--quest-yolo-imgsz", type=int, default=640)
    p.add_argument("--quest-yolo-max-det", type=int, default=20)
    p.add_argument("--quest-yolo-device", type=str, default=None)
    p.add_argument("--quest-disable-fake-pose", action="store_true", help="Do not create fake poseObjects from YOLO bbox + Quest depth")
    p.add_argument("--quest-fake-pose-depth-m", type=float, default=1.2)

    # Current DPT2 / SDK-reprojection controls.
    p.add_argument("--quest-depth-min-m", type=float, default=0.15)
    p.add_argument("--quest-depth-max-m", type=float, default=10.0)
    p.add_argument("--quest-alignment-splat-radius", type=int, default=1)
    p.add_argument("--quest-fill-depth-holes", action="store_true")
    p.add_argument("--quest-depth-hole-fill-kernel", type=int, default=3)

    # Alignment must normally keep the original texture layout unchanged.
    # The current SDK reprojection matrix already describes that grid.
    p.add_argument("--quest-alignment-rotate-180", action="store_true")
    p.add_argument("--quest-alignment-flip-horizontal", action="store_true")
    p.add_argument("--quest-alignment-flip-vertical", action="store_true")

    # Raw debug display is independent from alignment. Your current Quest raw
    # depth needs a display-only vertical flip, so this is enabled by default.
    p.add_argument(
        "--quest-raw-display-no-flip-vertical",
        action="store_true",
        help="Disable the default display-only vertical flip for raw Quest depth.",
    )

    p.add_argument(
        "--quest-allow-legacy-resize-fallback",
        action="store_true",
        help="If SDK reprojection is unavailable, resize raw depth to RGB. Debug only; not geometrically correct.",
    )
    p.add_argument("--quest-depth-log-every-n-frames", type=int, default=120)
    return p


def combined_main() -> None:
    args = build_combined_parser().parse_args()

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_stamp
    (run_dir / "rgbd").mkdir(parents=True, exist_ok=True)
    if args.enable_quest:
        (run_dir / "quest").mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Multi-RealSense + Quest RGBD HDF5 Recorder")
    print(f"Output: {run_dir}")
    print(f"Duration: {args.duration_sec:.1f} s")
    print(f"HDF5 compression: {args.h5_compression}")
    print("=" * 72)

    runtimes: list[CameraRuntime] = []
    quest_rt: Optional[QuestRuntime] = None

    try:
        if not args.disable_realsense:
            all_serials = enumerate_d400_serials()
            if not all_serials:
                print("[RealSense] No D400 RealSense cameras found")
            else:
                selected = all_serials
                if args.serial:
                    sel = set(args.serial)
                    selected = [s for s in all_serials if s in sel]
                if not selected:
                    print(f"[RealSense] Requested serials not found: {args.serial}")
                else:
                    print(f"[RealSense] Serials: {selected}")
                    print(f"[RealSense] Resolution: {args.width}x{args.height} @ {args.fps} fps")

                    # ── Load per-camera params from JSON files ──
                    cam_params: dict[str, dict] = {}
                    if args.load_params:
                        for p in args.load_params:
                            pp = Path(p)
                            if not pp.exists():
                                print(f"[WARNING] Params file not found: {pp}")
                                continue
                            data = json.loads(pp.read_text(encoding="utf-8"))
                            s = data.get("serial", "")
                            if s:
                                cam_params[s] = data
                                print(f"  [i] Loaded params for {s[:8]} from {pp}")
                            else:
                                print(f"[WARNING] No 'serial' in {pp}")

                    for serial in selected:
                        p = cam_params.get(serial, {})
                        width = int(p.get("width", args.width))
                        height = int(p.get("height", args.height))
                        fps = int(p.get("fps", args.fps))
                        rt = create_camera_runtime(
                            serial=serial,
                            width=width,
                            height=height,
                            fps=fps,
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
                            override_exposure=p.get("exposure"),
                            override_auto_exposure=p.get("auto_exposure"),
                            override_focus=p.get("focus"),
                            override_auto_focus=p.get("auto_focus"),
                        )
                        runtimes.append(rt)

                    metadata = {rt.serial: rt.metadata for rt in runtimes}
                    (run_dir / "camera_metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")

        if args.enable_quest:
            quest_rt = QuestRuntime(
                run_dir=run_dir,
                host=args.quest_host,
                port=args.quest_port,
                record_every_n_frames=max(int(args.quest_record_every_n_frames), 1),
                record_only_with_depth=bool(args.quest_record_only_with_depth),
                show=bool(args.quest_show),
                queue_size=int(args.quest_queue_size),
                write_batch_size=int(args.quest_write_batch_size),
                h5_compression=args.h5_compression,
                gzip_level=int(args.gzip_level),
                flush_every_batch=bool(args.flush_every_batch),
                yolo_model_path=args.quest_model_path,
                yolo_conf=float(args.quest_yolo_conf),
                yolo_iou=float(args.quest_yolo_iou),
                yolo_imgsz=int(args.quest_yolo_imgsz),
                yolo_max_det=int(args.quest_yolo_max_det),
                yolo_device=args.quest_yolo_device,
                fake_pose=not bool(args.quest_disable_fake_pose),
                fake_pose_depth_m=float(args.quest_fake_pose_depth_m),

                depth_valid_min_m=float(args.quest_depth_min_m),
                depth_valid_max_m=float(args.quest_depth_max_m),
                alignment_splat_radius=max(0, int(args.quest_alignment_splat_radius)),
                fill_depth_holes=bool(args.quest_fill_depth_holes),
                depth_hole_fill_kernel=max(3, int(args.quest_depth_hole_fill_kernel) | 1),
                alignment_rotate_180=bool(args.quest_alignment_rotate_180),
                alignment_flip_horizontal=bool(args.quest_alignment_flip_horizontal),
                alignment_flip_vertical=bool(args.quest_alignment_flip_vertical),
                raw_display_flip_vertical=not bool(args.quest_raw_display_no_flip_vertical),
                allow_legacy_resize_fallback=bool(args.quest_allow_legacy_resize_fallback),
                depth_log_every_n_frames=max(0, int(args.quest_depth_log_every_n_frames)),
            )
            quest_info = {
                "host": args.quest_host,
                "port": args.quest_port,
                "record_every_n_frames": args.quest_record_every_n_frames,
                "record_only_with_depth": bool(args.quest_record_only_with_depth),
                "yolo_model_path": args.quest_model_path,
                "fake_pose": not bool(args.quest_disable_fake_pose),
                "depth_storage": {
                    "depth": "uint16 millimeters geometrically aligned to Quest RGB; zeros where unavailable",
                    "depth_lowres": "uint16 metric millimeters on the transported depth grid",
                    "depth_raw_u16": "exact transported uint16 codes; format 3 is raw UNORM16, format 1/2 is millimeters",
                    "alignment": "DPT2 SDK reprojection matrix preferred; no naive resize unless explicitly enabled",
                },
                "quest_depth_protocol": "DPT2 current, DPT1 backward compatible",
                "coord": "unity_cam",
            }
            (run_dir / "quest_metadata.json").write_text(json.dumps(to_jsonable(quest_info), indent=2), encoding="utf-8")

        if not runtimes and quest_rt is None:
            print("[!] Nothing to record. Use RealSense cameras or pass --enable-quest.")
            return

        session_info = {
            "started_wall_time": datetime.now().isoformat(),
            "run_dir": str(run_dir),
            "duration_sec": float(args.duration_sec),
            "realsense_enabled": bool(runtimes),
            "quest_enabled": quest_rt is not None,
            "args": vars(args),
        }
        (run_dir / "session_info.json").write_text(json.dumps(to_jsonable(session_info), indent=2), encoding="utf-8")

        end_time_monotonic = time.monotonic() + float(args.duration_sec)

        # Start writers first.
        for rt in runtimes:
            rt.writer_thread = threading.Thread(target=writer_worker, args=(rt,), daemon=True, name=f"rs-writer-{rt.serial}")
            rt.writer_thread.start()

        if quest_rt is not None:
            quest_rt.writer_thread = threading.Thread(target=quest_writer_worker, args=(quest_rt,), daemon=True, name="quest-writer")
            quest_rt.writer_thread.start()

        # Start capture/server threads.
        for rt in runtimes:
            rt.capture_thread = threading.Thread(target=capture_worker, args=(rt, end_time_monotonic), daemon=True, name=f"rs-capture-{rt.serial}")
            rt.capture_thread.start()

        if quest_rt is not None:
            quest_rt.capture_thread = threading.Thread(target=quest_capture_worker, args=(quest_rt, end_time_monotonic), daemon=True, name="quest-capture")
            quest_rt.capture_thread.start()

        print("[*] Recording... Press Ctrl+C to stop early.")

        while True:
            alive_rs_capture = any(rt.capture_thread is not None and rt.capture_thread.is_alive() for rt in runtimes)
            alive_rs_writer = any(rt.writer_thread is not None and rt.writer_thread.is_alive() for rt in runtimes)
            alive_q_capture = quest_rt is not None and quest_rt.capture_thread is not None and quest_rt.capture_thread.is_alive()
            alive_q_writer = quest_rt is not None and quest_rt.writer_thread is not None and quest_rt.writer_thread.is_alive()

            parts = []
            for rt in runtimes:
                parts.append(f"RS {rt.serial[:6]} cap:{rt.frames_captured} wr:{rt.frames_written} q:{rt.frame_q.qsize()} drop:{rt.frames_dropped_queue_full}")
            if quest_rt is not None:
                parts.append(f"Quest rx:{quest_rt.frames_received} wr:{quest_rt.frames_written} q:{quest_rt.frame_q.qsize()} drop:{quest_rt.frames_dropped_queue_full} conn:{quest_rt.connected}")
            print("\r    " + "  ".join(parts), end="", flush=True)

            if not (alive_rs_capture or alive_rs_writer or alive_q_capture or alive_q_writer):
                break
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[*] KeyboardInterrupt received, stopping...")
        for rt in runtimes:
            rt.stop_event.set()
        if quest_rt is not None:
            quest_rt.stop_event.set()

    finally:
        for rt in runtimes:
            rt.stop_event.set()
        if quest_rt is not None:
            quest_rt.stop_event.set()

        for rt in runtimes:
            if rt.capture_thread is not None:
                rt.capture_thread.join(timeout=3.0)
        if quest_rt is not None and quest_rt.capture_thread is not None:
            quest_rt.capture_thread.join(timeout=3.0)

        for rt in runtimes:
            if rt.writer_thread is not None:
                rt.writer_thread.join(timeout=120.0)
        if quest_rt is not None and quest_rt.writer_thread is not None:
            quest_rt.writer_thread.join(timeout=120.0)

        for rt in runtimes:
            if rt.pipeline is not None:
                try:
                    rt.pipeline.stop()
                except Exception:
                    pass

        if args.show or (quest_rt is not None and quest_rt.show):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print("\n[OK] Recording finished")
    print(f"[*] Output: {run_dir}")
    print("\nSummary:")
    for rt in runtimes:
        print(f"  - RS {rt.serial}: captured={rt.frames_captured}, written={rt.frames_written}, dropped(queue_full)={rt.frames_dropped_queue_full}, capture_errors={rt.capture_errors}, writer_errors={len(rt.writer_errors)}")
        for e in rt.writer_errors[:5]:
            print(f"      writer_error: {e}")
    if quest_rt is not None:
        print(f"  - Quest: received={quest_rt.frames_received}, written={quest_rt.frames_written}, dropped(queue_full)={quest_rt.frames_dropped_queue_full}, capture_errors={quest_rt.capture_errors}, writer_errors={len(quest_rt.writer_errors)}")
        for e in quest_rt.writer_errors[:5]:
            print(f"      writer_error: {e}")


if __name__ == "__main__":
    combined_main()
