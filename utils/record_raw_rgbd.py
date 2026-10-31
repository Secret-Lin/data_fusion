#!/usr/bin/env python3
# pyright: basic
"""
Raw RGB-D recorder for multi-RealSense sessions.

This script only records raw data; it does not perform calibration.

Outputs are saved under:
  inputs/<timestamp>/
    camera_metadata.json
    rgbd/<serial>/chunk_XXXX.npz
    <serial>.mp4          (optional, if --save-raw-video)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BackendConfig, CameraConfig
from core import EventBus
from core.types import TimeStampedPacket
from managers import SensorManager


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


async def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Record raw RGBD chunks from RealSense")
    ap.add_argument("--serial", nargs="*", default=None, help="Camera serial list")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--duration-sec",
        type=float,
        default=120.0,
        help="Auto-stop duration in seconds (default: 120)",
    )
    ap.add_argument(
        "--save-raw-video",
        action="store_true",
        help="Also save per-camera raw color mp4",
    )
    ap.add_argument("--raw-video-fps", type=float, default=None)
    ap.add_argument("--raw-video-codec", type=str, default="mp4v")
    ap.add_argument(
        "--rgbd-chunk-size",
        type=int,
        default=300,
        help="Frames per chunk file (default: 300)",
    )
    ap.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable npz compression for faster write speed",
    )
    ap.add_argument(
        "--flush-timeout-sec",
        type=float,
        default=30.0,
        help="Max wait time for background writer flush on exit (default: 30)",
    )
    ap.add_argument("--show", action="store_true", help="Show live color preview")
    args = ap.parse_args(argv)

    # Output folder
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("inputs") / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RAW RGBD RECORDER")
    print(f"  Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"  Duration: {args.duration_sec:.1f}s")
    print(f"  Output dir: {run_dir}")
    print(f"  Save video: {bool(args.save_raw_video)}")
    print(f"  RGBD chunk size: {args.rgbd_chunk_size}")
    print(f"  RGBD compression: {not args.no_compress}")
    print("=" * 60)

    # Enumerate RealSense cameras
    import pyrealsense2 as rs

    ctx = rs.context()
    devices_rs = [
        d for d in ctx.devices if d.get_info(rs.camera_info.product_line) == "D400"
    ]
    if not devices_rs:
        print("[!] No RealSense devices found.")
        return

    target_serials = set(args.serial) if args.serial else None
    cam_devices = [
        d
        for d in devices_rs
        if target_serials is None
        or d.get_info(rs.camera_info.serial_number) in target_serials
    ]
    if not cam_devices:
        print(f"[!] None of requested serials found: {args.serial}")
        return

    serials = [d.get_info(rs.camera_info.serial_number) for d in cam_devices]
    print(f"[*] Recording {len(serials)} camera(s): {serials}")

    # Build config + managers
    cam_configs: list[CameraConfig] = []
    for i, serial in enumerate(serials):
        cam_configs.append(
            CameraConfig(
                name=f"rs_{i}",
                serial=serial,
                width=int(args.width),
                height=int(args.height),
                fps=int(args.fps),
                use_color=True,
                use_depth=True,
            )
        )

    cfg = BackendConfig()
    cfg.realsense_cameras = cam_configs

    loop = asyncio.get_running_loop()
    bus = EventBus()
    sensor_mgr = SensorManager(
        bus,
        loop,
        cfg,
        enable_hl2=False,
        enable_realsense=True,
        realsense_show_preview=False,
        realsense_cameras=cam_configs,
    )
    sensor_mgr.start()

    frame_queues: dict[str, asyncio.Queue] = {}
    for serial in serials:
        frame_queues[serial] = bus.subscribe(f"realsense/{serial}/stream")

    # Buffers
    chunk_size = max(int(args.rgbd_chunk_size), 1)
    rgbd_color_buf: dict[str, list[np.ndarray]] = {s: [] for s in serials}
    rgbd_depth_buf: dict[str, list[np.ndarray]] = {s: [] for s in serials}
    rgbd_t_buf: dict[str, list[float]] = {s: [] for s in serials}
    rgbd_chunk_idx: dict[str, int] = {s: 0 for s in serials}
    latest_cam_meta: dict[str, dict[str, Any]] = {}
    latest_color: dict[str, np.ndarray] = {}
    frame_counts: dict[str, int] = {s: 0 for s in serials}

    video_writers: dict[str, cv2.VideoWriter] = {}
    video_writer_failed: set[str] = set()

    write_queue: queue.Queue[Any] = queue.Queue(maxsize=128)
    write_errors: list[str] = []

    def writer_worker() -> None:
        while True:
            item = write_queue.get()
            if item is None:
                write_queue.task_done()
                break
            serial, idx, chunk_color, chunk_depth, chunk_t = item
            try:
                out_dir = run_dir / "rgbd" / serial
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"chunk_{idx:04d}.npz"
                if args.no_compress:
                    np.savez(
                        out_path, color=chunk_color, depth=chunk_depth, t_host=chunk_t
                    )
                else:
                    np.savez_compressed(
                        out_path,
                        color=chunk_color,
                        depth=chunk_depth,
                        t_host=chunk_t,
                    )
            except Exception as exc:
                write_errors.append(f"{serial}/chunk_{idx:04d}: {exc}")
            finally:
                write_queue.task_done()

    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    writer_thread.start()

    def flush_chunks(serial: str, force: bool = False) -> None:
        cbuf = rgbd_color_buf[serial]
        dbuf = rgbd_depth_buf[serial]
        tbuf = rgbd_t_buf[serial]
        while len(cbuf) >= chunk_size or (force and len(cbuf) > 0):
            n = chunk_size if len(cbuf) >= chunk_size else len(cbuf)
            chunk_color = np.stack(cbuf[:n], axis=0)
            chunk_depth = np.stack(dbuf[:n], axis=0)
            chunk_t = np.array(tbuf[:n], dtype=np.float64)

            idx = rgbd_chunk_idx[serial]
            # Background write to avoid blocking acquisition loop.
            write_queue.put((serial, idx, chunk_color, chunk_depth, chunk_t))
            rgbd_chunk_idx[serial] = idx + 1

            del cbuf[:n]
            del dbuf[:n]
            del tbuf[:n]

    t0 = time.monotonic()
    print("\n[*] Recording... Press Ctrl+C to stop early.")
    try:
        while True:
            if time.monotonic() - t0 >= float(args.duration_sec):
                break

            for serial, q in frame_queues.items():
                try:
                    while True:
                        pkt: TimeStampedPacket = q.get_nowait()
                        if not isinstance(pkt.payload, dict):
                            continue
                        payload = pkt.payload
                        cam_meta = payload.get("camera")
                        if isinstance(cam_meta, dict):
                            latest_cam_meta[serial] = cam_meta

                        frames = payload.get("frames")
                        if not isinstance(frames, dict):
                            continue
                        color = frames.get("color")
                        depth = frames.get("depth")
                        if color is None or depth is None:
                            continue

                        color_np = np.asarray(color, dtype=np.uint8)
                        depth_np = np.asarray(depth, dtype=np.uint16)
                        latest_color[serial] = color_np

                        rgbd_color_buf[serial].append(color_np.copy())
                        rgbd_depth_buf[serial].append(depth_np.copy())
                        rgbd_t_buf[serial].append(float(pkt.t_host))
                        frame_counts[serial] += 1

                        flush_chunks(serial, force=False)

                        if args.save_raw_video:
                            writer = video_writers.get(serial)
                            if writer is None and serial not in video_writer_failed:
                                h, w = color_np.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(
                                    *str(args.raw_video_codec)[:4]
                                )
                                out_path = run_dir / f"{serial}.mp4"
                                writer = cv2.VideoWriter(
                                    str(out_path),
                                    fourcc,
                                    float(
                                        args.raw_video_fps
                                        if args.raw_video_fps is not None
                                        else args.fps
                                    ),
                                    (int(w), int(h)),
                                )
                                if writer is None or not writer.isOpened():
                                    video_writer_failed.add(serial)
                                    writer = None
                                else:
                                    video_writers[serial] = writer
                            if writer is not None:
                                writer.write(color_np)

                except asyncio.QueueEmpty:
                    pass

            status = "  ".join(f"{s[:6]}:{frame_counts[s]}" for s in serials)
            total = sum(frame_counts.values())
            print(f"\r    {status}  |  total:{total}", end="", flush=True)

            if args.show:
                for serial in serials:
                    frame = latest_color.get(serial)
                    if frame is None:
                        continue
                    vis = frame.copy()
                    cv2.putText(
                        vis,
                        f"{serial[:8]} frames:{frame_counts[serial]}",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(f"REC {serial}", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break

            await asyncio.sleep(0.002)

    except KeyboardInterrupt:
        print("\n[*] KeyboardInterrupt received, stopping and flushing...")
    finally:
        print("\n[*] Finalizing files...")
        for serial in serials:
            try:
                flush_chunks(serial, force=True)
            except BaseException:
                pass

        for w in video_writers.values():
            try:
                w.release()
            except BaseException:
                pass

        for adapter in sensor_mgr.realsense_adapters.values():
            try:
                await adapter.stop()
            except BaseException:
                pass

        # Wait for background writer to finish.
        write_queue.put(None)
        deadline = time.monotonic() + max(float(args.flush_timeout_sec), 0.0)
        while write_queue.unfinished_tasks > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        try:
            writer_thread.join(timeout=0.1)
        except BaseException:
            pass

        pending = write_queue.unfinished_tasks
        if pending > 0:
            print(
                f"[!] Writer flush timeout reached with {pending} pending task(s). "
                f"Increase --flush-timeout-sec or reduce --rgbd-chunk-size."
            )
        if write_errors:
            print(f"[!] Writer errors: {len(write_errors)}")
            for e in write_errors[:5]:
                print(f"    - {e}")

        if args.show:
            cv2.destroyAllWindows()

    # Save metadata
    meta_path = run_dir / "camera_metadata.json"
    meta_path.write_text(
        json.dumps(_to_jsonable(latest_cam_meta), indent=2), encoding="utf-8"
    )

    print("\n[OK] Recording finished.")
    print(f"[*] Output dir: {run_dir}")
    print(f"[*] Metadata: {meta_path}")
    if args.save_raw_video:
        print(f"[*] Raw videos: {run_dir}")
    print(f"[*] Raw RGBD chunks: {run_dir / 'rgbd'}")


if __name__ == "__main__":
    asyncio.run(main())
