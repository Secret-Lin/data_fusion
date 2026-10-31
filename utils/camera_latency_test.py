#!/usr/bin/env python3
"""Measure per-camera acquisition latency for all connected RealSense devices.

Latency is computed as host time minus frame timestamp when timestamps are in
system time domain. If the device reports hardware timestamps, we still report
the delta but mark the domain so you can interpret it correctly.

Example:
  python utils/camera_latency_test.py --frames 200 --warmup 30 --width 640 --height 480 --fps 30
"""

import argparse
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pyrealsense2 as rs


@dataclass
class CameraResult:
    serial: str
    domain: str
    count: int
    avg_ms: float
    std_ms: float
    min_ms: float
    max_ms: float


def _timestamp_domain_name(domain) -> str:
    if domain == rs.timestamp_domain.system_time:
        return "system_time"
    if domain == rs.timestamp_domain.hardware_clock:
        return "hardware_clock"
    if domain == rs.timestamp_domain.global_time:
        return "global_time"
    return "unknown"


def _measure_camera(
    serial: str,
    width: int,
    height: int,
    fps: int,
    frames: int,
    warmup: int,
    results: Dict[str, CameraResult],
) -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipeline.start(config)
    try:
        for _ in range(warmup):
            pipeline.wait_for_frames()

        latencies: List[float] = []
        domain_name: Optional[str] = None

        for _ in range(frames):
            frameset = pipeline.wait_for_frames()
            frame = frameset.get_color_frame()
            if not frame:
                continue

            ts_ms = frame.get_timestamp()
            domain = frame.get_frame_timestamp_domain()
            domain_name = _timestamp_domain_name(domain)

            now_ms = time.time() * 1000.0
            latencies.append(now_ms - ts_ms)

        if not latencies:
            return

        avg_ms = statistics.mean(latencies)
        std_ms = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
        res = CameraResult(
            serial=serial,
            domain=domain_name or "unknown",
            count=len(latencies),
            avg_ms=avg_ms,
            std_ms=std_ms,
            min_ms=min(latencies),
            max_ms=max(latencies),
        )
        results[serial] = res
    finally:
        pipeline.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure RealSense per-camera latency.")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to measure")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup frames to skip")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return

    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
    print("Found devices:", ", ".join(serials))

    results: Dict[str, CameraResult] = {}
    threads: List[threading.Thread] = []

    for serial in serials:
        t = threading.Thread(
            target=_measure_camera,
            args=(serial, args.width, args.height, args.fps, args.frames, args.warmup, results),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\nLatency results (ms):")
    for serial in serials:
        res = results.get(serial)
        if res is None:
            print(f"  {serial}: no frames captured")
            continue
        print(
            f"  {res.serial} | domain={res.domain} | n={res.count} | "
            f"avg={res.avg_ms:.2f} | std={res.std_ms:.2f} | "
            f"min={res.min_ms:.2f} | max={res.max_ms:.2f}"
        )

    print("\nNotes:")
    print("- If domain=system_time, avg is a true host-side latency estimate.")
    print("- If domain=hardware_clock, avg is relative and should be compared across cameras cautiously.")


if __name__ == "__main__":
    main()
