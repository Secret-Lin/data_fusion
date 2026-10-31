#!/usr/bin/env python3
# pyright: basic, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
"""
utils/tag_cube_calibrator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OptiTrack-referenced extrinsic calibration for multi-RealSense setups.

Physical setup:
  - A rigid cube with AprilTag TAG16H5 on all 6 faces (IDs 0-5).
  - OptiTrack tracks the cube → provides T_world_cube (world → cube body).
    If OptiTrack is unavailable, T_world_cube defaults to identity.
  - All cameras observe the cube simultaneously.

Algorithm:
  1. OptiTrack streams T_world_cube via NatNet UDP.
  2. Each camera sees the cube → solvePnP → T_cam_cube.
  3. Per frame: T_world_cam = T_world_cube @ inv(T_cam_cube).
  4. Accumulate frames; average via rotation averaging → final T_world_cam.

Usage::

    conda activate mr_nav

    # OTS disabled — cube at world origin
    python utils/tag_cube_calibrator.py --show --ots-off

    # With OptiTrack — pass the rigid-body ID from Motive asset panel
    python utils/tag_cube_calibrator.py --show --ots-host 192.168.1.1 --ots-rb-id 1

    # Headless (no GUI)
    python utils/tag_cube_calibrator.py --ots-off --min-frames 30 --output calibration/tag_cube_extrinsics.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BackendConfig, CameraConfig
from core import EventBus
from core.types import TimeStampedPacket
from managers import SensorManager
from utils.tag_cube_calibrator_core import (
    CubeObservation,
    TagCubeCalibrator,
    _TAG_OFFSETS,
    average_rotation,
    average_translation,
    make_T,
    matrix_to_rotvec,
    rotvec_to_matrix,
)

# ---------------------------------------------------------------------------
# NatNet / OptiTrack
# ---------------------------------------------------------------------------


def _load_ots_listener():
    """Lazy-load so the calibrator still works without NatNet."""
    from utils.ots_listener import OptiTrackListener

    return OptiTrackListener


# ---------------------------------------------------------------------------
# Tag detection helpers
# ---------------------------------------------------------------------------

_aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
_aruco_params = cv2.aruco.DetectorParameters()
try:
    _detector = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)
except AttributeError:
    _detector = None


def detect_tags(gray: np.ndarray) -> tuple[Optional[list], Optional[list]]:
    """Run ArUco/AprilTag detection on a grayscale frame.

    Returns (ids_list, corners_list) or (None, None) on failure.
    corners_list is a list of 4×2 arrays, one per detected tag.
    """
    try:
        if _detector is not None:
            corners_all, ids_all, _ = _detector.detectMarkers(gray)  # type: ignore
        else:
            corners_all, ids_all, _ = cv2.aruco.detectMarkers(  # type: ignore
                gray, _aruco_dict, parameters=_aruco_params
            )
    except Exception:
        return None, None

    if ids_all is None or len(ids_all) == 0:
        return None, None

    return ids_all.ravel().tolist(), corners_all  # type: ignore


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


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TAG16H5 Cube Extrinsic Calibrator — OptiTrack-referenced",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Camera args ----
    parser.add_argument(
        "--serial",
        nargs="*",
        default=None,
        help="RealSense serial(s) to use (default: all found)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)

    # ---- Cube args ----
    parser.add_argument(
        "--cube-size",
        type=float,
        default=50.0,
        help="Cube edge length in mm (default: 50)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Minimum observations per camera before solve (default: 20)",
    )
    parser.add_argument(
        "--max-reproj-error-px",
        type=float,
        default=3.0,
        help="Reject frames with reprojection error above this threshold (default: 3.0)",
    )

    # ---- Output ----
    parser.add_argument(
        "--output",
        default="tag_cube_extrinsics.json",
        help="Output JSON filename (saved under inputs/<timestamp>/)",
    )
    parser.add_argument(
        "--save-raw-video",
        action="store_true",
        help="Save per-camera raw color video to inputs/<timestamp>/<serial>.mp4",
    )
    parser.add_argument(
        "--raw-video-fps",
        type=float,
        default=None,
        help="Raw video FPS (default: camera fps)",
    )
    parser.add_argument(
        "--raw-video-codec",
        type=str,
        default="mp4v",
        help="FourCC codec for raw video (default: mp4v)",
    )
    parser.add_argument(
        "--save-raw-rgbd",
        action="store_true",
        help="Save per-camera raw RGBD chunks to inputs/<timestamp>/rgbd/<serial>/chunk_XXXX.npz",
    )
    parser.add_argument(
        "--rgbd-chunk-size",
        type=int,
        default=300,
        help="Frames per RGBD chunk file (default: 300)",
    )

    # ---- Display ----
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview with tag overlay per camera",
    )

    # ---- OptiTrack args ----
    ots_grp = parser.add_argument_group("OptiTrack (NatNet)")
    ots_grp.add_argument(
        "--ots-host",
        default="192.168.1.1",
        help="OptiTrack workstation IP (default: 192.168.1.1)",
    )
    ots_grp.add_argument(
        "--ots-port",
        type=int,
        default=1511,
        help="NatNet data port (default: 1511)",
    )
    ots_grp.add_argument(
        "--ots-multicast",
        default="239.255.42.99",
        help="NatNet multicast address (default: 239.255.42.99)",
    )
    ots_grp.add_argument(
        "--ots-rb-id",
        type=int,
        default=None,
        help="Rigid-body ID for the cube (look up in Motive asset panel)",
    )
    ots_grp.add_argument(
        "--ots-rb-name",
        default=None,
        help="Rigid-body name (optional; --ots-rb-id is preferred)",
    )
    ots_grp.add_argument(
        "--ots-off",
        action="store_true",
        help="Disable OptiTrack; use identity T_world_cube (cube at world origin)",
    )

    args = parser.parse_args(argv)

    # -------------------------------------------------------------------------
    # Output folder (inputs/<timestamp>/)
    # -------------------------------------------------------------------------
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("inputs") / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    output_name = Path(args.output).name
    output_json_path = run_dir / output_name

    # -------------------------------------------------------------------------
    # Banner
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("TAG CUBE EXTRINSIC CALIBRATOR (async / multi-camera)")
    print("  Cube: TAG16H5 on all 6 faces, IDs 0-5")
    print(f"  Cube size: {args.cube_size} mm")
    print(f"  Min frames per camera: {args.min_frames}")
    print(f"  Output dir: {run_dir}")
    print(f"  Output JSON: {output_json_path.name}")
    if Path(args.output).name != str(args.output):
        print(f"  Note: --output path stripped to filename: {output_json_path.name}")
    if args.save_raw_video:
        print(
            f"  Raw video: ON ({args.raw_video_codec} @ "
            f"{args.raw_video_fps if args.raw_video_fps is not None else args.fps} fps)"
        )
    else:
        print("  Raw video: OFF")
    if args.save_raw_rgbd:
        print(f"  Raw RGBD: ON (chunk_size={args.rgbd_chunk_size})")
    else:
        print("  Raw RGBD: OFF")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # OptiTrack listener
    # -------------------------------------------------------------------------
    ots_enabled = not args.ots_off
    ots = None
    ots_rb_key: Optional[str] = None

    if ots_enabled:
        try:
            OptiTrackListener = _load_ots_listener()
            ots = OptiTrackListener(
                ots_ip=args.ots_host,
                multicast_addr=args.ots_multicast,
                rb_names=[args.ots_rb_name] if args.ots_rb_name else None,
                data_port=args.ots_port,
            )
            if args.ots_rb_id is not None:
                ots.register_rb(
                    args.ots_rb_id, args.ots_rb_name or f"rb_{args.ots_rb_id}"
                )
                ots_rb_key = (args.ots_rb_name or f"rb_{args.ots_rb_id}").lower()
            ots.start()
            print(f"[*] OptiTrack: {args.ots_host}:{args.ots_port}")
            if args.ots_rb_id is not None:
                print(
                    f"    Tracking rigid body ID={args.ots_rb_id} ({args.ots_rb_name})"
                )
            else:
                print(
                    "    Warning: no --ots-rb-id set — use --ots-off or provide --ots-rb-id"
                )
        except Exception as exc:
            print(f"[!] OptiTrack listener failed: {exc}")
            print("    Falling back to identity T_world_cube")
            ots = None

    if not ots_enabled or ots is None:
        print("[*] OTS disabled — using identity T_world_cube")

    print("=" * 60)

    # -------------------------------------------------------------------------
    # Enumerate RealSense cameras
    # -------------------------------------------------------------------------
    try:
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
            print(f"[!] None of the requested serials found: {args.serial}")
            return

        print(f"[*] {len(cam_devices)} RealSense camera(s) found:")
        for dev in cam_devices:
            print(f"    {dev.get_info(rs.camera_info.serial_number)}")
    except Exception as exc:
        print(f"[!] Failed to enumerate cameras: {exc}")
        return

    # -------------------------------------------------------------------------
    # Build BackendConfig + SensorManager (each camera in its own thread)
    # -------------------------------------------------------------------------
    cam_configs: list[CameraConfig] = []
    for dev in cam_devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        cam_configs.append(
            CameraConfig(
                name=f"rs_{serial}",
                serial=serial,
                width=args.width,
                height=args.height,
                fps=args.fps,
                use_color=True,
                use_depth=True,
            )
        )

    cfg = BackendConfig()
    cfg.realsense_cameras = cam_configs
    cfg.hl2.host = ""  # Disable HL2
    cfg.hl2.calibration_path = ""

    loop = asyncio.get_running_loop()
    bus = EventBus()

    sensor_mgr = SensorManager(
        bus,
        loop,
        cfg,
        enable_hl2=False,
        enable_realsense=True,
        realsense_show_preview=False,  # We handle our own display
        realsense_cameras=cam_configs,
    )
    sensor_mgr.start()

    # -------------------------------------------------------------------------
    # Subscribe to camera frame topics
    # -------------------------------------------------------------------------
    serials = [c.serial for c in cam_configs]
    frame_queues: dict[str, asyncio.Queue] = {}
    for serial in serials:
        topic = f"realsense/{serial}/stream"
        frame_queues[serial] = bus.subscribe(topic)

    # -------------------------------------------------------------------------
    # Calibrator
    # -------------------------------------------------------------------------
    calib = TagCubeCalibrator(
        cube_size_mm=args.cube_size,
        min_frames=args.min_frames,
        max_reproj_error_px=args.max_reproj_error_px,
    )
    for serial in serials:
        calib.register_camera(serial)

    frame_counts: dict[str, int] = {s: 0 for s in serials}
    ready = False
    latest_color: dict[str, np.ndarray] = {}  # serial → BGR frame
    latest_depth: dict[str, np.ndarray] = {}  # serial → depth (u16)
    latest_cam_meta: dict[str, dict[str, Any]] = {}  # serial → camera metadata
    latest_tags: dict[str, list] = {}  # serial → list of detected tag IDs
    video_writers: dict[str, cv2.VideoWriter] = {}
    video_writer_failed: set[str] = set()

    rgbd_chunk_size = max(int(args.rgbd_chunk_size), 1)
    rgbd_color_buf: dict[str, list[np.ndarray]] = {s: [] for s in serials}
    rgbd_depth_buf: dict[str, list[np.ndarray]] = {s: [] for s in serials}
    rgbd_t_buf: dict[str, list[float]] = {s: [] for s in serials}
    rgbd_chunk_idx: dict[str, int] = {s: 0 for s in serials}

    def flush_rgbd_chunks(serial: str, force: bool = False) -> None:
        cbuf = rgbd_color_buf.setdefault(serial, [])
        dbuf = rgbd_depth_buf.setdefault(serial, [])
        tbuf = rgbd_t_buf.setdefault(serial, [])
        while len(cbuf) >= rgbd_chunk_size or (force and len(cbuf) > 0):
            n = rgbd_chunk_size if len(cbuf) >= rgbd_chunk_size else len(cbuf)
            chunk_color = np.stack(cbuf[:n], axis=0)
            chunk_depth = np.stack(dbuf[:n], axis=0)
            chunk_t = np.array(tbuf[:n], dtype=np.float64)

            out_dir = run_dir / "rgbd" / serial
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = rgbd_chunk_idx.get(serial, 0)
            out_path = out_dir / f"chunk_{idx:04d}.npz"
            np.savez_compressed(
                out_path,
                color=chunk_color,
                depth=chunk_depth,
                t_host=chunk_t,
            )
            rgbd_chunk_idx[serial] = idx + 1

            del cbuf[:n]
            del dbuf[:n]
            del tbuf[:n]

    # -------------------------------------------------------------------------
    # Create preview windows
    # -------------------------------------------------------------------------
    if args.show:
        for serial in serials:
            cv2.namedWindow(f"CAM {serial}", cv2.WINDOW_NORMAL)

    print("\n[*] Place the cube in view of all cameras.")
    print("    q / Esc — finish and compute result\n")

    # -------------------------------------------------------------------------
    # Main async loop
    # -------------------------------------------------------------------------
    try:
        while True:
            new_color_frame_serials: set[str] = set()

            # Drain all frame queues (non-blocking)
            for serial, q in frame_queues.items():
                try:
                    while True:
                        pkt: TimeStampedPacket = q.get_nowait()
                        if pkt is None:
                            continue
                        payload = getattr(pkt, "payload", None)
                        if not isinstance(payload, dict):
                            continue
                        cam_meta = payload.get("camera")
                        if isinstance(cam_meta, dict):
                            latest_cam_meta[serial] = cam_meta
                        frames = payload.get("frames")
                        if not isinstance(frames, dict):
                            continue
                        color = frames.get("color")
                        depth = frames.get("depth")
                        if color is not None:
                            latest_color[serial] = color
                            new_color_frame_serials.add(serial)
                        if depth is not None:
                            latest_depth[serial] = depth

                        if (
                            args.save_raw_rgbd
                            and color is not None
                            and depth is not None
                        ):
                            rgbd_color_buf[serial].append(
                                np.asarray(color, dtype=np.uint8).copy()
                            )
                            rgbd_depth_buf[serial].append(
                                np.asarray(depth, dtype=np.uint16).copy()
                            )
                            rgbd_t_buf[serial].append(float(pkt.t_host))
                            flush_rgbd_chunks(serial, force=False)
                except asyncio.QueueEmpty:
                    pass

            # Get T_world_cube from OptiTrack (or identity)
            T_world_cube: Optional[np.ndarray] = None
            if ots is not None and ots_rb_key is not None:
                T_world_cube = ots.get_latest(ots_rb_key)
            ots_ok = T_world_cube is not None
            ots_status = "OTS:OK" if ots_ok else "OTS:---"

            # Process each camera
            for serial in serials:
                frame = latest_color.get(serial)
                tags = latest_tags.get(serial, [])

                if frame is None:
                    # Show placeholder if no frame yet
                    if args.show:
                        vis = np.full((480, 640, 3), 20, dtype=np.uint8)
                        cv2.putText(
                            vis,
                            f"[{serial[:8]}] NO SIGNAL",
                            (80, 235),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            vis,
                            "last_reject:no_frame",
                            (80, 265),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(f"CAM {serial}", vis)
                    continue

                # ArUco detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ids_list, corners_all = detect_tags(gray)

                # Optional raw video save (only when a new frame arrives)
                if args.save_raw_video and serial in new_color_frame_serials:
                    writer = video_writers.get(serial)
                    if writer is None and serial not in video_writer_failed:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*str(args.raw_video_codec)[:4])
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
                            print(f"\n[!] Failed to open raw video writer: {out_path}")
                            writer = None
                        else:
                            video_writers[serial] = writer
                    if writer is not None:
                        writer.write(frame)

                has_tags = ids_list is not None and len(ids_list) > 0
                if has_tags:
                    tags = ids_list
                    latest_tags[serial] = tags
                else:
                    tags = []

                # Build corners_per_id dict
                corners_per_id: dict[int, np.ndarray] = {}
                if has_tags and corners_all is not None:
                    for tag_id, corners in zip(ids_list, corners_all):
                        corners_per_id[int(tag_id)] = corners

                # Get intrinsics from cached camera metadata
                K = np.eye(3, dtype=np.float64)
                dist = np.zeros(5, dtype=np.float64)
                cam_meta = latest_cam_meta.get(serial)
                if cam_meta is not None:
                    K_arr = cam_meta.get("K")
                    if K_arr and isinstance(K_arr, list):
                        K = np.array(K_arr, dtype=np.float64)
                    dist_arr = cam_meta.get("dist")
                    if dist_arr and isinstance(dist_arr, list):
                        dist = np.array(dist_arr[:5], dtype=np.float64)

                # Add frame to calibrator
                now_t = time.monotonic()
                contrib = calib.add_frame(
                    serial,
                    now_t,
                    corners_per_id,
                    K,
                    dist,
                    T_world_cube,
                )
                if contrib:
                    frame_counts[serial] += 1

                # Display
                if args.show:
                    vis = frame.copy()
                    if has_tags:
                        cv2.aruco.drawDetectedMarkers(
                            vis, corners_all, np.array(ids_list, dtype=np.int32)
                        )
                    fc = frame_counts[serial]
                    last_reason = calib.get_last_reason(serial)
                    status = f"{serial[:8]} tags:{tags} obs:{fc} {ots_status}"
                    reason_status = f"last_reject:{last_reason}"
                    color_txt = (0, 255, 255) if has_tags else (160, 160, 160)
                    cv2.putText(
                        vis,
                        status,
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        color_txt,
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        vis,
                        reason_status,
                        (8, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 200, 255) if last_reason == "ok" else (0, 120, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(f"CAM {serial}", vis)

            # Auto-ready
            all_ready = all(count >= args.min_frames for count in frame_counts.values())
            if all_ready and not ready:
                ready = True
                print(
                    "[+] Minimum frame count reached. "
                    "Press q to finish early or keep collecting."
                )

            # Status bar
            status = "  ".join(f"{s[:6]}:{frame_counts[s]}" for s in serials)
            total = sum(frame_counts.values())
            status += f"  |  total:{total}  {ots_status}"
            if ready:
                status += "  [READY]"
            print(f"\r    {status}", end="", flush=True)

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        print(
            "\n[*] KeyboardInterrupt received, stopping capture and computing calibration..."
        )
    finally:
        # Stop adapters
        for adapter in sensor_mgr.realsense_adapters.values():
            try:
                await adapter.stop()
            except BaseException:
                pass
        if ots is not None:
            try:
                ots.stop()
            except BaseException:
                pass
        for writer in video_writers.values():
            try:
                writer.release()
            except BaseException:
                pass
        if args.save_raw_rgbd:
            for serial in serials:
                try:
                    flush_rgbd_chunks(serial, force=True)
                except BaseException:
                    pass
        cv2.destroyAllWindows()

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    print("\n\n[*] Computing calibration...")
    report = calib.calibration_report()

    for serial, count in report["n_frames_per_camera"].items():
        print(f"    [{serial}] {count} observations")

    print("\n--- Frame Stats ---")
    for serial in serials:
        s = report["frame_stats"].get(serial, {})
        print(
            "  [{}] recv={} ok={} rej_faces={} rej_ids={} rej_pnp={} "
            "rej_pose={} rej_world={} rej_reproj_nan={} rej_reproj_high={} rot_fallback={}".format(
                serial,
                s.get("received", 0),
                s.get("accepted", 0),
                s.get("reject_too_few_faces", 0),
                s.get("reject_no_valid_ids", 0),
                s.get("reject_solvepnp", 0),
                s.get("reject_non_finite_pose", 0),
                s.get("reject_non_finite_world", 0),
                s.get("reject_non_finite_reproj", 0),
                s.get("reject_high_reproj", 0),
                s.get("pnp_rotation_fallback", 0),
            )
        )

    failed = [s for s, T in report["T_world_cam"].items() if T is None]
    if failed:
        print(f"[!] Failed to estimate: {failed}")
    else:
        print("\n[OK] All cameras calibrated successfully.")

    # Print T_world_cam
    print("\n--- T_world_cam (world → camera) ---")
    for serial, T in report["T_world_cam"].items():
        if T is not None:
            t = T[:3, 3]
            print(
                f"  [{serial}]  t=[{t[0] * 1000:.1f}, {t[1] * 1000:.1f}, {t[2] * 1000:.1f}] mm"
            )

    # Save
    calib.save(output_json_path)

    # Save camera intrinsics metadata for replay mode.
    meta_path = run_dir / "camera_metadata.json"
    try:
        meta_path.write_text(
            json.dumps(_to_jsonable(latest_cam_meta), indent=2),
            encoding="utf-8",
        )
        print(f"[*] Camera metadata saved -> {meta_path}")
    except Exception as exc:
        print(f"[!] Failed to save camera metadata: {exc}")

    if args.save_raw_video:
        print(f"[*] Raw videos saved under: {run_dir}")
    if args.save_raw_rgbd:
        print(f"[*] Raw RGBD chunks saved under: {run_dir / 'rgbd'}")
    print("\n[+] Next step: load this JSON in FusionManager as known extrinsics.")


# ---------------------------------------------------------------------------
# Module guard / CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
