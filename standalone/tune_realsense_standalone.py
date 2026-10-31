#!/usr/bin/env python3
"""
RealSense tuning tool: live preview + adjust exposure/focus/resolution +
AprilTag cube detection overlay to verify tag visibility.

Controls:
  +/-        adjust exposure (step 100us)
  [/]        adjust focus (step 10)
  r          cycle resolution
  c          toggle auto-exposure on/off
  f          toggle auto-focus on/off
  Space      toggle AprilTag cube overlay
  s          save current parameters to JSON
  q          quit

Saved JSON can be loaded by other record scripts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


# ── Cube tag geometry (50mm cube, 40mm tags) ──
CUBE_HALF_MM = 25.0
TAG_HALF_MM = 20.0

_CUBE_FACE_CENTERS: dict[int, np.ndarray] = {
    0: np.array([+CUBE_HALF_MM, 0.0, 0.0], dtype=np.float64),
    1: np.array([0.0, -CUBE_HALF_MM, 0.0], dtype=np.float64),
    2: np.array([0.0, 0.0, -CUBE_HALF_MM], dtype=np.float64),
    3: np.array([-CUBE_HALF_MM, 0.0, 0.0], dtype=np.float64),
    4: np.array([0.0, 0.0, +CUBE_HALF_MM], dtype=np.float64),
    5: np.array([0.0, +CUBE_HALF_MM, 0.0], dtype=np.float64),
}

_DEFAULT_TAG_FACE_BASIS: dict[int, tuple[np.ndarray, np.ndarray]] = {
    0: (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    1: (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    2: (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
    3: (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    4: (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    5: (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
}


def tag_object_corners(tag_id: int, roll: int = 0) -> np.ndarray | None:
    c = _CUBE_FACE_CENTERS.get(tag_id)
    basis = _DEFAULT_TAG_FACE_BASIS.get(tag_id)
    if c is None or basis is None:
        return None
    u, v = basis
    obj = np.array([
        c + (-TAG_HALF_MM) * u + (+TAG_HALF_MM) * v,
        c + (+TAG_HALF_MM) * u + (+TAG_HALF_MM) * v,
        c + (+TAG_HALF_MM) * u + (-TAG_HALF_MM) * v,
        c + (-TAG_HALF_MM) * u + (-TAG_HALF_MM) * v,
    ], dtype=np.float64)
    r = int(roll) % 4
    return np.roll(obj, -r, axis=0) if r else obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RealSense tuning with cube tag preview")
    p.add_argument("--serial", default=None, help="Camera serial (auto if omitted)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    ctx = rs.context()
    devices = ctx.devices
    if not devices:
        print("[!] No RealSense camera found")
        return

    if args.serial:
        serial = args.serial
        found = any(dev.get_info(rs.camera_info.serial_number) == serial for dev in devices)
        if not found:
            print(f"[!] Serial {serial} not found")
            return
    else:
        serial = devices[0].get_info(rs.camera_info.serial_number)
        print(f"[i] Using camera: {serial}")

    # ── Resolution presets ──
    resolutions = [(640, 480), (1280, 720), (1920, 1080)]
    res_idx = 1  # start at 1280x720
    width, height = resolutions[res_idx]
    fps = 30

    # ── Build pipeline ──
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    # ── Sensors ──
    dev = profile.get_device()
    color_sensor = dev.first_color_sensor()
    depth_sensor = dev.first_depth_sensor()

    # ── Get current params ──
    current_exposure = color_sensor.get_option(rs.option.exposure)
    current_focus = color_sensor.get_option(rs.option.enable_auto_exposure)
    # Focus is only available when auto-exposure is off
    try:
        current_focus_val = color_sensor.get_option(rs.option.manual_focus)
    except Exception:
        current_focus_val = 0

    auto_exp = color_sensor.get_option(rs.option.enable_auto_exposure) != 0
    auto_focus = False
    try:
        auto_focus = color_sensor.get_option(rs.option.enable_auto_focus) != 0
    except Exception:
        pass

    # ── ArUco detector ──
    ar_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    ar_params = cv2.aruco.DetectorParameters()
    try:
        ar_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    except Exception:
        pass

    show_overlay = True
    saved = False

    print("=" * 60)
    print("RealSense tuning tool — keyboard controls:")
    print("  +/-       exposure (step 100)")
    print("  [/]       focus (step 10)")
    print("  r         cycle resolution")
    print("  c         toggle auto-exposure")
    print("  f         toggle auto-focus")
    print("  Space     toggle tag overlay")
    print("  s         save params to JSON")
    print("  q         quit")
    print("=" * 60)

    def set_exposure(val: float) -> None:
        try:
            color_sensor.set_option(rs.option.exposure, val)
        except Exception:
            pass

    def set_focus(val: float) -> None:
        try:
            color_sensor.set_option(rs.option.manual_focus, val)
        except Exception:
            pass

    def restart_pipeline(new_w: int, new_h: int) -> None:
        nonlocal width, height, profile
        pipe.stop()
        width, height = new_w, new_h
        cfg2 = rs.config()
        cfg2.enable_device(serial)
        cfg2.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg2.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = pipe.start(cfg2)
        nonlocal align
        align = rs.align(rs.stream.color)
        # Re-apply exposure/focus after restart
        if not auto_exp:
            set_exposure(current_exposure)
        if not auto_focus:
            set_focus(current_focus_val)

    try:
        while True:
            frames = pipe.wait_for_frames(timeout_ms=1000)
            if frames is None:
                continue
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asarray(color_frame.get_data()).copy()
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            # ── AprilTag detection ──
            tag_ids_detected: list[int] = []
            overlay = color.copy()
            if show_overlay:
                try:
                    corners, ids, _ = cv2.aruco.detectMarkers(gray, ar_dict, parameters=ar_params)
                    if ids is not None and len(ids) > 0:
                        cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
                        for i, id_arr in enumerate(ids):
                            tid = int(id_arr[0])
                            if 0 <= tid <= 5:
                                tag_ids_detected.append(tid)
                                # Draw tag ID
                                c = corners[i][0]
                                cx = int(round(float(np.mean(c[:, 0]))))
                                cy = int(round(float(np.mean(c[:, 1]))))
                                cv2.putText(overlay, f"id={tid}", (cx - 10, cy - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception:
                    pass

            # ── HUD ──
            lines = [
                f"RES: {width}x{height}",
                f"EXP: {current_exposure:.0f}  {'A' if auto_exp else 'M'}",
                f"FOCUS: {current_focus_val:.0f}  {'A' if auto_focus else 'M'}",
                f"TAGS: {len(tag_ids_detected)} {tag_ids_detected}",
                "[s]ave  [q]uit",
            ]
            for i, txt in enumerate(lines):
                cv2.putText(overlay, txt, (8, 24 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(f"Tune {serial[:8]}", overlay)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("+") or key == ord("="):
                current_exposure = min(current_exposure + 100, 10000)
                set_exposure(current_exposure)
                auto_exp = False
            elif key == ord("-") or key == ord("_"):
                current_exposure = max(current_exposure - 100, 1)
                set_exposure(current_exposure)
                auto_exp = False
            elif key == ord("]"):
                current_focus_val = min(current_focus_val + 10, 250)
                set_focus(current_focus_val)
                auto_focus = False
            elif key == ord("["):
                current_focus_val = max(current_focus_val - 10, 0)
                set_focus(current_focus_val)
                auto_focus = False
            elif key == ord("r"):
                res_idx = (res_idx + 1) % len(resolutions)
                nw, nh = resolutions[res_idx]
                print(f"[i] Resolution: {nw}x{nh}")
                restart_pipeline(nw, nh)
            elif key == ord("c"):
                auto_exp = not auto_exp
                color_sensor.set_option(rs.option.enable_auto_exposure, 1 if auto_exp else 0)
                if not auto_exp:
                    set_exposure(current_exposure)
                print(f"[i] Auto-exposure: {'ON' if auto_exp else 'OFF'}")
            elif key == ord("f"):
                auto_focus = not auto_focus
                try:
                    color_sensor.set_option(rs.option.enable_auto_focus, 1 if auto_focus else 0)
                except Exception:
                    auto_focus = not auto_focus
                if not auto_focus:
                    set_focus(current_focus_val)
                print(f"[i] Auto-focus: {'ON' if auto_focus else 'OFF'}")
            elif key == ord(" "):
                show_overlay = not show_overlay
            elif key == ord("s"):
                params = {
                    "serial": serial,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "exposure": float(current_exposure),
                    "auto_exposure": bool(auto_exp),
                    "focus": float(current_focus_val),
                    "auto_focus": bool(auto_focus),
                }
                out_path = Path(f"realsense_params_{serial[:8]}.json")
                out_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
                print(f"[OK] Params saved to {out_path}")
                saved = True

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    if not saved:
        ans = input("\nSave current params before exit? [y/N]: ").strip().lower()
        if ans == "y":
            params = {
                "serial": serial,
                "width": width,
                "height": height,
                "fps": fps,
                "exposure": float(current_exposure),
                "auto_exposure": bool(auto_exp),
                "focus": float(current_focus_val),
                "auto_focus": bool(auto_focus),
            }
            out_path = Path(f"realsense_params_{serial[:8]}.json")
            out_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
            print(f"[OK] Params saved to {out_path}")


if __name__ == "__main__":
    main()
