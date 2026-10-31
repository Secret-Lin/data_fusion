#!/usr/bin/env python3
"""
utils/tag_detector.py
~~~~~~~~~~~~~~~~~~~~~

Real-time multi-family ArUco/AprilTag detector using RealSense.
Detects all common tag families, shows annotated image and prints results.

Usage::

    conda activate mr_nav
    python utils/tag_detector.py

Controls::

    q / Esc  Quit
"""

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tag family catalogue — all common dictionaries
# ---------------------------------------------------------------------------

_TAG_FAMILIES = [
    # ArUco
    ("4X4_50", cv2.aruco.DICT_4X4_50, (0, 220, 0)),
    ("4X4_100", cv2.aruco.DICT_4X4_100, (0, 200, 0)),
    ("4X4_250", cv2.aruco.DICT_4X4_250, (0, 180, 0)),
    ("4X4_1000", cv2.aruco.DICT_4X4_1000, (0, 160, 0)),
    ("5X5_50", cv2.aruco.DICT_5X5_50, (0, 220, 220)),
    ("5X5_100", cv2.aruco.DICT_5X5_100, (0, 200, 200)),
    ("5X5_250", cv2.aruco.DICT_5X5_250, (0, 180, 180)),
    ("5X5_1000", cv2.aruco.DICT_5X5_1000, (0, 160, 160)),
    ("6X6_50", cv2.aruco.DICT_6X6_50, (220, 100, 0)),
    ("6X6_100", cv2.aruco.DICT_6X6_100, (200, 100, 0)),
    ("6X6_250", cv2.aruco.DICT_6X6_250, (180, 100, 0)),
    ("6X6_1000", cv2.aruco.DICT_6X6_1000, (160, 100, 0)),
    ("7X7_50", cv2.aruco.DICT_7X7_50, (220, 0, 220)),
    ("7X7_100", cv2.aruco.DICT_7X7_100, (200, 0, 200)),
    ("7X7_250", cv2.aruco.DICT_7X7_250, (180, 0, 180)),
    ("7X7_1000", cv2.aruco.DICT_7X7_1000, (160, 0, 160)),
    ("ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL, (255, 200, 0)),
    # AprilTag (OpenCV built-in)
    ("TAG16H5", cv2.aruco.DICT_APRILTAG_16h5, (0, 120, 255)),
    ("TAG25H9", cv2.aruco.DICT_APRILTAG_25h9, (0, 80, 255)),
    ("TAG36H10", cv2.aruco.DICT_APRILTAG_36H10, (120, 0, 255)),
    ("TAG36H11", cv2.aruco.DICT_APRILTAG_36H11, (180, 0, 255)),
]

# Name (no separator) → (constant, color) for fast lookup
_NAME_TO_ENTRY = {
    name.upper().replace("-", "").replace("_", ""): (const, color)
    for name, const, color in _TAG_FAMILIES
}

# All families in detection order (AprilTag first — most common)
ALL_FAMILIES = [name for name, _, _ in _TAG_FAMILIES]

# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


@dataclass
class TagDetection:
    family: str
    ids: np.ndarray  # shape (N,)
    corners: list[np.ndarray]  # list of (4,1,2) float arrays


@dataclass
class FrameResult:
    t_host: float
    tags: list[TagDetection] = field(default_factory=list)
    fps: float = 0.0


# ---------------------------------------------------------------------------
# RealSense camera
# ---------------------------------------------------------------------------


class RealSenseCamera:
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        import pyrealsense2 as _rs

        rs: Any = _rs

        # Resolve constants dynamically (stub-safe)
        # API is rs.stream.color / rs.stream.depth, NOT rs.stream_color
        self._stream_color = getattr(rs.stream, "color", None)
        self._stream_depth = getattr(rs.stream, "depth", None)

        if self._stream_color is None:
            raise RuntimeError(
                "pyrealsense2 is installed but stream constants are missing. "
                "Try: pip install --force-reinstall pyrealsense2"
            )

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(self._stream_color, width, height, rs.format.bgr8, fps)
        try:
            cfg.enable_stream(self._stream_depth, width, height, rs.format.z16, fps)
        except Exception:
            pass
        try:
            profile = self._pipeline.start(cfg)
            self._align = rs.align(rs.stream.color)
            self._intrinsics = None
            try:
                p = profile.get_stream(rs.stream.color)
                self._intrinsics = p.as_video_stream_profile().get_intrinsics()
            except Exception:
                pass
        except Exception as exc:
            raise RuntimeError(f"RealSense start failed: {exc}")

    def read(self) -> tuple[bool, Any]:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=2000)
            aligned = self._align.process(frames)
            color = aligned.get_color_frame()
            if not color:
                return False, None
            return True, np.asanyarray(color.as_frame().get_data())
        except Exception:
            return False, None

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Multi-family tag detector
# ---------------------------------------------------------------------------


class TagDetector:
    """Detect ArUco/AprilTag markers across all configured families."""

    def __init__(self, families: list[str] | None = None, confidence: int = 2):
        if families is None:
            families = list(ALL_FAMILIES)

        self._families: list[tuple[str, int, tuple[int, int, int]]] = []
        for name in families:
            key = name.upper().replace("-", "").replace("_", "")
            entry = _NAME_TO_ENTRY.get(key)
            if entry is None:
                print(f"[!] Unknown family '{name}', skipping.")
                continue
            canonical = next(
                n for n, _, _ in _TAG_FAMILIES if n.replace("_", "") == key
            )
            self._families.append((canonical, entry[0], entry[1]))

        self._confidence = max(1, confidence)

        # Build one detector per family
        self._detectors: list[tuple[str, Any, tuple[int, int, int]]] = []
        self._fallback: dict[str, Any] = {}
        for name, const, color in self._families:
            dictionary = cv2.aruco.getPredefinedDictionary(const)
            try:
                detector = cv2.aruco.ArucoDetector(
                    dictionary, cv2.aruco.DetectorParameters()
                )
            except AttributeError:
                detector = None
            self._detectors.append((name, detector, color))

        # Confidence counters per family
        self._counters: dict[str, int] = {name: 0 for name, _, _ in self._families}

    def process(self, frame_bgr: np.ndarray | None, t_host: float) -> FrameResult:
        if frame_bgr is None:
            return FrameResult(t_host=t_host)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        active: list[TagDetection] = []

        for name, detector, _ in self._detectors:
            try:
                if detector is not None:
                    corners, ids, _ = detector.detectMarkers(gray)
                else:
                    if name not in self._fallback:
                        const = next(c for n, c, _ in self._families if n == name)
                        self._fallback[name] = cv2.aruco.getPredefinedDictionary(const)
                    corners, ids, _ = cv2.aruco.detectMarkers(
                        gray, self._fallback[name]
                    )
            except Exception:
                corners, ids = None, None

            if ids is not None and len(ids) > 0 and corners is not None:
                self._counters[name] = min(self._confidence, self._counters[name] + 1)
                if self._counters[name] >= self._confidence:
                    active.append(TagDetection(family=name, ids=ids, corners=corners))
            else:
                self._counters[name] = max(0, self._counters[name] - 1)

        return FrameResult(t_host=t_host, tags=active)

    def draw(self, frame_bgr: np.ndarray | None, result: FrameResult) -> np.ndarray:
        if frame_bgr is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        out = frame_bgr.copy()

        for tag in result.tags:
            color = next(
                (c for n, _, c in self._families if n == tag.family),
                (255, 255, 0),
            )
            try:
                cv2.aruco.drawDetectedMarkers(out, tag.corners, tag.ids)
            except Exception:
                pass

            for i, corners in enumerate(tag.corners):
                pts = corners.reshape(-1, 2).astype(np.int32)
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())
                tag_id = int(tag.ids[i])
                label = f"{tag.family} #{tag_id}"
                cv2.putText(
                    out,
                    label,
                    (cx - 50, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.circle(out, (cx, cy), 4, color, -1)

        return out


# ---------------------------------------------------------------------------
# Banner overlay
# ---------------------------------------------------------------------------


def _build_banner(result: FrameResult, fps: float, width: int) -> np.ndarray:
    active = result.tags
    banner_h = 40
    banner = np.full((banner_h, width, 3), 18, dtype=np.uint8)
    parts: list[str] = []
    for tag in active:
        ids = ",".join(str(int(i)) for i in tag.ids)
        parts.append(f"{tag.family}: [{ids}]")
    label = " | ".join(parts) if parts else "(no tags)"
    label = f"Families: {label}    FPS {fps:.1f}"
    cv2.putText(
        banner,
        label[:120],
        (8, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return banner


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("MULTI-FAMILY TAG DETECTOR (RealSense)")
    print("=" * 60)
    print(f"  Families  : {', '.join(ALL_FAMILIES)}")
    print(f"  Count     : {len(ALL_FAMILIES)} dictionaries")
    print("=" * 60)

    # Camera
    try:
        cam = RealSenseCamera(width=1280, height=720, fps=30)
        print("[*] RealSense camera opened.")
    except Exception as exc:
        print(f"[!] Camera error: {exc}")
        return

    # Detector — all families
    detector = TagDetector(families=list(ALL_FAMILIES), confidence=2)

    fps = 0.0
    last_t = time.monotonic()
    frame_count = 0

    print("\n[*] Running — q / Esc to quit\n")

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                continue

            t_host = time.monotonic()
            result = detector.process(frame, t_host)
            frame_count += 1

            dt = t_host - last_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last_t = t_host

            # Print detection results
            if result.tags:
                for tag in result.tags:
                    ids = [int(i) for i in tag.ids]
                    print(f"  {tag.family:20s}  IDs: {ids}")

            # Display
            annotated = detector.draw(frame, result)
            banner = _build_banner(result, fps, annotated.shape[1])
            display = np.vstack([banner, annotated])
            cv2.imshow("Tag Detector", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()
        print(f"\n[*] {frame_count} frames processed. Done.")


if __name__ == "__main__":
    main()
