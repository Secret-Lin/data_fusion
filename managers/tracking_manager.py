# managers/tracking_manager.py
# pyright: reportMissingTypeArgument=false, reportConstantRedefinition=false

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Any

import time

from common.structured_log import slog

import numpy as np
import cv2

from core.event_bus import EventBus
from core.coordinate_frames import normalize_coordinate_system, transform_to_y_down
from core.types import TimeStampedPacket, DetectionResult
from managers.tracking_inputs import TrackingRGBDInput

from adapters.dope_runner import DopeRunner

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


@dataclass
class CubeModel:
    """Calibrated tag-cube layout loaded from cube_model.yaml.

    World frame = cube centre, +Z outward through Tag 0 (see cube_model.yaml).
    T_world_tag maps tag-local corner coordinates into the cube world frame.
    """

    source_path: str
    cube_size_m: float
    tag_size_m: float
    T_world_tag: dict[int, np.ndarray]  # tag id -> 4x4

    def world_tag_corners(self, tag_id: int) -> np.ndarray:
        half = self.tag_size_m / 2.0
        # OpenCV ArUco corner order: top-left, top-right, bottom-right, bottom-left
        local = np.array(
            [
                [-half, -half, 0.0],
                [+half, -half, 0.0],
                [+half, +half, 0.0],
                [-half, +half, 0.0],
            ],
            dtype=np.float64,
        )
        T = self.T_world_tag[int(tag_id)]
        homogeneous = np.column_stack((local, np.ones(4, dtype=np.float64)))
        return (T @ homogeneous.T).T[:, :3]


def load_cube_model(path: str | Path) -> CubeModel | None:
    """Load cube_model.yaml; return None if unavailable or invalid."""
    if yaml is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    tags = data.get("tags")
    if not isinstance(tags, dict):
        return None
    transforms: dict[int, np.ndarray] = {}
    try:
        coordinate_system = normalize_coordinate_system(
            data.get("coordinate_system", data.get("coord", "y_down"))
        )
    except ValueError:
        return None
    for key, entry in tags.items():
        try:
            tag_id = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(entry, dict) or "T_world_tag" not in entry:
            continue
        T = np.asarray(entry["T_world_tag"], dtype=np.float64)
        if T.shape == (4, 4) and np.all(np.isfinite(T)):
            transforms[tag_id] = transform_to_y_down(T, coordinate_system)
    if not transforms:
        return None
    return CubeModel(
        source_path=str(p),
        cube_size_m=float(data.get("cube_size_m", 0.05)),
        tag_size_m=float(data.get("tag_size_m", 0.04)),
        T_world_tag=transforms,
    )


_DOPE_CUBOID_EDGES = [
    (1, 0),
    (0, 3),
    (3, 2),
    (2, 1),
    (5, 4),
    (4, 7),
    (7, 6),
    (6, 5),
    (2, 6),
    (1, 5),
    (3, 7),
    (0, 4),
]


@dataclass
class RGBDFeatureFrame:
    descriptors: np.ndarray
    keypoints_uv: np.ndarray
    points3d: np.ndarray
    n_keypoints_raw: int


@dataclass
class RGBDVoStep:
    T_prev_curr: np.ndarray | None
    score: float
    reproj_error: float | None
    n_matches: int
    n_inliers: int


@dataclass
class RGBDVoState:
    pose_world_camera: np.ndarray
    last_frame: RGBDFeatureFrame | None = None


def run_dope_sync(model: DopeRunner, frame_bgr: np.ndarray, camera: dict | None):
    det = model.infer(frame_bgr, camera)
    return det


def _resolve_aruco_dictionary(name: str):
    """Map user-friendly names to OpenCV predefined ArUco/AprilTag dictionaries."""
    if not hasattr(cv2, "aruco"):
        raise ImportError("OpenCV ArUco module not found (cv2.aruco)")

    raw = (name or "").strip()
    key = raw.lower().replace("-", "").replace("_", "")

    mapping = {
        # ArUco
        "4x450": cv2.aruco.DICT_4X4_50,
        "4x4100": cv2.aruco.DICT_4X4_100,
        "4x4250": cv2.aruco.DICT_4X4_250,
        "4x41000": cv2.aruco.DICT_4X4_1000,
        "5x550": cv2.aruco.DICT_5X5_50,
        "5x5100": cv2.aruco.DICT_5X5_100,
        "5x5250": cv2.aruco.DICT_5X5_250,
        "5x51000": cv2.aruco.DICT_5X5_1000,
        "6x650": cv2.aruco.DICT_6X6_50,
        "6x6100": cv2.aruco.DICT_6X6_100,
        "6x6250": cv2.aruco.DICT_6X6_250,
        "6x61000": cv2.aruco.DICT_6X6_1000,
        "7x750": cv2.aruco.DICT_7X7_50,
        "7x7100": cv2.aruco.DICT_7X7_100,
        "7x7250": cv2.aruco.DICT_7X7_250,
        "7x71000": cv2.aruco.DICT_7X7_1000,
        "arucooriginal": cv2.aruco.DICT_ARUCO_ORIGINAL,
        # AprilTag (OpenCV built-in dictionaries)
        "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }

    if key in mapping:
        return cv2.aruco.getPredefinedDictionary(mapping[key]), raw

    # Allow passing OpenCV constant name like DICT_6X6_250
    if raw.startswith("DICT_"):
        const = getattr(cv2.aruco, raw, None)
        if const is not None:
            return cv2.aruco.getPredefinedDictionary(const), raw

    raise ValueError(
        f"Unknown aruco/apriltag family '{name}'. "
        "Try one of: 4x4_50, 5x5_100, 6x6_250, 7x7_1000, aruco_original, tag36h11, tag25h9, tag16h5, tag36h10."
    )


class TrackingManager:
    """
    Subscriptions should match SensorManager topics:
      - RealSense:  realsense_stream
      - HL2 AHAT:   hl2/ahat
      - (optional) HL2 PV:   hl2/pv
    """

    def __init__(
        self,
        bus: EventBus,
        loop,
        model: Optional[Any] = None,
        use_realsense_for_dope: bool = True,
        use_hl2_pv_for_dope: bool = False,
        use_quest_for_dope: bool = False,
        enable_ahat_markers: bool = True,
        enable_realsense_apriltag: bool = True,
        apriltag_tag_size_m: float = 0.04,
        apriltag_families: list[str] | None = None,
        apriltag_show_preview: bool = False,
        dope_show_preview: bool = False,
        realsense_topics: list[str] | None = None,
        quest_topics: list[str] | None = None,
        enable_rgbd_vo: bool = False,
        calibration_result_path: str | None = None,
        cube_model_path: str | None = None,
        sensor_modes: dict[str, str] | None = None,
        quest_world_extrinsic_path: str | None = None,
        quest_tag_correction_alpha: float = 0.2,
        quest_tag_correction_max_translation_m: float = 0.25,
        quest_tag_correction_max_rotation_deg: float = 30.0,
    ):
        self.bus = bus
        self.loop = loop
        self.model = model
        self._calibration_result_path = str(calibration_result_path or "").strip()
        self._sensor_modes = {str(k): str(v).lower() for k, v in (sensor_modes or {}).items()}
        self._quest_tag_correction_alpha = float(np.clip(quest_tag_correction_alpha, 0.0, 1.0))
        self._quest_tag_correction_max_translation_m = float(quest_tag_correction_max_translation_m)
        self._quest_tag_correction_max_rotation_deg = float(quest_tag_correction_max_rotation_deg)

        # Calibrated cube model (cube_model.yaml); None disables cube-model tracking.
        self._cube_model: CubeModel | None = None
        cube_model_str = str(cube_model_path or "").strip()
        if cube_model_str:
            self._cube_model = load_cube_model(cube_model_str)
            if self._cube_model is None:
                slog(
                    component="TrackingManager",
                    event="cube_model_load_failed",
                    path=cube_model_str,
                )

        self.use_realsense_for_dope = use_realsense_for_dope
        self.use_hl2_pv_for_dope = use_hl2_pv_for_dope
        self.use_quest_for_dope = use_quest_for_dope
        self.enable_ahat_markers = enable_ahat_markers

        self.enable_realsense_apriltag = enable_realsense_apriltag
        self.apriltag_tag_size_m = float(apriltag_tag_size_m)
        self.apriltag_families = apriltag_families or ["TAG16H5"]
        self.apriltag_show_preview = apriltag_show_preview

        self.dope_show_preview = dope_show_preview
        self.enable_rgbd_vo = enable_rgbd_vo

        self.realsense_topics = realsense_topics or ["realsense_stream"]
        self.quest_topics = quest_topics or []

        # Low-latency knobs
        self._dope_min_interval_s: float = 0.1  # 10 Hz per camera
        self._last_dope_ts: dict[str, float] = {}
        self._drop_backlog = True
        self._inflight_packets = 0

        # Per-camera overlay windows.
        self._overlay_preview_enabled = bool(apriltag_show_preview or dope_show_preview)
        self._overlay_preview_images: dict[str, np.ndarray] = {}
        self._overlay_preview_task: Optional[asyncio.Task] = None

        # ArUco detector (OpenCV contrib).
        if not hasattr(cv2, "aruco"):
            raise ImportError(
                "OpenCV ArUco module not found (cv2.aruco). Install opencv-contrib-python in your environment."
            )

        self._aruco_dict, self._aruco_dict_name = _resolve_aruco_dictionary(
            self.apriltag_families[0]
        )
        self._aruco_params = cv2.aruco.DetectorParameters()

        # Multi-family: one detector per family
        self._aruco_detectors: list[tuple[str, Any, Any]] = []
        for fam in self.apriltag_families:
            d, name = _resolve_aruco_dictionary(fam)
            params = cv2.aruco.DetectorParameters()
            try:
                det = cv2.aruco.ArucoDetector(d, params)
            except Exception:
                det = None
            self._aruco_detectors.append((name, det, d))
        self._apriltag_confidence: dict[str, int] = {
            f: 0 for f in self.apriltag_families
        }

        # ---------- subscriptions ----------
        # Always subscribe, then conditionally process based on flags.
        # This lets the manager "try" all inputs and only do work when packets arrive.
        self.realsense_queues = [bus.subscribe(t) for t in self.realsense_topics]
        self.quest_queues = [bus.subscribe(t) for t in self.quest_topics]
        self.hl2_pv_queue = bus.subscribe("hl2/pv")
        self.hl2_ahat_queue = bus.subscribe("hl2/ahat")

        # Quest tag-cube reference, keyed by quest source:
        #   _quest_cube_ref[src] = T_quest_cube (camera-frame cube pose at the ref frame)
        #   _quest_odom_ref[src] = T_world_quest  (quest odometry at the ref frame)
        # Refreshed every frame a cube is detected; used to re-project Quest DOPE
        # detections into the tag-cube frame via odometry extrapolation.
        self._quest_cube_ref: dict[str, np.ndarray] = {}
        self._quest_odom_ref: dict[str, np.ndarray] = {}
        self._T_world_quest: np.ndarray | None = self._load_quest_world_extrinsic(
            quest_world_extrinsic_path
        )
        # Calibrated fixed-camera extrinsics; missing entries are never replaced
        # with identity.
        self._T_world_camera: dict[str, np.ndarray] = {}
        self._reproj_error_px: dict[str, float] = {}
        self._load_static_extrinsics_from_config()

        # ---------- threadpools ----------
        # Use more workers to allow multiple cameras to process concurrently without blocking each other.
        total_streams = max(2, len(self.realsense_topics) + len(self.quest_topics))
        self._dope_executor = ThreadPoolExecutor(
            max_workers=total_streams
        )
        self._apriltag_executor = ThreadPoolExecutor(
            max_workers=total_streams
        )
        self._vo_executor = ThreadPoolExecutor(
            max_workers=total_streams
        )

        # ---------- RGB-D visual odometry ----------
        self._vo_orb_nfeatures = 1200
        self._vo_min_valid_features = 60
        self._vo_min_matches = 20
        self._vo_match_ratio = 0.75
        self._vo_ransac_reproj_px = 3.0
        self._vo_ransac_iters = 100
        self._vo_states: dict[str, RGBDVoState] = {}

        # ---------- marker tracking state (keep your old logic) ----------
        self.tracked_markers = []
        self.SMOOTHING_WINDOW_SIZE = 10
        self.MATCHING_DISTANCE_THRESHOLD = 0.05
        self.frame_idx = 0

        self.log_enabled = True
        self._publish_vis = False
        self._vis_topic = "visualization/frames"
        self._publish_dope_vis = True
        self._publish_apriltag_vis = True

    def set_visualization_stream(
        self,
        *,
        enabled: bool,
        topic: str = "visualization/frames",
        publish_dope: bool = True,
        publish_apriltag: bool = True,
    ) -> None:
        self._publish_vis = bool(enabled)
        self._vis_topic = str(topic)
        self._publish_dope_vis = bool(publish_dope)
        self._publish_apriltag_vis = bool(publish_apriltag)

    def configure_runtime(
        self, *, drop_old_frames: bool = True, dope_min_interval_s: float = 0.1
    ) -> None:
        self._drop_backlog = bool(drop_old_frames)
        self._dope_min_interval_s = max(float(dope_min_interval_s), 0.0)
    def _serial_from_source(self, source: str) -> str | None:
        s = str(source or "").strip()

        for part in s.split("/"):
            if part.isdigit() and len(part) >= 6:
                return part

        if s.isdigit():
            return s

        return None
    def _sensor_mode(self, source: str) -> str:
        serial = self._serial_from_source(source)
        for key in (source, serial):
            if key and key in self._sensor_modes:
                mode = self._sensor_modes[key]
                return "mobile" if mode == "move" else mode
        return "mobile" if str(source).startswith("quest") else "fixed"

    def _lookup_T_world_cam(self, source: str) -> np.ndarray | None:
        serial = self._serial_from_source(source)

        candidate_keys = []

        if serial is not None:
            candidate_keys.extend([
                serial,
                f"realsense/{serial}",
                f"realsense/{serial}/stream",
            ])

        candidate_keys.append(str(source))

        for k in candidate_keys:
            T = self._T_world_camera.get(k)
            if T is not None:
                return T

        # slog(
        #     component=self.__class__.__name__,
        #     event="static_extrinsics_not_found_for_source",
        #     source=str(source),
        #     serial=serial,
        #     available=list(self._T_world_camera.keys()),
        # )

        return None
    def pending_input_count(self) -> int:
        total = int(self._inflight_packets)
        for q in self.realsense_queues:
            total += int(q.qsize())
        for q in self.quest_queues:
            total += int(q.qsize())
        total += int(self.hl2_pv_queue.qsize())
        total += int(self.hl2_ahat_queue.qsize())
        return total

    async def run(self):
        slog(
            component=self.__class__.__name__,
            event="run",
            use_realsense_for_dope=self.use_realsense_for_dope,
            use_hl2_pv_for_dope=self.use_hl2_pv_for_dope,
            enable_ahat_markers=self.enable_ahat_markers,
            enable_realsense_apriltag=self.enable_realsense_apriltag,
            apriltag_tag_size_m=self.apriltag_tag_size_m,
            apriltag_families=self.apriltag_families,
            aruco_dict=self._aruco_dict_name,
            apriltag_show_preview=self.apriltag_show_preview,
            dope_show_preview=self.dope_show_preview,
            enable_rgbd_vo=self.enable_rgbd_vo,
            topic_vis=(self._vis_topic if self._publish_vis else None),
            drop_old_frames=bool(self._drop_backlog),
            dope_min_interval_s=float(self._dope_min_interval_s),
        )

        if self._overlay_preview_enabled:
            self._overlay_preview_task = self.loop.create_task(
                self._overlay_preview_loop()
            )
        # Pre-create persistent reader tasks (eliminate task create/cancel per frame).
        all_queues: list[asyncio.Queue] = []
        all_queues.extend(self.realsense_queues)
        all_queues.extend(self.quest_queues)
        if self.hl2_pv_queue is not None:
            all_queues.append(self.hl2_pv_queue)
        if self.hl2_ahat_queue is not None:
            all_queues.append(self.hl2_ahat_queue)

        self._ready_queue: asyncio.Queue = asyncio.Queue()
        self._reader_tasks: list[asyncio.Task] = []

        async def _forward_to_ready(q):
            while True:
                pkt = await q.get()
                await self._ready_queue.put(pkt)

        for q in all_queues:
            self._reader_tasks.append(asyncio.create_task(_forward_to_ready(q)))

        # One loop, react to whichever stream has data.
        while True:
            pkt = await self._ready_queue.get()
            if pkt is None:
                continue

            # Collect latest packet per source from the ready queue.
            # This naturally drains backlog: older frames from the same source are discarded.
            latest_by_source: dict[str, Any] = {}
            source_serial = getattr(pkt, "source", "") or ""
            latest_by_source[source_serial] = pkt

            if self._drop_backlog and source_serial.startswith(("realsense", "quest")):
                while True:
                    try:
                        nxt = self._ready_queue.get_nowait()
                        src = getattr(nxt, "source", "") or ""
                        cur = latest_by_source.get(src)
                        if cur is None or getattr(nxt, "t_host", 0) > getattr(cur, "t_host", 0):
                            latest_by_source[src] = nxt
                    except asyncio.QueueEmpty:
                        break
            else:
                latest_by_source[source_serial] = pkt

            for src, latest_pkt in latest_by_source.items():
                self._inflight_packets += 1
                try:
                    await self._handle_packet(latest_pkt)
                except Exception as e:
                    slog(
                        component=self.__class__.__name__,
                        event="packet_error",
                        reason=repr(e),
                    )
                finally:
                    self._inflight_packets = max(0, self._inflight_packets - 1)

    async def _overlay_preview_loop(self) -> None:
        if not self._overlay_preview_enabled:
            return
        while True:
            try:
                for key, img in list(self._overlay_preview_images.items()):
                    if img is None:
                        continue
                    window = f"realsense_overlay_{key}".replace(":", "_").replace(
                        "/", "_"
                    )
                    cv2.imshow(window, img)
                cv2.waitKey(1)
            except Exception as e:
                slog(
                    component=self.__class__.__name__,
                    event="overlay_preview_disabled",
                    reason=repr(e),
                )
                self._overlay_preview_enabled = False
                return
            await asyncio.sleep(1.0 / 30.0)

    def _draw_dope_on_image(self, out_bgr: np.ndarray, dope_det) -> None:
        if dope_det is None or dope_det.projected_points is None:
            return

        pts = np.array(dope_det.projected_points, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 8:
            return

        s = float(getattr(dope_det, "scale_factor", 1.0) or 1.0)
        if s != 1.0:
            pts = pts / s

        pts_i = np.round(pts).astype(int)
        for a, b in _DOPE_CUBOID_EDGES:
            pa = tuple(pts_i[a])
            pb = tuple(pts_i[b])
            cv2.line(out_bgr, pa, pb, (0, 255, 0), 2)

        try:
            score = float(getattr(dope_det, "score", 0.0))
        except Exception:
            score = 0.0
        cv2.putText(
            out_bgr,
            f"DOPE score={score:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _world_tag_blocks(
        tag_corners_by_id: dict[int, np.ndarray],
        model: CubeModel,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Build object/image point blocks for all visible tags from the cube model."""
        object_blocks: list[np.ndarray] = []
        image_blocks: list[np.ndarray] = []
        used_ids: list[int] = []
        for tag_id in sorted(tag_corners_by_id.keys()):
            if int(tag_id) not in model.T_world_tag:
                continue
            object_blocks.append(model.world_tag_corners(int(tag_id)))
            image_blocks.append(
                np.asarray(tag_corners_by_id[tag_id], dtype=np.float64).reshape(4, 2)
            )
            used_ids.append(int(tag_id))
        if not object_blocks:
            return np.empty((0, 3)), np.empty((0, 2)), []
        return np.vstack(object_blocks), np.vstack(image_blocks), used_ids

    @staticmethod
    def _reprojection_errors(
        objp: np.ndarray,
        imgp: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> np.ndarray:
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        return np.linalg.norm(projected.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1)

    def _estimate_cube_pose_with_model(
        self,
        tag_corners_by_id: dict[int, np.ndarray],
        K: np.ndarray,
        dist: np.ndarray,
        model: CubeModel | None,
    ) -> tuple[np.ndarray | None, float]:
        """Estimate T_cam_world from visible AprilTags using the calibrated cube model.

        Uses the calibrated T_world_tag layout (no hard-coded face mapping, no
        corner-roll search). A single visible tag is solved with IPPE (which
        explicitly handles the two-solution ambiguity of planar points);
        multiple tags use SQPNP RANSAC followed by iterative refinement.

        Returns (T_cam_world 4x4, rms reprojection px) or (None, inf).
        """
        if not tag_corners_by_id or K is None or model is None:
            return None, float("inf")
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        objp, imgp, used_ids = self._world_tag_blocks(tag_corners_by_id, model)
        if not used_ids:
            return None, float("inf")
        try:
            if len(used_ids) == 1:
                result = cv2.solvePnPGeneric(objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE)
                if not result or not bool(result[0]):
                    return None, float("inf")
                best = None
                best_rms = float("inf")
                for rvec, tvec in zip(result[1], result[2]):
                    errs = self._reprojection_errors(objp, imgp, rvec, tvec, K, dist)
                    rms = float(np.sqrt(np.mean(errs * errs)))
                    if rms < best_rms:
                        best_rms = rms
                        best = (rvec, tvec)
                if best is None:
                    return None, float("inf")
                rvec, tvec = best
            else:
                try:
                    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                        objp, imgp, K, dist,
                        flags=cv2.SOLVEPNP_SQPNP,
                        confidence=0.999,
                        reprojectionError=4.0,
                        iterationsCount=300,
                    )
                except cv2.error:
                    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                        objp, imgp, K, dist,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                        confidence=0.999,
                        reprojectionError=4.0,
                        iterationsCount=300,
                    )
                if not ok or rvec is None or tvec is None:
                    return None, float("inf")
                try:
                    refined, rv2, tv2 = cv2.solvePnP(
                        objp, imgp, K, dist,
                        rvec=rvec, tvec=tvec,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                    if refined:
                        rvec, tvec = rv2, tv2
                except cv2.error:
                    pass
        except cv2.error:
            return None, float("inf")
        T_cam_world = np.eye(4, dtype=np.float64)
        T_cam_world[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cam_world[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
        errs = self._reprojection_errors(objp, imgp, rvec, tvec, K, dist)
        return T_cam_world, float(np.sqrt(np.mean(errs * errs)))

    def _detect_aruco_and_draw(
        self, frame_bgr: np.ndarray, camera: dict | None
    ) -> tuple[np.ndarray, int, dict[int, np.ndarray]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        K = None
        dist = None
        if isinstance(camera, dict):
            K = camera.get("K")
            dist = camera.get("dist")
        if K is not None:
            K = np.array(K, dtype=np.float64)
        if dist is not None:
            dist = np.array(dist, dtype=np.float64)

        fx = fy = cx = cy = None
        if K is not None:
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])

        if dist is None:
            dist = np.zeros((5,), dtype=np.float64)

        out = frame_bgr.copy()

        # Multi-family detection: try each family, accumulate results
        all_corners: list[Any] = []
        all_ids: list[int] = []
        all_family_names: list[str] = []
        tag_corners_by_id: dict[int, np.ndarray] = {}

        for fam_name, detector, aruco_dict in self._aruco_detectors:
            try:
                if detector is not None:
                    corners_f, ids_f, _rejected = detector.detectMarkers(gray)
                else:
                    corners_f, ids_f, _rejected = cv2.aruco.detectMarkers(
                        gray, aruco_dict, parameters=self._aruco_params
                    )
            except Exception:
                corners_f, ids_f = None, None

            if ids_f is not None and len(ids_f) > 0 and corners_f is not None:
                self._apriltag_confidence[fam_name] = min(
                    2, self._apriltag_confidence.get(fam_name, 0) + 1
                )
                if self._apriltag_confidence.get(fam_name, 0) >= 1:
                    all_corners.extend(corners_f)
                    all_ids.extend(int(i) for i in ids_f)
                    all_family_names.extend([fam_name] * len(ids_f))
                    for c, tid in zip(corners_f, ids_f):
                        tag_corners_by_id[int(tid)] = np.asarray(c, dtype=np.float64).reshape(4, 2)
            else:
                self._apriltag_confidence[fam_name] = max(
                    0, self._apriltag_confidence.get(fam_name, 0) - 1
                )

        if not all_corners:
            return out, 0, {}

        # Draw all detected markers
        if all_ids:
            cv2.aruco.drawDetectedMarkers(out, all_corners, np.array(all_ids))
            # Label each marker with its family
            for i, (corners_i, tag_id, fam) in enumerate(
                zip(all_corners, all_ids, all_family_names)
            ):
                pts = corners_i.reshape(-1, 2).astype(np.int32)
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())
                label = f"{fam} #{tag_id}"
                cv2.putText(
                    out,
                    label,
                    (cx - 30, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.circle(out, (cx, cy), 3, (0, 255, 255), -1)

        # Pose estimation
        if K is not None:
            s = float(self.apriltag_tag_size_m)
            half = s * 0.5
            objp = np.array(
                [
                    [-half, half, 0.0],
                    [half, half, 0.0],
                    [half, -half, 0.0],
                    [-half, -half, 0.0],
                ],
                dtype=np.float64,
            )
            pnp_flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", None)
            if pnp_flag is None:
                pnp_flag = cv2.SOLVEPNP_ITERATIVE
            for c in all_corners:
                imgp = np.array(c, dtype=np.float64).reshape(4, 2)
                ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=pnp_flag)
                if ok:
                    cv2.drawFrameAxes(out, K, dist, rvec, tvec, s * 0.5)

        return out, len(all_ids), tag_corners_by_id

    @staticmethod
    def _parse_camera_model(
        camera: dict | None,
    ) -> tuple[np.ndarray | None, np.ndarray, float]:
        K = None
        dist = np.zeros((5,), dtype=np.float64)
        depth_scale_m = 0.001

        if not isinstance(camera, dict):
            return K, dist, depth_scale_m

        K_raw = camera.get("K")
        if K_raw is not None:
            try:
                K = np.array(K_raw, dtype=np.float64).reshape(3, 3)
                if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
                    K = None
            except Exception:
                K = None

        dist_raw = camera.get("dist")
        if dist_raw is not None:
            try:
                dist = np.array(dist_raw, dtype=np.float64).reshape(-1)
            except Exception:
                dist = np.zeros((5,), dtype=np.float64)

        ds_raw = camera.get("depth_scale_m", 0.001)
        try:
            depth_scale_m = float(ds_raw)
        except Exception:
            depth_scale_m = 0.001
        if depth_scale_m <= 0.0:
            depth_scale_m = 0.001

        return K, dist, depth_scale_m

    def _extract_rgbd_features_sync(
        self,
        frame_bgr: np.ndarray,
        depth_u16: np.ndarray,
        K: np.ndarray,
        depth_scale_m: float,
    ) -> RGBDFeatureFrame | None:
        if depth_u16.ndim != 2:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        orb_create_name = "ORB_create"
        orb_create = getattr(cv2, orb_create_name, None)
        if orb_create is None:
            return None
        orb = orb_create(nfeatures=self._vo_orb_nfeatures)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) == 0:
            return None

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        if fx <= 1e-9 or fy <= 1e-9:
            return None

        h, w = depth_u16.shape
        valid_desc: list[np.ndarray] = []
        valid_uv: list[tuple[float, float]] = []
        valid_pts3d: list[tuple[float, float, float]] = []

        for idx, kp in enumerate(keypoints):
            u = int(round(kp.pt[0]))
            v = int(round(kp.pt[1]))
            if u < 0 or v < 0 or u >= w or v >= h:
                continue

            depth_raw = float(depth_u16[v, u])
            if depth_raw <= 0.0:
                continue

            z = depth_raw * depth_scale_m
            if z <= 0.0 or not np.isfinite(z):
                continue

            x = (float(kp.pt[0]) - cx) * z / fx
            y = (float(kp.pt[1]) - cy) * z / fy
            if not np.isfinite(x) or not np.isfinite(y):
                continue

            valid_desc.append(descriptors[idx])
            valid_uv.append((float(kp.pt[0]), float(kp.pt[1])))
            valid_pts3d.append((x, y, z))

        if len(valid_desc) < self._vo_min_valid_features:
            return None

        return RGBDFeatureFrame(
            descriptors=np.array(valid_desc, dtype=np.uint8),
            keypoints_uv=np.array(valid_uv, dtype=np.float64),
            points3d=np.array(valid_pts3d, dtype=np.float64),
            n_keypoints_raw=int(len(keypoints)),
        )

    def _estimate_rgbd_vo_step_sync(
        self,
        prev_frame: RGBDFeatureFrame,
        curr_frame: RGBDFeatureFrame,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> RGBDVoStep:
        if prev_frame.descriptors.shape[0] < self._vo_min_matches:
            return RGBDVoStep(None, 0.0, None, 0, 0)
        if curr_frame.descriptors.shape[0] < self._vo_min_matches:
            return RGBDVoStep(None, 0.0, None, 0, 0)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = matcher.knnMatch(prev_frame.descriptors, curr_frame.descriptors, k=2)

        good_matches = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self._vo_match_ratio * n.distance:
                good_matches.append(m)

        n_matches = len(good_matches)
        if n_matches < self._vo_min_matches:
            return RGBDVoStep(None, 0.0, None, n_matches, 0)

        obj_pts = np.array(
            [prev_frame.points3d[m.queryIdx] for m in good_matches],
            dtype=np.float64,
        )
        img_pts = np.array(
            [curr_frame.keypoints_uv[m.trainIdx] for m in good_matches],
            dtype=np.float64,
        )

        if obj_pts.shape[0] < 6:
            return RGBDVoStep(None, 0.0, None, n_matches, 0)

        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=obj_pts,
                imagePoints=img_pts,
                cameraMatrix=K,
                distCoeffs=dist,
                iterationsCount=self._vo_ransac_iters,
                reprojectionError=self._vo_ransac_reproj_px,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            return RGBDVoStep(None, 0.0, None, n_matches, 0)

        if not ok or inliers is None or len(inliers) < 4:
            return RGBDVoStep(None, 0.0, None, n_matches, 0)

        inlier_ids = inliers.reshape(-1).astype(np.int64)
        n_inliers = int(inlier_ids.size)

        R_curr_prev, _ = cv2.Rodrigues(rvec)
        t_curr_prev = np.array(tvec, dtype=np.float64).reshape(3)

        T_prev_curr = np.eye(4, dtype=np.float64)
        T_prev_curr[:3, :3] = R_curr_prev.T
        T_prev_curr[:3, 3] = -R_curr_prev.T @ t_curr_prev

        inlier_ratio = n_inliers / max(n_matches, 1)
        feat_ratio = n_matches / max(
            min(prev_frame.descriptors.shape[0], curr_frame.descriptors.shape[0]),
            1,
        )
        score = float(np.clip(0.75 * inlier_ratio + 0.25 * feat_ratio, 0.0, 1.0))

        reproj_error = None
        try:
            obj_in = obj_pts[inlier_ids]
            img_in = img_pts[inlier_ids]
            proj, _ = cv2.projectPoints(obj_in, rvec, tvec, K, dist)
            proj = proj.reshape(-1, 2)
            err = np.linalg.norm(proj - img_in, axis=1)
            if err.size > 0:
                reproj_error = float(np.mean(err))
        except Exception:
            reproj_error = None

        return RGBDVoStep(
            T_prev_curr=T_prev_curr,
            score=score,
            reproj_error=reproj_error,
            n_matches=n_matches,
            n_inliers=n_inliers,
        )

    async def _run_rgbd_vo(
        self,
        pkt: TimeStampedPacket,
        frame_bgr: np.ndarray,
        depth_u16: np.ndarray | None,
        camera: dict | None,
    ) -> DetectionResult | None:
        K, dist, depth_scale_m = self._parse_camera_model(camera)
        if depth_u16 is None or K is None:
            return None

        if depth_u16.ndim != 2:
            return None

        curr_frame = await self.loop.run_in_executor(
            self._vo_executor,
            self._extract_rgbd_features_sync,
            frame_bgr,
            depth_u16,
            K,
            depth_scale_m,
        )

        state = self._vo_states.get(pkt.source)
        if state is None:
            T0 = self._T_world_camera.get(pkt.source)
            if T0 is None:
                T0 = np.eye(4, dtype=np.float64)
            state = RGBDVoState(pose_world_camera=np.array(T0, dtype=np.float64))
            self._vo_states[pkt.source] = state

        if curr_frame is None:
            state.last_frame = None
            return None

        if state.last_frame is None:
            state.last_frame = curr_frame
            return None

        step = await self.loop.run_in_executor(
            self._vo_executor,
            self._estimate_rgbd_vo_step_sync,
            state.last_frame,
            curr_frame,
            K,
            dist,
        )
        state.last_frame = curr_frame

        if step.T_prev_curr is None:
            return None

        state.pose_world_camera = state.pose_world_camera @ step.T_prev_curr

        return DetectionResult(
            t_host=pkt.t_host,
            source=f"{pkt.source}/rgbd_vo",
            pose_world=np.array(state.pose_world_camera, dtype=np.float64),
            score=float(step.score),
            reproj_error=step.reproj_error,
        )

    # ---------------- DOPE from RealSense ----------------
    async def _handle_packet(self, pkt: TimeStampedPacket) -> None:
        dope_det = None

        payload = pkt.payload
        if isinstance(payload, dict):
            stream = payload.get("stream")
        else:
            stream = None

        # RealSense packets carry frames in payload["frames"].
        if pkt.source.startswith("realsense"):
            try:
                rgbd_input = TrackingRGBDInput.from_packet(pkt)
            except (TypeError, ValueError, KeyError):
                return
            frame_bgr = rgbd_input.color_bgr
            depth_u16 = rgbd_input.depth_raw
            camera = payload.get("camera") if isinstance(payload, dict) else rgbd_input.legacy_camera()
            sensor_mode = self._sensor_mode(pkt.source)

            dope_task = None
            if self.use_realsense_for_dope and isinstance(self.model, DopeRunner):
                # print("source:", pkt.source)
                # print("camera:", camera)
                now = time.time()
                last = self._last_dope_ts.get(pkt.source, 0.0)
                if now - last >= self._dope_min_interval_s:
                    self._last_dope_ts[pkt.source] = now
                    dope_task = self.loop.run_in_executor(
                        self._dope_executor,
                        run_dope_sync,
                        self.model,
                        frame_bgr,
                        camera,
                    )

            apriltag_task = None
            if self.enable_realsense_apriltag and self._sensor_mode(pkt.source) != "fixed":
                apriltag_task = self.loop.run_in_executor(
                    self._apriltag_executor,
                    self._detect_aruco_and_draw,
                    frame_bgr,
                    camera,
                )

            # ── Run DOPE / VO / AprilTag in parallel via gather ──────────
            gather_tasks: dict[str, asyncio.Task | asyncio.coroutines] = {}
            if dope_task is not None:
                gather_tasks["dope"] = dope_task
            if self.enable_rgbd_vo:
                gather_tasks["vo"] = self._run_rgbd_vo(pkt, frame_bgr, depth_u16, camera)
            if apriltag_task is not None:
                gather_tasks["apriltag"] = apriltag_task

            dope_results: list = []

            if gather_tasks:
                completed = await asyncio.gather(*gather_tasks.values(), return_exceptions=True)
                results = dict(zip(gather_tasks.keys(), completed))

                # ── Process DOPE result (now a list of DopeDetection, one per object) ──
                dope_results: list = results.get("dope")
                if dope_results is None or isinstance(dope_results, BaseException):
                    dope_results = []
                elif not isinstance(dope_results, list):
                    dope_results = [dope_results]

                for dope_det in dope_results:
                    if dope_det is None or dope_det.pose_cam is None:
                        continue
                    if sensor_mode != "fixed":
                        slog(component="TrackingManager", event="mobile_sensor_pose_missing", source=pkt.source)
                        continue
                    T_world_cam = self._lookup_T_world_cam(pkt.source)
                    if T_world_cam is None:
                        slog(component="TrackingManager", event="missing_required_extrinsic", source=pkt.source, mode="fixed")
                        continue
                    T_cam_obj = np.array(dope_det.pose_cam, dtype=np.float64).reshape(4, 4)
                    T_cam_obj[:3, 3] *= 0.01
                    pose_world = T_world_cam @ T_cam_obj
                    reproj_error = self._reproj_error_px.get(pkt.source)
                    obj_name = getattr(dope_det, "object_name", "") or ""
                    det = DetectionResult(
                        t_host=pkt.t_host,
                        source=f"{pkt.source}/dope/{obj_name}" if obj_name else f"{pkt.source}/dope",
                        pose_world=pose_world,
                        score=float(dope_det.score),
                        reproj_error=reproj_error,
                        detector="dope",
                        object_name=obj_name or None,
                        sync_group=pkt.sync_group,
                    )
                    await self.bus.publish("tracking/detections", det)

                    if dope_det.vis_bgr is not None:
                        await self.bus.publish(
                            "tracking/dope_vis",
                            {
                                "t_host": pkt.t_host,
                                "source": pkt.source,
                                "object_name": obj_name,
                                "score": float(dope_det.score),
                                "image_bgr": dope_det.vis_bgr,
                            },
                        )

                # ── Process VO result ──
                vo_det = results.get("vo")
                if vo_det is not None and not isinstance(vo_det, BaseException):
                    await self.bus.publish("tracking/detections", vo_det)

                # ── Process AprilTag result ──
                apriltag_result = results.get("apriltag")
                if apriltag_result is not None and not isinstance(apriltag_result, BaseException):
                    vis_bgr, n, _tag_corners_by_id = apriltag_result
                    overlay_bgr = vis_bgr
                    if n > 0:
                        await self.bus.publish(
                            "tracking/aruco_vis",
                            {
                                "t_host": pkt.t_host,
                                "source": pkt.source,
                                "n": n,
                                "image_bgr": vis_bgr,
                            },
                        )
                        await self.bus.publish(
                            "tracking/apriltag_vis",
                            {
                                "t_host": pkt.t_host,
                                "source": pkt.source,
                                "n": n,
                                "image_bgr": vis_bgr,
                            },
                        )
                else:
                    overlay_bgr = frame_bgr.copy()
            else:
                dope_results = []
                overlay_bgr = frame_bgr.copy()

            if self._publish_vis:
                vis_all = np.array(overlay_bgr, copy=True)
                for dd in dope_results:
                    self._draw_dope_on_image(vis_all, dd)
                await self.bus.publish(
                    self._vis_topic,
                    {
                        "t_host": float(pkt.t_host),
                        "source": str(pkt.source),
                        "stream": "all_vis",
                        "image_bgr": vis_all,
                    },
                )

            # Compose an overlay image for preview.
            if self._overlay_preview_enabled:
                for dd in dope_results:
                    self._draw_dope_on_image(overlay_bgr, dd)
                self._overlay_preview_images[pkt.source] = overlay_bgr
            return

        # ---------------- DOPE + AprilTag from Quest ----------------
        if pkt.source.startswith("quest") and self.use_quest_for_dope:
            frame_bgr = None
            depth_u16 = None
            T_world_cam_per_frame: np.ndarray | None = None

            if isinstance(payload, dict):
                try:
                    packet_coordinate_system = normalize_coordinate_system(
                        payload.get(
                            "coordinate_system", payload.get("coord", "y_down")
                        )
                    )
                except ValueError as exc:
                    slog(
                        component="TrackingManager",
                        event="invalid_coordinate_system",
                        source=pkt.source,
                        reason=str(exc),
                    )
                    return
                if packet_coordinate_system != "y_down":
                    slog(
                        component="TrackingManager",
                        event="non_y_down_packet_rejected",
                        source=pkt.source,
                    )
                    return
                frames = payload.get("frames")
                if isinstance(frames, dict):
                    frame_bgr = frames.get("color")
                    depth_u16 = frames.get("depth")

                camera = payload.get("camera")
                tracking_value = payload.get("T_tracking_cam")
                if tracking_value is not None:
                    T_world_cam_per_frame = np.asarray(
                        tracking_value, dtype=np.float64
                    ).reshape(4, 4)
                embedded_extrinsic = payload.get("T_world_tracking")
                if self._T_world_quest is None and embedded_extrinsic is not None:
                    candidate = np.asarray(embedded_extrinsic, dtype=np.float64).reshape(4, 4)
                    if np.isfinite(candidate).all():
                        self._T_world_quest = candidate
                        slog(component="TrackingManager", event="quest_world_extrinsic_received", source=pkt.source)
            else:
                camera = None

            if frame_bgr is None:
                return
            if T_world_cam_per_frame is None or not np.isfinite(
                T_world_cam_per_frame
            ).all():
                slog(
                    component="TrackingManager",
                    event="missing_required_tracking_pose",
                    source=pkt.source,
                )
                return
            if self._T_world_quest is None:
                slog(
                    component="TrackingManager",
                    event="missing_required_extrinsic",
                    source=pkt.source,
                    mode="mobile",
                )
                return
            sensor_mode = self._sensor_mode(pkt.source)

            dope_task = None
            if isinstance(self.model, DopeRunner):
                now = time.time()
                last = self._last_dope_ts.get(pkt.source, 0.0)
                if now - last >= self._dope_min_interval_s:
                    self._last_dope_ts[pkt.source] = now
                    dope_task = self.loop.run_in_executor(
                        self._dope_executor,
                        run_dope_sync,
                        self.model,
                        frame_bgr,
                        camera,
                    )

            apriltag_task = None
            if self.enable_realsense_apriltag and sensor_mode == "mobile":
                apriltag_task = self.loop.run_in_executor(
                    self._apriltag_executor,
                    self._detect_aruco_and_draw,
                    frame_bgr,
                    camera,
                )

            tasks: dict[str, asyncio.Task | asyncio.coroutines] = {}
            if dope_task is not None:
                tasks["dope"] = dope_task
            if apriltag_task is not None:
                tasks["apriltag"] = apriltag_task

            dope_results: list = []
            n = 0
            tag_corners_by_id = None
            T_cam_cube = None
            overlay_bgr = frame_bgr.copy()

            if tasks:
                completed = await asyncio.gather(*tasks.values(), return_exceptions=True)
                results = dict(zip(tasks.keys(), completed))

                # ── AprilTag cube first: cube pose drives the cube-frame reference ──
                apriltag_result = results.get("apriltag")
                if apriltag_result is not None and not isinstance(apriltag_result, BaseException):
                    vis_bgr, n, tag_corners_by_id = apriltag_result
                    overlay_bgr = vis_bgr
                    if n > 0:
                        await self.bus.publish(
                            "tracking/aruco_vis",
                            {
                                "t_host": pkt.t_host,
                                "source": pkt.source,
                                "n": n,
                                "image_bgr": vis_bgr,
                            },
                        )
                        await self.bus.publish(
                            "tracking/apriltag_vis",
                            {
                                "t_host": pkt.t_host,
                                "source": pkt.source,
                                "n": n,
                                "image_bgr": vis_bgr,
                            },
                        )

                        # Estimate cube pose from AprilTags using the calibrated cube model
                        if (
                            tag_corners_by_id
                            and camera is not None
                            and self._cube_model is not None
                        ):
                            K_mat = np.array(camera.get("K"), dtype=np.float64).reshape(3, 3) if camera.get("K") is not None else None
                            if K_mat is not None:
                                dist_arr = np.array(camera.get("dist", np.zeros(5)), dtype=np.float64).reshape(-1)
                                T_cam_cube, reproj_err = self._estimate_cube_pose_with_model(
                                    tag_corners_by_id, K_mat, dist_arr, self._cube_model,
                                )
                                if T_cam_cube is not None and self._T_world_quest is not None:
                                    T_world_camera = (
                                        self._T_world_quest @ T_world_cam_per_frame
                                    )
                                    T_world_cube = T_world_camera @ T_cam_cube
                                    det = DetectionResult(
                                        t_host=pkt.t_host,
                                        source=f"{pkt.source}/apriltag_cube",
                                        pose_world=T_world_cube,
                                        score=float(max(0.0, 1.0 - reproj_err / 10.0)),
                                        reproj_error=float(reproj_err),
                                        sync_group=pkt.sync_group,
                                    )
                                    await self.bus.publish("tracking/detections", det)

                # ── DOPE results ──
                dope_results = results.get("dope")
                if dope_results is None or isinstance(dope_results, BaseException):
                    dope_results = []
                elif not isinstance(dope_results, list):
                    dope_results = [dope_results]

            # A mobile sensor requires world<-tracking. Tags refine that fixed
            # transform; per-frame SLAM supplies tracking<-camera.
            if T_cam_cube is not None and sensor_mode == "mobile":
                observed_world_tracking = (
                    np.linalg.inv(T_cam_cube)
                    @ np.linalg.inv(T_world_cam_per_frame)
                )
                self._correct_quest_world_extrinsic(observed_world_tracking, pkt.source)

            if sensor_mode == "fixed":
                T_world_camera = self._lookup_T_world_cam(pkt.source)
            else:
                T_world_camera = (
                    None
                    if self._T_world_quest is None
                    else self._T_world_quest @ T_world_cam_per_frame
                )
            if T_world_camera is not None:
                for dope_det in dope_results:
                    if dope_det is None or dope_det.pose_cam is None:
                        continue
                    T_cam_obj = np.array(dope_det.pose_cam, dtype=np.float64).reshape(4, 4)
                    T_cam_obj[:3, 3] *= 0.01  # DOPE outputs cm
                    pose_world = T_world_camera @ T_cam_obj
                    obj_name = getattr(dope_det, "object_name", "") or ""
                    det = DetectionResult(
                        t_host=pkt.t_host,
                        source=f"{pkt.source}/dope/{obj_name}" if obj_name else f"{pkt.source}/dope",
                        pose_world=pose_world,
                        score=float(dope_det.score),
                        detector="dope",
                        object_name=obj_name or None,
                        sync_group=pkt.sync_group,
                    )
                    await self.bus.publish("tracking/detections", det)

                    if dope_det.vis_bgr is not None and self._publish_dope_vis:
                        await self.bus.publish(
                            "tracking/dope_vis",
                            {
                                "t_host": pkt.t_host,
                                "source": pkt.source,
                                "object_name": obj_name,
                                "score": float(dope_det.score),
                                "image_bgr": dope_det.vis_bgr,
                            },
                        )
            elif dope_results:
                slog(component="TrackingManager", event="missing_required_extrinsic", source=pkt.source, mode="mobile")

            slog(
                component="TrackingManager",
                event="quest_frame_processed",
                source=str(pkt.source),
                t_host=float(pkt.t_host),
                n_apriltags=n if apriltag_task is not None else 0,
                dope_detected=(dope_det is not None and dope_det.pose_cam is not None),
            )

            if self._publish_vis:
                vis_all = np.array(overlay_bgr, copy=True)
                self._draw_dope_on_image(vis_all, dope_det)
                await self.bus.publish(
                    self._vis_topic,
                    {
                        "t_host": float(pkt.t_host),
                        "source": str(pkt.source),
                        "stream": "all_vis",
                        "image_bgr": vis_all,
                    },
                )

            if self._overlay_preview_enabled:
                self._draw_dope_on_image(overlay_bgr, dope_det)
                self._overlay_preview_images[pkt.source] = overlay_bgr
            return

        # ---------------- (optional) DOPE from HL2 PV ----------------
        # HL2 PV packets arrive on topic "hl2/pv" (source == "hl2", payload stream == "pv").
        if (
            self.use_hl2_pv_for_dope
            and pkt.source == "hl2"
            and stream == "pv"
            and isinstance(self.model, DopeRunner)
        ):
            frame_bgr = payload.get("image") if isinstance(payload, dict) else None
            if frame_bgr is None:
                return

            dope_det = await self.loop.run_in_executor(
                self._dope_executor,
                run_dope_sync,
                self.model,
                frame_bgr,
                None,
            )
            if dope_det.pose_cam is not None:
                det = DetectionResult(
                    t_host=pkt.t_host,
                    source="hl2/pv/dope",
                    pose_world=np.array(dope_det.pose_cam, dtype=np.float64),
                    score=float(dope_det.score),
                    detector="dope",
                    sync_group=pkt.sync_group,
                )
                await self.bus.publish("tracking/detections", det)
            return

        # ---------------- HL2 AHAT markers ----------------
        # HL2 AHAT marker extraction.
        if (
            self.enable_ahat_markers
            and pkt.source == "hl2"
            and stream == "ahat"
            and isinstance(payload, dict)
        ):
            points_world = payload.get("points_world")
            ab_image = payload.get("ab_image")
            if ab_image is None or points_world is None:
                return

            raw_ahat_markers_3d = []
            max_brightness = float(np.max(ab_image))
            if max_brightness > 0:
                thr = max_brightness / 2.0
                _, mask = cv2.threshold(ab_image, thr, 255, cv2.THRESH_BINARY)
                mask = mask.astype(np.uint8)

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                for c in contours[:3]:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        p3d = points_world[cy, cx]
                        if np.any(p3d):
                            raw_ahat_markers_3d.append(p3d)

            # Type-stub friendly: OpenCV normalizes into a provided dst array.
            ab_norm_u8 = np.empty(ab_image.shape, dtype=np.uint8)
            ab_norm = cv2.normalize(ab_image, ab_norm_u8, 0, 255, cv2.NORM_MINMAX)
            ahat_output = cv2.cvtColor(ab_norm, cv2.COLOR_GRAY2BGR)

            await self.bus.publish(
                "hl2_marker_vis",
                {
                    "t_host": pkt.t_host,
                    "source": "hl2/ahat",
                    "image_bgr": ahat_output,
                    "raw_markers_3d": raw_ahat_markers_3d,
                },
            )
            return

    def _load_static_extrinsics_from_config(self) -> None:
        path_str = self._calibration_result_path
        if not path_str:
            return

        p = Path(path_str)
        if not p.exists():
            slog(
                component=self.__class__.__name__,
                event="static_extrinsics_missing",
                path=str(p),
            )
            return

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            slog(
                component=self.__class__.__name__,
                event="static_extrinsics_parse_error",
                path=str(p),
                error=repr(exc),
            )
            return

        m = data.get("T_world_cam")
        if not isinstance(m, dict):
            return

        loaded = 0
        for serial_or_source, T in m.items():
            try:
                Tn = np.array(T, dtype=np.float64).reshape(4, 4)
            except Exception:
                continue

            key = str(serial_or_source)
            # Store under multiple candidate keys for flexible lookup
            candidate_keys = [key]
            if not key.startswith("realsense/") and not key.startswith("quest/"):
                candidate_keys.extend([f"realsense/{key}", f"quest/{key}"])
            for ck in candidate_keys:
                self._T_world_camera[ck] = Tn
            loaded += 1

        slog(
            component=self.__class__.__name__,
            event="static_extrinsics_loaded",
            path=str(p),
            n_loaded=int(loaded),
        )

    @staticmethod
    def _pose_from_position_quaternion(position: Any, quaternion_xyzw: Any) -> np.ndarray:
        p = np.asarray(position, dtype=np.float64).reshape(3)
        x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm([x, y, z, w]))
        if norm <= 1e-9:
            raise ValueError("zero quaternion")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])
        T[:3, 3] = p
        return T

    def _load_quest_world_extrinsic(self, path: str | None) -> np.ndarray | None:
        if not path or not Path(path).is_file():
            return None
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            coordinate_system = data.get(
                "coordinate_system", data.get("coord", "y_down")
            )
            value = data.get("T_world_tracking", data.get("T_world_quest"))
            if isinstance(value, dict):
                T = self._pose_from_position_quaternion(
                    value.get("pos", value.get("position")),
                    value.get("rot_xyzw", value.get("rotation_xyzw")),
                )
            else:
                T = np.asarray(value, dtype=np.float64).reshape(4, 4)
            if isinstance(value, dict):
                coordinate_system = value.get(
                    "coordinate_system", value.get("coord", coordinate_system)
                )
            T = transform_to_y_down(T, coordinate_system)
            if not np.isfinite(T).all():
                raise ValueError("non-finite transform")
        except Exception as exc:
            raise ValueError(f"Invalid Quest world extrinsic {path}: {exc}") from exc
        slog(component="TrackingManager", event="quest_world_extrinsic_loaded", path=str(path))
        return T

    def _correct_quest_world_extrinsic(self, observed: np.ndarray, source: str) -> None:
        if self._T_world_quest is None:
            slog(component="TrackingManager", event="quest_tag_correction_skipped_no_external_extrinsic", source=source)
            return
        delta = np.linalg.inv(self._T_world_quest) @ observed
        translation = float(np.linalg.norm(delta[:3, 3]))
        rvec, _ = cv2.Rodrigues(delta[:3, :3])
        rotation_deg = float(np.degrees(np.linalg.norm(rvec)))
        if translation > self._quest_tag_correction_max_translation_m or rotation_deg > self._quest_tag_correction_max_rotation_deg:
            slog(component="TrackingManager", event="quest_tag_correction_rejected", source=source, translation_m=translation, rotation_deg=rotation_deg)
            return
        alpha = self._quest_tag_correction_alpha
        correction = np.eye(4, dtype=np.float64)
        correction[:3, :3], _ = cv2.Rodrigues(rvec * alpha)
        correction[:3, 3] = delta[:3, 3] * alpha
        self._T_world_quest = self._T_world_quest @ correction
