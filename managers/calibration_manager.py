# managers/calibration_manager.py

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from common.structured_log import slog

import cv2
import numpy as np

from core.event_bus import EventBus
from core.coordinate_frames import transform_to_y_down

from config import BackendConfig


class CalibrationManager:
    """负责相机 / HL2 / OTS 的标定和坐标系对齐。
    
    - 监听来自 HMD/Unity 的“开始标定”命令；
    - 利用传感器数据（带棋盘/AprilTag板）估计 intrinsics & extrinsics；
    - 将结果写入全局配置 / 发布到 EventBus。
    """

    def __init__(self, bus: EventBus, cfg: BackendConfig) -> None:
        self.bus = bus
        self.cfg = cfg
        self._calib_params: Dict[str, dict] = {}  # key: source_name

        # Extrinsics stay empty until a real calibration is available.
        self._T_world_camera: Dict[str, list[list[float]]] = {}

        # Cube definition (world frame): center at origin.
        self._cube_edge_m = 0.05
        self._tag_size_m = 0.05

        # Tag -> face mapping (opposites): (2,4), (5,1), (0,3)
        # World axes: +X->2, +Y->5, +Z->0
        self._T_world_tag = self._build_world_tag_transforms(self._cube_edge_m)

        if not hasattr(cv2, "aruco"):
            raise ImportError(
                "OpenCV ArUco module not found (cv2.aruco). Install opencv-contrib-python in your environment."
            )
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
        self._aruco_params = cv2.aruco.DetectorParameters()
        try:
            self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        except Exception:
            self._aruco_detector = None

        self._executor = ThreadPoolExecutor(max_workers=1)

        # Subscribe to RealSense streams (single + multi-camera topics)
        self._rs_topics = self._resolve_realsense_topics()
        self._rs_queues = [bus.subscribe(t) for t in self._rs_topics]

        # 示例：订阅某个 topic 来触发标定
        self._cmd_queue = bus.subscribe("calib_commands")

    async def run(self) -> None:
        slog(
            component=self.__class__.__name__,
            event="run",
            topic_commands="calib_commands",
            topic_result="calib_result",
            topic_extrinsics="calibration/extrinsics",
            realsense_topics=self._rs_topics,
        )

        while True:
            get_tasks = [asyncio.create_task(self._cmd_queue.get())]
            get_tasks.extend([asyncio.create_task(q.get()) for q in self._rs_queues])
            try:
                done, _pending = await asyncio.wait(get_tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for t in get_tasks:
                    if not t.done():
                        t.cancel()

            for t in done:
                msg = t.result()
                if msg is None:
                    continue

                if isinstance(msg, dict):
                    # cmd 示例：{"action": "start", "sensor": "cam0"}
                    if msg.get("action") == "start" and "sensor" in msg:
                        await self._run_calibration_for_sensor(str(msg["sensor"]))
                    continue

                # Otherwise assume it's a TimeStampedPacket from a camera stream
                await self._on_realsense_packet(msg)

    def _resolve_realsense_topics(self) -> list[str]:
        topics: list[str] = []
        cams = list(getattr(self.cfg, "realsense_cameras", []) or [])
        if not cams:
            topics = ["realsense_stream"]
        else:
            for cam in cams:
                cam_id = cam.serial or cam.name or "realsense"
                topics.append(f"realsense/{cam_id}/stream")
        # Always include legacy alias if present.
        if "realsense_stream" not in topics:
            topics.append("realsense_stream")
        return topics

    @staticmethod
    def _build_world_tag_transforms(L: float) -> dict[int, np.ndarray]:
        """Return T_world_tag for ids {0,1,2,3,4,5}.

        World frame:
        - origin at cube center
        - +X points to face id=2, -X to id=4
        - +Y points to face id=5, -Y to id=1
        - +Z points to face id=0, -Z to id=3

        Tag axis convention:
        - x_tag: to the right on the printed tag
        - y_tag: to the top on the printed tag
        - z_tag: out of the tag plane

        Given on face id=0 (+Z): right -> face 4 (-X), up -> face 1 (-Y)
        => x_tag=-X, y_tag=-Y, z_tag=+Z.
        """

        def T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
            out = np.eye(4, dtype=np.float64)
            out[:3, :3] = R
            out[:3, 3] = t
            return out

        ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        half = 0.5 * float(L)

        # R_world_tag columns are x_tag, y_tag, z_tag expressed in world.
        R0 = np.stack([-ex, -ey, ez], axis=1)  # id0: +Z face
        R3 = np.stack([-ex, ey, -ez], axis=1)  # id3: -Z face
        R2 = np.stack([-ey, -ez, ex], axis=1)  # id2: +X face
        R4 = np.stack([-ey, ez, -ex], axis=1)  # id4: -X face
        R5 = np.stack([-ez, -ex, ey], axis=1)  # id5: +Y face
        R1 = np.stack([-ez, ex, -ey], axis=1)  # id1: -Y face

        y_up_transforms = {
            0: T(R0, np.array([0.0, 0.0, half], dtype=np.float64)),
            3: T(R3, np.array([0.0, 0.0, -half], dtype=np.float64)),
            2: T(R2, np.array([half, 0.0, 0.0], dtype=np.float64)),
            4: T(R4, np.array([-half, 0.0, 0.0], dtype=np.float64)),
            5: T(R5, np.array([0.0, half, 0.0], dtype=np.float64)),
            1: T(R1, np.array([0.0, -half, 0.0], dtype=np.float64)),
        }
        return {
            tag_id: transform_to_y_down(transform, "y_up")
            for tag_id, transform in y_up_transforms.items()
        }

    def _estimate_T_world_cam_from_frame(self, frame_bgr: np.ndarray, camera: dict | None):
        if frame_bgr is None or not isinstance(camera, dict) or camera.get("K") is None:
            return None

        K = np.array(camera["K"], dtype=np.float64).reshape(3, 3)
        dist = camera.get("dist")
        if dist is None:
            dist = np.zeros((5,), dtype=np.float64)
        else:
            dist = np.array(dist, dtype=np.float64).reshape(-1)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._aruco_detector is not None:
            corners, ids, _rej = self._aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _rej = cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._aruco_params)

        if ids is None or len(ids) == 0:
            return None

        s = float(self._tag_size_m)
        half = s * 0.5
        objp = np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
            ],
            dtype=np.float64,
        )

        best = None
        best_err = 1e9
        best_id = None

        pnp_flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", None)
        if pnp_flag is None:
            pnp_flag = cv2.SOLVEPNP_ITERATIVE

        for c, id_arr in zip(corners, ids, strict=False):
            tag_id = int(id_arr[0])
            if tag_id not in self._T_world_tag:
                continue

            imgp = np.array(c, dtype=np.float64).reshape(4, 2)
            ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=pnp_flag)
            if not ok:
                continue

            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            proj = proj.reshape(4, 2)
            err = float(np.mean(np.linalg.norm(proj - imgp, axis=1)))
            if err < best_err:
                best_err = err
                best = (rvec, tvec)
                best_id = tag_id

        if best is None or best_id is None:
            return None

        rvec, tvec = best
        R_cam_tag, _ = cv2.Rodrigues(rvec)
        T_cam_tag = np.eye(4, dtype=np.float64)
        T_cam_tag[:3, :3] = R_cam_tag
        T_cam_tag[:3, 3] = np.array(tvec, dtype=np.float64).reshape(3)

        T_world_tag = self._T_world_tag[best_id]
        T_world_cam = T_world_tag @ np.linalg.inv(T_cam_tag)

        return {
            "T_world_cam": T_world_cam,
            "tag_id": best_id,
            "reproj_error": best_err,
        }

    async def _on_realsense_packet(self, pkt) -> None:
        payload = getattr(pkt, "payload", None)
        if not isinstance(payload, dict):
            return
        frames = payload.get("frames")
        if not isinstance(frames, dict):
            return
        frame_bgr = frames.get("color")
        camera = payload.get("camera")
        if frame_bgr is None:
            return

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._executor, self._estimate_T_world_cam_from_frame, frame_bgr, camera)
        if result is None:
            return

        source = str(getattr(pkt, "source", "unknown"))
        T_world_cam = result["T_world_cam"]

        await self.bus.publish(
            "calibration/extrinsics",
            {
                "t_host": float(getattr(pkt, "t_host", time.time())),
                "T_world_camera": {source: T_world_cam.tolist()},
                "method": "aruco_cube",
                "coordinate_system": "y_down",
                "tag_id": int(result["tag_id"]),
                "reproj_error": float(result["reproj_error"]),
            },
        )

    async def _run_calibration_for_sensor(self, sensor_name: str) -> None:
        # TODO: 从对应 sensor 的数据流中抓带棋盘的图像，调用 OpenCV 标定
        # self._calib_params[sensor_name] = {...}
        # 标定完成后，可以通过 EventBus 广播结果
        await self.bus.publish("calib_result", {
            "sensor": sensor_name,
            "params": self._calib_params.get(sensor_name, {})
        })
