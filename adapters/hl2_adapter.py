# mr_nav_backend/adapters/hl2_adapter.py

import asyncio
import threading
import time
from typing import Optional, Dict, Any
import os
import sys

import numpy as np
import cv2

from common.structured_log import slog

# ---- Ensure HL2SS python modules are importable (viewer/) ----
_THIS_DIR = os.path.dirname(__file__)
_HL2SS_VIEWER_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "thirdparty", "hl2ss", "viewer"))
if _HL2SS_VIEWER_DIR not in sys.path:
    sys.path.insert(0, _HL2SS_VIEWER_DIR)

import hl2ss
import hl2ss_lnm
import hl2ss_3dcv

from core.types import TimeStampedPacket
from adapters import BaseBlockingAdapter


class HL2MultiSensorAdapter(BaseBlockingAdapter):
    """
    Streams from HoloLens 2:
      - RM Depth AHAT (depth + ab + pose)
      - PV (personal video) (image + pose)
      - IMU (accelerometer/gyroscope/magnetometer packets)

    Implementation choice:
      - Use hl2ss_lnm clients directly (NO hl2ss_mp), one thread per stream.
      - Push packets into out_queue with a 'stream' field so downstream can route.
    """

    def __init__(
        self,
        loop,
        out_queue,
        host: str,
        calibration_path: str,
        enable_ahat: bool = True,
        enable_pv: bool = True,
        enable_imu: bool = True,
        source_name: str = "hl2",
        pv_width: int = 1280,
        pv_height: int = 720,
        pv_fps: int = 30,
        pv_profile: int = hl2ss.VideoProfile.H264_BASE,  # adjust if needed
    ):
        super().__init__(loop, out_queue, source_name)

        self.host = host
        self.calibration_path = calibration_path

        self.enable_ahat = enable_ahat
        self.enable_pv = enable_pv
        self.enable_imu = enable_imu

        # AHAT calibration cache
        self.calibration_ahat = None
        self.xy1 = None
        self.scale = None

        # PV params
        self.pv_width = pv_width
        self.pv_height = pv_height
        self.pv_fps = pv_fps
        self.pv_profile = pv_profile

        # lnm clients
        self.client_ahat = None
        self.client_pv = None

        # IMU clients (HL2SS may expose different function names depending on version)
        self.client_imu_accel = None
        self.client_imu_gyro = None
        self.client_imu_mag = None

        # stream worker threads
        self._threads = []

        self._running = False

        self._init_clients_and_calib()

        slog(
            component=self.__class__.__name__,
            event="init",
            source=self.source_name,
            host=self.host,
            calibration_path=self.calibration_path,
            enable_ahat=self.enable_ahat,
            enable_pv=self.enable_pv,
            enable_imu=self.enable_imu,
        )

    # ----------------------------
    # Init
    # ----------------------------
    def _init_clients_and_calib(self):
        if self.enable_ahat:
            print("Getting AHAT calibration...")
            self.calibration_ahat = hl2ss_3dcv.get_calibration_rm(
                self.host, hl2ss.StreamPort.RM_DEPTH_AHAT, self.calibration_path
            )
            uv2xy = self.calibration_ahat.uv2xy
            self.xy1, self.scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, self.calibration_ahat.scale)
            print("AHAT Calibration OK.")

            self.client_ahat = hl2ss_lnm.rx_rm_depth_ahat(self.host, hl2ss.StreamPort.RM_DEPTH_AHAT)
            self.client_ahat.open()
            print("AHAT client opened.")

        if self.enable_pv:
            self.client_pv = hl2ss_lnm.rx_pv(
                self.host, hl2ss.StreamPort.PERSONAL_VIDEO,
                width=self.pv_width, height=self.pv_height, framerate=self.pv_fps,
                decoded_format="bgr24"
            )
            self.client_pv.open()
            print("PV client opened.")

        if self.enable_imu:
            # HL2SS naming differs by version. We try multiple possibilities.
            # Typical ports:
            #   RM_IMU_ACCELEROMETER, RM_IMU_GYROSCOPE, RM_IMU_MAGNETOMETER
            # Typical constructors (varies): rx_rm_imu_accelerometer, rx_rm_imu_gyroscope, rx_rm_imu_magnetometer
            accel_ctor = getattr(hl2ss_lnm, "rx_rm_imu_accelerometer", None)
            gyro_ctor  = getattr(hl2ss_lnm, "rx_rm_imu_gyroscope", None)
            mag_ctor   = getattr(hl2ss_lnm, "rx_rm_imu_magnetometer", None)

            if accel_ctor is None or gyro_ctor is None or mag_ctor is None:
                # Some versions expose a generic rx_rm_imu(port) or similar; try fallbacks.
                generic_ctor = getattr(hl2ss_lnm, "rx_rm_imu", None)
                if generic_ctor is None:
                    raise ImportError(
                        "IMU enabled but cannot find IMU constructors in hl2ss_lnm "
                        "(tried rx_rm_imu_accelerometer/gyroscope/magnetometer and rx_rm_imu)."
                    )
                self.client_imu_accel = generic_ctor(self.host, hl2ss.StreamPort.RM_IMU_ACCELEROMETER)
                self.client_imu_gyro  = generic_ctor(self.host, hl2ss.StreamPort.RM_IMU_GYROSCOPE)
                self.client_imu_mag   = generic_ctor(self.host, hl2ss.StreamPort.RM_IMU_MAGNETOMETER)
            else:
                self.client_imu_accel = accel_ctor(self.host, hl2ss.StreamPort.RM_IMU_ACCELEROMETER)
                self.client_imu_gyro  = gyro_ctor(self.host, hl2ss.StreamPort.RM_IMU_GYROSCOPE)
                self.client_imu_mag   = mag_ctor(self.host, hl2ss.StreamPort.RM_IMU_MAGNETOMETER)

            self.client_imu_accel.open()
            self.client_imu_gyro.open()
            self.client_imu_mag.open()
            print("IMU clients opened (accel/gyro/mag).")

    # ----------------------------
    # BaseBlockingAdapter override:
    # Instead of one _read_from_sdk_blocking, we start multiple stream threads.
    # ----------------------------
    def start(self):
        """Start background stream threads."""
        slog(
            component=self.__class__.__name__,
            event="start",
            source=self.source_name,
            enable_ahat=self.enable_ahat,
            enable_pv=self.enable_pv,
            enable_imu=self.enable_imu,
        )
        super().start()
        self._running = True

        if self.enable_ahat and self.client_ahat is not None:
            t = threading.Thread(target=self._worker_ahat, daemon=True)
            t.start()
            self._threads.append(t)

        if self.enable_pv and self.client_pv is not None:
            t = threading.Thread(target=self._worker_pv, daemon=True)
            t.start()
            self._threads.append(t)

        if self.enable_imu and self.client_imu_accel is not None:
            t1 = threading.Thread(target=self._worker_imu, args=("imu_accel", self.client_imu_accel), daemon=True)
            t2 = threading.Thread(target=self._worker_imu, args=("imu_gyro", self.client_imu_gyro), daemon=True)
            t3 = threading.Thread(target=self._worker_imu, args=("imu_mag", self.client_imu_mag), daemon=True)
            t1.start(); t2.start(); t3.start()
            self._threads.extend([t1, t2, t3])

        print(f"[HL2MultiSensorAdapter] Started threads: {len(self._threads)}")

    # ----------------------------
    # Stream workers
    # ----------------------------
    def _emit(self, stream: str, payload: Dict[str, Any], t_host: Optional[float] = None):
        """
        TimeStampedPacket:
        t_host: float (PC monotonic time)
        source: str
        payload: Any
        """
        if t_host is None:
            t_host = time.monotonic()

        pkt = TimeStampedPacket(
            t_host=t_host,
            source=self.source_name,
            payload={"stream": stream, **payload},
        )

        self.loop.call_soon_threadsafe(self.out_queue.put_nowait, pkt)

    def _worker_ahat(self):
        while self._running:
            try:
                data_ahat = self.client_ahat.get_next_packet()
                if data_ahat is None:
                    time.sleep(0.001)
                    continue

                # depth normalize + points
                depth = hl2ss_3dcv.rm_depth_normalize(data_ahat.payload.depth, self.scale)
                points_cam = hl2ss_3dcv.rm_depth_to_points(depth, self.xy1)

                depth_to_world = (
                    hl2ss_3dcv.camera_to_rignode(self.calibration_ahat.extrinsics)
                    @ hl2ss_3dcv.reference_to_world(data_ahat.pose)
                )
                points_world = hl2ss_3dcv.transform(points_cam, depth_to_world)

                ab_image = data_ahat.payload.ab

                self._emit(
                    "ahat",
                    {
                        "depth": depth,
                        "ab_image": ab_image,
                        "points_world": points_world,
                        "pose": data_ahat.pose,  # raw pose
                    },
                )
            except Exception as e:
                # avoid hard-crash on transient network errors
                self._emit("ahat_error", {"error": repr(e)})
                time.sleep(0.01)

    def _worker_pv(self):
        while self._running:
            try:
                data_pv = self.client_pv.get_next_packet()
                if data_pv is None:
                    time.sleep(0.001)
                    continue

                img = getattr(data_pv.payload, "image", None)  # bgr24 decoded
                self._emit(
                    "pv",
                    {
                        "image": img,
                        "intrinsics": getattr(data_pv.payload, "intrinsics", None),
                        "pose": getattr(data_pv, "pose", None),
                    },
                )
            except Exception as e:
                self._emit("pv_error", {"error": repr(e)})
                time.sleep(0.01)

    def _worker_imu(self, stream_name: str, client):
        while self._running:
            try:
                data_imu = client.get_next_packet()
                if data_imu is None:
                    time.sleep(0.001)
                    continue

                # IMU packet structure differs; pass-through safely
                self._emit(
                    stream_name,
                    {
                        "imu_payload": data_imu.payload,
                        "pose": getattr(data_imu, "pose", None),
                    },
                )
            except Exception as e:
                self._emit(f"{stream_name}_error", {"error": repr(e)})
                time.sleep(0.01)

    def stop(self):
        slog(
            component=self.__class__.__name__,
            event="stop",
            source=self.source_name,
        )
        self._running = False
        super().stop()

        # Close clients
        try:
            if self.client_ahat is not None:
                self.client_ahat.close()
        except Exception:
            pass
        try:
            if self.client_pv is not None:
                self.client_pv.close()
        except Exception:
            pass
        try:
            if self.client_imu_accel is not None:
                self.client_imu_accel.close()
            if self.client_imu_gyro is not None:
                self.client_imu_gyro.close()
            if self.client_imu_mag is not None:
                self.client_imu_mag.close()
        except Exception:
            pass

        print("[HL2MultiSensorAdapter] Stopped.")
