from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    K: np.ndarray
    depth_scale: float = 0.001

    def __post_init__(self) -> None:
        matrix = np.asarray(self.K, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("CameraModel.K must be a finite 3x3 matrix")
        if not np.isfinite(self.depth_scale) or self.depth_scale <= 0:
            raise ValueError("CameraModel.depth_scale must be positive")
        object.__setattr__(self, "K", matrix)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CameraModel":
        scale = value.get("depth_scale_m", value.get("depth_scale", 0.001))
        return cls(np.asarray(value["K"], dtype=np.float64), float(scale))


@dataclass(frozen=True)
class RGBDFrame:
    color_bgr: np.ndarray
    depth_raw: np.ndarray
    camera: CameraModel
    source_id: str
    capture_time: float
    arrival_time: float
    frame_index: int | None = None
    sync_group: int | None = None
    device_time: float | None = None

    def __post_init__(self) -> None:
        color, depth = np.asarray(self.color_bgr), np.asarray(self.depth_raw)
        if color.ndim != 3 or color.shape[2] != 3:
            raise ValueError("RGBDFrame.color_bgr must have shape HxWx3")
        if depth.ndim != 2 or depth.shape != color.shape[:2]:
            raise ValueError("RGBDFrame depth and color resolutions must match")
        if not np.isfinite(self.capture_time) or not np.isfinite(self.arrival_time):
            raise ValueError("RGBDFrame timestamps must be finite")

    def legacy_payload(self) -> dict[str, Any]:
        return {"t_device": self.device_time, "camera": {"K": self.camera.K, "depth_scale": self.camera.depth_scale}, "frames": {"color": self.color_bgr, "depth": self.depth_raw}, "frame_index": self.frame_index, "sync_group": self.sync_group}


@dataclass
class TimeStampedPacket:
    t_host: float
    source: str
    payload: Any
    t_capture: float | None = None
    t_arrival: float | None = None
    frame_index: int | None = None
    sync_group: int | None = None

    def __post_init__(self) -> None:
        if self.t_capture is None:
            self.t_capture = self.t_host
        if self.t_arrival is None:
            self.t_arrival = self.t_host


def rgbd_frame_from_packet(packet: TimeStampedPacket) -> RGBDFrame:
    if isinstance(packet.payload, RGBDFrame):
        return packet.payload
    payload = packet.payload
    if not isinstance(payload, Mapping):
        raise TypeError("RGB-D packet payload must be RGBDFrame or a mapping")
    frames, camera_value = payload.get("frames", payload), payload.get("camera")
    if not isinstance(frames, Mapping) or not isinstance(camera_value, Mapping):
        raise ValueError("Legacy RGB-D payload requires 'frames' and 'camera' mappings")
    return RGBDFrame(color_bgr=np.asarray(frames["color"]), depth_raw=np.asarray(frames["depth"]), camera=CameraModel.from_mapping(camera_value), source_id=packet.source, capture_time=float(packet.t_capture), arrival_time=float(packet.t_arrival), frame_index=packet.frame_index if packet.frame_index is not None else payload.get("frame_index"), sync_group=packet.sync_group if packet.sync_group is not None else payload.get("sync_group"), device_time=payload.get("t_device"))


@dataclass
class DetectionResult:
    t_host: float
    source: str
    pose_world: np.ndarray
    score: float
    reproj_error: float | None = None
    detector: str | None = None
    object_name: str | None = None
    instance_id: str | None = None
    sync_group: int | None = None
    covariance: np.ndarray | None = None
    # Optional fields used by the full factor-graph path.  They are additive
    # so existing record/replay files and producers remain valid.
    pose_sensor: np.ndarray | None = None
    sensor_type: str | None = None
    known_extrinsic: np.ndarray | None = None
    capture_time: float | None = None
    quality_metrics: Mapping[str, Any] | None = None
    imu_delta_pose: np.ndarray | None = None
    imu_delta_covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose_world, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("DetectionResult.pose_world must be a finite 4x4 matrix")
        self.pose_world = pose
        if self.covariance is not None:
            covariance = np.asarray(self.covariance, dtype=np.float64)
            if covariance.shape != (6, 6) or not np.isfinite(covariance).all():
                raise ValueError("DetectionResult.covariance must be a finite 6x6 matrix")
            self.covariance = covariance
        for name in ("pose_sensor", "known_extrinsic", "imu_delta_pose"):
            value = getattr(self, name)
            if value is None:
                continue
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(f"DetectionResult.{name} must be a finite 4x4 matrix")
            setattr(self, name, matrix)
        if self.imu_delta_covariance is not None:
            imu_covariance = np.asarray(self.imu_delta_covariance, dtype=np.float64)
            if imu_covariance.shape != (6, 6) or not np.isfinite(imu_covariance).all():
                raise ValueError(
                    "DetectionResult.imu_delta_covariance must be a finite 6x6 matrix"
                )
            self.imu_delta_covariance = imu_covariance
        if self.capture_time is None:
            self.capture_time = float(self.t_host)


@dataclass
class FusedState:
    t_host: float
    pose_world: np.ndarray
    covariance: Optional[np.ndarray] = None
    confidence: float = 1.0
