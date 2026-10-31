from dataclasses import dataclass

import numpy as np

from core.types import CameraModel, TimeStampedPacket, rgbd_frame_from_packet


@dataclass(frozen=True)
class TrackingRGBDInput:
    """Narrow input contract shared by DOPE, RGB-D VO and AprilTag workers."""

    color_bgr: np.ndarray
    depth_raw: np.ndarray
    camera: CameraModel

    @classmethod
    def from_packet(cls, packet: TimeStampedPacket) -> "TrackingRGBDInput":
        frame = rgbd_frame_from_packet(packet)
        return cls(frame.color_bgr, frame.depth_raw, frame.camera)

    def legacy_camera(self) -> dict[str, object]:
        return {"K": self.camera.K, "depth_scale_m": self.camera.depth_scale}
