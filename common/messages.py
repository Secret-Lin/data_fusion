# common/messages.py
from dataclasses import dataclass
import numpy as np
from typing import Dict, Optional

@dataclass
class PoseObservation:
    sensor_name: str                     # e.g., "hl2_ahat", "optitrack_rig", "ext_cam_rgbd"
    t_world_sensor: np.ndarray           # (4, 4) homogeneous transform
    timestamp: float                     # in seconds
    covariance: Optional[np.ndarray]     # (6, 6) or None if unknown
    quality: Dict[str, float]           # e.g. {"num_features": 120, "rms_reproj": 0.8, "latency_ms": 15.3}
