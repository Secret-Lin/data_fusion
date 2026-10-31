from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Any

import numpy as np


@dataclass
class FrontendDetection:
    camera_id: str
    t_host: float
    class_name: str
    score: float
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int] | None = None
    points_cam: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    points_world: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    quality: dict[str, float] = field(default_factory=dict)


class RGBDFrontend(Protocol):
    """Swappable frontend interface for RGB-D object detection."""

    def infer(
        self,
        *,
        camera_id: str,
        t_host: float,
        color_bgr: np.ndarray,
        depth_u16: np.ndarray | None,
        camera_meta: dict[str, Any] | None,
        T_world_cam: np.ndarray | None,
    ) -> list[FrontendDetection]: ...
