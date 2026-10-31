from __future__ import annotations

from config import FrontendConfig

from .base import FrontendDetection, RGBDFrontend
from .yolo_seg_frontend import YOLOSegFrontend


def build_frontend(cfg: FrontendConfig) -> RGBDFrontend:
    kind = str(cfg.detector or "").strip().lower()
    if kind in {"yolo", "yolo_seg", "yolo-seg", "ultralytics"}:
        return YOLOSegFrontend(cfg)
    if kind in {"dope", "dope_frontend"}:
        raise NotImplementedError(
            "Optional DOPE-based frontend extension point is reserved. "
            "Implement a DOPE frontend adapter matching RGBDFrontend interface."
        )
    raise ValueError(f"Unknown frontend detector '{cfg.detector}'")


__all__ = [
    "FrontendDetection",
    "RGBDFrontend",
    "YOLOSegFrontend",
    "build_frontend",
]
