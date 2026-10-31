from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import FrontendConfig

from .base import FrontendDetection, RGBDFrontend
from .filters import (
    backproject_masked_depth,
    largest_cluster,
    remove_outliers_statistical,
    transform_points,
    voxel_downsample,
)


def _safe_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if ys.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return (x1, y1, x2, y2)


@dataclass
class _RawSeg:
    mask: np.ndarray
    score: float
    class_name: str
    bbox: tuple[int, int, int, int] | None


class YOLOSegFrontend(RGBDFrontend):
    """YOLO-seg frontend for RGB-D object observation extraction."""

    def __init__(self, cfg: FrontendConfig):
        self.cfg = cfg
        self._target = {c.lower() for c in cfg.target_classes}

        # Preferred path: local wrapper in thirdparty/Ultralytics
        self._wrapper = None
        self._native_model = None

        try:
            from thirdparty.Ultralytics import YOLOWorldSeg

            self._wrapper = YOLOWorldSeg(
                model_path=cfg.model_path,
                device=cfg.device,
                conf_threshold=cfg.conf_threshold,
                iou_threshold=cfg.iou_threshold,
            )
        except Exception:
            self._wrapper = None

        if self._wrapper is None:
            try:
                from ultralytics import YOLO

                self._native_model = YOLO(cfg.model_path)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize YOLO segmentation frontend. "
                    "Install ultralytics and verify model path."
                ) from exc

    def _raw_infer(self, color_bgr: np.ndarray) -> list[_RawSeg]:
        out: list[_RawSeg] = []

        if self._wrapper is not None:
            res = self._wrapper.segment(color_bgr)
            for i in range(len(res)):
                mask = np.asarray(res.get_mask(i)).astype(np.uint8)
                cls_name = str(res.get_class_name(i))
                score = float(res.get_score(i))
                if self._target and cls_name.lower() not in self._target:
                    continue
                bbox = None
                try:
                    bbox = tuple(int(v) for v in res.boxes[i])
                except Exception:
                    bbox = _safe_bbox_from_mask(mask)
                out.append(
                    _RawSeg(mask=mask, score=score, class_name=cls_name, bbox=bbox)
                )
            return out

        # Native ultralytics fallback
        assert self._native_model is not None
        preds = self._native_model.predict(
            color_bgr,
            conf=float(self.cfg.conf_threshold),
            iou=float(self.cfg.iou_threshold),
            device=str(self.cfg.device),
            verbose=False,
        )
        if not preds:
            return out
        p = preds[0]
        if p.masks is None or p.boxes is None:
            return out

        names = getattr(p, "names", {}) or {}
        masks = p.masks.data
        boxes_xyxy = p.boxes.xyxy
        boxes_conf = p.boxes.conf
        boxes_cls = p.boxes.cls
        h, w = color_bgr.shape[:2]

        import cv2

        for m, b, c, cls_idx in zip(masks, boxes_xyxy, boxes_conf, boxes_cls):
            mask = m.detach().cpu().numpy()
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
            mask = (mask > 0.5).astype(np.uint8)
            score = float(c.detach().cpu().item())
            cls_id = int(cls_idx.detach().cpu().item())
            cls_name = str(names.get(cls_id, f"class_{cls_id}"))
            if self._target and cls_name.lower() not in self._target:
                continue
            box_np = b.detach().cpu().numpy().tolist()
            bbox = (int(box_np[0]), int(box_np[1]), int(box_np[2]), int(box_np[3]))
            out.append(_RawSeg(mask=mask, score=score, class_name=cls_name, bbox=bbox))

        return out

    def infer(
        self,
        *,
        camera_id: str,
        t_host: float,
        color_bgr: np.ndarray,
        depth_u16: np.ndarray | None,
        camera_meta: dict[str, Any] | None,
        T_world_cam: np.ndarray | None,
    ) -> list[FrontendDetection]:
        raw = self._raw_infer(color_bgr)
        out: list[FrontendDetection] = []

        if depth_u16 is None:
            # Return mask-only detections (for logging / debug)
            for r in raw:
                out.append(
                    FrontendDetection(
                        camera_id=camera_id,
                        t_host=t_host,
                        class_name=r.class_name,
                        score=r.score,
                        mask=r.mask,
                        bbox_xyxy=r.bbox,
                        quality={"num_points": 0.0},
                    )
                )
            return out

        for r in raw:
            pts_cam = backproject_masked_depth(
                mask=r.mask,
                depth_u16=depth_u16,
                camera_meta=camera_meta,
                min_depth_m=float(self.cfg.min_depth_m),
                max_depth_m=float(self.cfg.max_depth_m),
            )

            if pts_cam.size > 0:
                pts_cam = voxel_downsample(pts_cam, float(self.cfg.voxel_size_m))
                pts_cam = remove_outliers_statistical(
                    pts_cam,
                    min_neighbors=int(self.cfg.outlier_min_neighbors),
                    std_mul=float(self.cfg.outlier_std_mul),
                )
                if bool(self.cfg.largest_cluster):
                    pts_cam = largest_cluster(pts_cam, float(self.cfg.voxel_size_m))

            pts_world = transform_points(T_world_cam, pts_cam)

            out.append(
                FrontendDetection(
                    camera_id=camera_id,
                    t_host=t_host,
                    class_name=r.class_name,
                    score=float(r.score),
                    mask=r.mask,
                    bbox_xyxy=r.bbox,
                    points_cam=pts_cam,
                    points_world=pts_world,
                    quality={
                        "num_points": float(pts_world.shape[0]),
                        "mask_area": float(np.count_nonzero(r.mask)),
                    },
                )
            )

        return out
