from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.types import DetectionResult
from experiments.frontends.base import FrontendDetection


@dataclass
class SegPointCloudObservation:
    camera_id: str
    t_host: float
    points_world: np.ndarray
    confidence: float
    quality: dict[str, float] = field(default_factory=dict)
    covariance: np.ndarray | None = None


def point_cloud_to_pose_world(points_world: np.ndarray) -> np.ndarray:
    """Simple SE(3) from point-cloud centroid with identity rotation."""
    T = np.eye(4, dtype=np.float64)
    if points_world.size == 0:
        return T
    T[:3, 3] = np.mean(points_world, axis=0)
    return T


def frontend_detection_to_observation(
    det: FrontendDetection,
) -> SegPointCloudObservation:
    q = dict(det.quality)
    q.setdefault("score", float(det.score))
    q.setdefault("num_points", float(det.points_world.shape[0]))
    return SegPointCloudObservation(
        camera_id=str(det.camera_id),
        t_host=float(det.t_host),
        points_world=np.array(det.points_world, dtype=np.float64),
        confidence=float(det.score),
        quality=q,
        covariance=None,
    )


def observation_to_detection_result(
    obs: SegPointCloudObservation,
    *,
    source_suffix: str = "seg3d",
) -> DetectionResult | None:
    if obs.points_world.size == 0:
        return None
    T = point_cloud_to_pose_world(obs.points_world)
    if not np.isfinite(T).all():
        return None

    score = float(obs.confidence)
    # Quality proxy: higher point count -> higher confidence cap.
    n_pts = float(obs.quality.get("num_points", obs.points_world.shape[0]))
    score = float(np.clip(score * min(1.0, n_pts / 300.0 + 0.2), 0.0, 1.0))

    return DetectionResult(
        t_host=float(obs.t_host),
        source=f"{obs.camera_id}/{source_suffix}",
        pose_world=T,
        score=score,
        reproj_error=None,
    )


def observation_to_fusion_dict(obs: SegPointCloudObservation) -> dict[str, Any] | None:
    """Optional dict-format observation for fusion placeholders."""
    if obs.points_world.size == 0:
        return None
    T = point_cloud_to_pose_world(obs.points_world)
    if not np.isfinite(T).all():
        return None

    return {
        "source": f"{obs.camera_id}/seg3d",
        "t_host": float(obs.t_host),
        "T": T.tolist(),
        "quality_metrics": {
            "score": float(obs.confidence),
            "num_points": float(
                obs.quality.get("num_points", obs.points_world.shape[0])
            ),
            "q_time": 1.0,
        },
        "covariance": None if obs.covariance is None else obs.covariance.tolist(),
    }
