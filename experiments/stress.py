from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StressConfig:
    mode: str = "normal"
    delay_s_by_camera: dict[str, float] = field(default_factory=dict)
    drop_prob_by_camera: dict[str, float] = field(default_factory=dict)
    noise_trans_m: float = 0.0
    noise_rot_deg: float = 0.0
    camera_start_time_s: dict[str, float] = field(default_factory=dict)


def should_activate_camera(camera_id: str, t_run_s: float, cfg: StressConfig) -> bool:
    t0 = float(cfg.camera_start_time_s.get(camera_id, 0.0))
    return t_run_s >= t0


def should_drop_observation(camera_id: str, cfg: StressConfig) -> bool:
    p = float(cfg.drop_prob_by_camera.get(camera_id, 0.0))
    if p <= 0.0:
        return False
    p = min(max(p, 0.0), 1.0)
    return random.random() < p


def delay_for_camera(camera_id: str, cfg: StressConfig) -> float:
    return float(max(cfg.delay_s_by_camera.get(camera_id, 0.0), 0.0))


def inject_pose_noise(T: np.ndarray, cfg: StressConfig) -> np.ndarray:
    """Inject translational and small rotational noise to SE(3)."""
    out = np.array(T, dtype=np.float64, copy=True)
    if out.shape != (4, 4):
        return out

    trans_std = float(max(cfg.noise_trans_m, 0.0))
    if trans_std > 0.0:
        out[:3, 3] += np.random.normal(0.0, trans_std, size=(3,))

    rot_std_deg = float(max(cfg.noise_rot_deg, 0.0))
    if rot_std_deg > 0.0:
        rot_std = np.deg2rad(rot_std_deg)
        w = np.random.normal(0.0, rot_std, size=(3,))
        theta = float(np.linalg.norm(w))
        if theta > 1e-12:
            k = w / theta
            K = np.array(
                [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
                dtype=np.float64,
            )
            Rn = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
            out[:3, :3] = Rn @ out[:3, :3]

    return out
