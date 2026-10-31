import math
import time

import numpy as np

from common.structured_log import slog

from config import BackendConfig
from core.event_bus import EventBus
from core.types import DetectionResult


def _rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion [w, x, y, z]."""
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12)) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12)) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12)) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in q]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _avg_quat_markley(quats: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted quaternion averaging (Markley)."""
    q_ref = quats[0]
    A = np.zeros((4, 4), dtype=np.float64)
    for q, w in zip(quats, weights):
        q_use = q.copy()
        if float(np.dot(q_use, q_ref)) < 0.0:
            q_use = -q_use
        A += float(w) * np.outer(q_use, q_use)

    eigvals, eigvecs = np.linalg.eigh(A)
    q = eigvecs[:, int(np.argmax(eigvals))]
    if q[0] < 0.0:
        q = -q
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


class BaselineManager:
    """
    Runs two online baselines from tracking detections:
      1) best-camera baseline
      2) weighted multi-camera average baseline
    """

    def __init__(self, bus: EventBus, cfg: BackendConfig):
        self.bus = bus
        self.cfg = cfg
        self.baseline_cfg = cfg.baseline
        self._q_tracking = bus.subscribe("tracking/detections")

        # Sensor key -> latest DetectionResult
        self._latest: dict[str, DetectionResult] = {}

        self._perf_interval = 20
        self._perf_count = 0
        self._perf_compute_ms_sum = 0.0
        self._perf_end_to_end_ms_sum = 0.0
        self._perf_sched_lag_ms_sum = 0.0
        self._perf_t0_wall: float | None = None
        self._perf_t0_host: float | None = None

    @staticmethod
    def _sensor_key(source: str) -> str:
        parts = str(source).split("/")
        if len(parts) <= 1:
            return str(source)
        # Remove the trailing algorithm name (e.g. /dope, /rgbd_vo)
        return "/".join(parts[:-1])

    def _single_camera_key(self) -> str | None:
        key = str(getattr(self.baseline_cfg, "single_camera_source", "") or "").strip()
        return key or None

    def _is_eligible(self, det: DetectionResult) -> bool:
        if self.baseline_cfg.dope_only and not str(det.source).endswith("/dope"):
            return False
        if float(det.score) < float(self.baseline_cfg.min_score):
            return False
        if det.reproj_error is not None:
            rr = float(det.reproj_error)
            if (not np.isfinite(rr)) or rr > float(
                self.baseline_cfg.max_reproj_error_px
            ):
                return False
        T = np.array(det.pose_world, dtype=np.float64)
        if T.shape != (4, 4) or (not np.isfinite(T).all()):
            return False
        return True

    def _quality_weight(self, det: DetectionResult) -> float:
        w = max(float(det.score), 1e-6)
        rr = det.reproj_error
        if rr is not None and np.isfinite(float(rr)):
            sigma = max(float(self.baseline_cfg.reproj_sigma_px), 1e-6)
            w *= math.exp(-max(float(rr), 0.0) / sigma)
        return max(w, 1e-6)

    def _current_valid(self, t_now: float) -> list[DetectionResult]:
        max_age = float(self.baseline_cfg.max_age_s)
        out: list[DetectionResult] = []
        stale_keys: list[str] = []

        for key, det in self._latest.items():
            if t_now - float(det.t_host) > max_age:
                stale_keys.append(key)
                continue
            if self._is_eligible(det):
                out.append(det)

        for key in stale_keys:
            self._latest.pop(key, None)

        return out

    async def _publish_best(self, valid: list[DetectionResult], t_now: float) -> None:
        best = max(valid, key=self._quality_weight)
        T = np.array(best.pose_world, dtype=np.float64)
        msg = {
            "T": T.tolist(),
            "t_host": float(t_now),
            "stamp": time.time(),
            "source": "baseline/best",
            "picked_source": str(best.source),
            "score": float(best.score),
            "reproj_error": (
                None if best.reproj_error is None else float(best.reproj_error)
            ),
            "n_used": int(len(valid)),
        }
        await self.bus.publish(self.baseline_cfg.topic_best, msg)

    async def _publish_single(self, valid: list[DetectionResult], t_now: float) -> None:
        key = self._single_camera_key()
        picked: DetectionResult | None = None
        if key is not None:
            picked = self._latest.get(key)
            if picked is not None and not self._is_eligible(picked):
                picked = None

        if picked is None:
            # Fallback: choose best from current valid set.
            picked = max(valid, key=self._quality_weight)

        T = np.array(picked.pose_world, dtype=np.float64)
        msg = {
            "T": T.tolist(),
            "t_host": float(t_now),
            "stamp": time.time(),
            "source": "baseline/single",
            "picked_source": str(picked.source),
            "score": float(picked.score),
            "reproj_error": (
                None if picked.reproj_error is None else float(picked.reproj_error)
            ),
            "n_used": int(len(valid)),
            "single_camera_key": key,
        }
        await self.bus.publish(self.baseline_cfg.topic_single, msg)

    async def _publish_avg(self, valid: list[DetectionResult], t_now: float) -> None:
        Ts = [np.array(v.pose_world, dtype=np.float64) for v in valid]
        ws = [self._quality_weight(v) for v in valid]

        # Weighted translation average
        t_sum = np.zeros(3, dtype=np.float64)
        w_sum = 0.0
        quats: list[np.ndarray] = []
        for T, w in zip(Ts, ws):
            t_sum += float(w) * T[:3, 3]
            w_sum += float(w)
            quats.append(_rot_to_quat_wxyz(T[:3, :3]))

        if w_sum <= 1e-12:
            return

        t_avg = t_sum / w_sum
        q_avg = _avg_quat_markley(quats, ws)
        R_avg = _quat_wxyz_to_rot(q_avg)

        T_avg = np.eye(4, dtype=np.float64)
        T_avg[:3, :3] = R_avg
        T_avg[:3, 3] = t_avg

        msg = {
            "T": T_avg.tolist(),
            "t_host": float(t_now),
            "stamp": time.time(),
            "source": "baseline/avg",
            "n_used": int(len(valid)),
            "sources": [str(v.source) for v in valid],
            "weights": [float(w) for w in ws],
        }
        await self.bus.publish(self.baseline_cfg.topic_avg, msg)

    async def run(self) -> None:
        slog(
            component=self.__class__.__name__,
            event="run",
            topic_in="tracking/detections",
            topic_best=self.baseline_cfg.topic_best,
            topic_avg=self.baseline_cfg.topic_avg,
            topic_single=self.baseline_cfg.topic_single,
            dope_only=bool(self.baseline_cfg.dope_only),
            min_score=float(self.baseline_cfg.min_score),
            max_reproj_error_px=float(self.baseline_cfg.max_reproj_error_px),
            max_age_s=float(self.baseline_cfg.max_age_s),
        )

        try:
            while True:
                msg = await self._q_tracking.get()
                if not isinstance(msg, DetectionResult):
                    continue

                t_stage_start = time.perf_counter()
                t_now_wall = time.time()

                det = msg
                if self._is_eligible(det):
                    key = self._sensor_key(str(det.source))
                    self._latest[key] = det

                if self._perf_t0_wall is None or self._perf_t0_host is None:
                    self._perf_t0_wall = t_now_wall
                    self._perf_t0_host = float(det.t_host)

                expected_wall = self._perf_t0_wall + (
                    float(det.t_host) - self._perf_t0_host
                )
                sched_lag_ms = (t_now_wall - expected_wall) * 1000.0

                t_now = float(det.t_host)
                valid = self._current_valid(t_now)
                if not valid:
                    continue

                await self._publish_best(valid, t_now)
                await self._publish_single(valid, t_now)
                if len(valid) >= 2:
                    await self._publish_avg(valid, t_now)

                stage_compute_ms = (time.perf_counter() - t_stage_start) * 1000.0
                end_to_end_ms = stage_compute_ms + sched_lag_ms

                self._perf_count += 1
                self._perf_compute_ms_sum += stage_compute_ms
                self._perf_end_to_end_ms_sum += end_to_end_ms
                self._perf_sched_lag_ms_sum += sched_lag_ms

                if self._perf_count % self._perf_interval == 0:
                    n = float(self._perf_interval)
                    slog(
                        component=self.__class__.__name__,
                        event="perf",
                        samples=int(self._perf_interval),
                        compute_ms_avg=self._perf_compute_ms_sum / n,
                        frontend_to_baseline_ms_avg=self._perf_sched_lag_ms_sum / n,
                        end_to_end_ms_avg=self._perf_end_to_end_ms_sum / n,
                    )
                    self._perf_compute_ms_sum = 0.0
                    self._perf_end_to_end_ms_sum = 0.0
                    self._perf_sched_lag_ms_sum = 0.0
        finally:
            rem = self._perf_count % self._perf_interval
            if rem > 0:
                n = float(rem)
                slog(
                    component=self.__class__.__name__,
                    event="perf_final",
                    samples=int(rem),
                    compute_ms_avg=self._perf_compute_ms_sum / n,
                    frontend_to_baseline_ms_avg=self._perf_sched_lag_ms_sum / n,
                    end_to_end_ms_avg=self._perf_end_to_end_ms_sum / n,
                )
