"""Independent offline SE(3) estimators. No GT, production consensus or iSAM2.

Pose error is right-local: X_true = X Exp(dx); velocity is an additive body-twist
coordinate in rad/s, m/s. All tangent vectors use rotation then translation.
QualityScorer is reused ONLY to match the existing observation-noise mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import gtsam
import numpy as np

from config import FusionConfig
from managers.fusion_manager import Observation, QualityScorer


METHODS = (
    "se3_gaussian_quality", "se3_huber_fixed", "se3_huber_quality",
    "eskf_fixed", "eskf_quality", "eskf_gated_fixed", "eskf_gated_quality",
)


@dataclass(frozen=True)
class BaselineConfig:
    huber_k: float = 1.345
    max_iterations: int = 60
    relative_error_tol: float = 1e-7
    absolute_error_tol: float = 1e-9
    # Velocity random-walk spectral amplitudes: rad/s^(3/2), m/s^(3/2).
    angular_accel_density: float = 1.0
    linear_accel_density: float = 0.5
    initial_angular_velocity_sigma: float = 1.0
    initial_linear_velocity_sigma: float = 0.5
    nis_threshold: float = 16.8119  # chi-square(6), 99%; engineering prior
    max_prediction_gap_s: float = 0.5
    max_group_span_s: float = 0.04

    def __post_init__(self) -> None:
        if self.max_iterations < 1 or not isinstance(self.max_iterations, int):
            raise ValueError("max_iterations must be a positive integer")
        for key, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")


@dataclass
class Measurement:
    sensor: str
    pose: gtsam.Pose3
    covariance: np.ndarray


class BaselineRunError(RuntimeError):
    def __init__(self, message: str, predictions: list[dict], failure: dict):
        super().__init__(message)
        self.predictions, self.failure = predictions, failure


def rigid_pose(matrix: list | np.ndarray) -> gtsam.Pose3:
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)):
        raise ValueError("Expected a finite rigid 4x4 world pose")
    u, singular, vt = np.linalg.svd(matrix[:3, :3])
    if np.linalg.det(matrix[:3, :3]) <= 0 or np.max(np.abs(singular - 1)) > .01:
        raise ValueError("Non-rigid pose rotation")
    return gtsam.Pose3(gtsam.Rot3(u @ vt), matrix[:3, 3])


def positive_covariance(matrix: np.ndarray) -> np.ndarray:
    matrix = .5 * (matrix + matrix.T)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite covariance")
    eigenvalues, vectors = np.linalg.eigh(matrix)
    if eigenvalues.min() < -1e-8 * max(1., np.max(np.abs(eigenvalues))):
        raise ValueError("Indefinite covariance")
    return (vectors * np.maximum(eigenvalues, 1e-12)) @ vectors.T


def measurement(row: dict, cfg: FusionConfig, adaptive: bool) -> Measurement:
    """Match DetectionResult parsing in the cached production world-pose path.

    No IMU is supplied by this protocol, so velocity-induced noise is zero.
    Fixed means detector-specific constant noise, ignoring supplied covariance.
    Quality means exactly the supplied/full-or-quality-scaled production noise.
    """
    qm = row.get("quality_metrics") or {}
    source = row["source"].lower()
    detector = next((s for s in ("foundationpose", "megapose", "m3t", "seg3d", "dope")
                     if s in source), None)
    if detector is None:
        raise ValueError(f"Unknown cached detector source: {row['source']}")
    q_area = max(float(qm.get("q_area", qm.get("score", row.get("score", 1.)))), 0.)
    q_fit = math.exp(-max(float(qm.get("reproj_error", 0.)), 0.) / max(cfg.sigma_reproj, 1e-6))
    if "q_fit" in qm:
        q_fit = float(qm["q_fit"])
    elif qm.get("depth_inlier_fraction") is not None:
        q_fit = float(qm["depth_inlier_fraction"])
    capture = float(row.get("capture_time", row["t_host"]))
    q_time = min(float(qm.get("q_time", 1.)),
                 math.exp(-abs(float(row["t_host"]) - capture) / max(cfg.sync_window_s, 1e-3)))
    if not np.isfinite([q_area, q_fit, q_time, capture]).all():
        raise ValueError("Invalid measurement quality/time")
    pose = rigid_pose(row["pose_world"])
    noise_cfg = replace(cfg, use_adaptive_covariance=adaptive)
    if not adaptive:
        noise_cfg.covariance_calibration_path = ""
    obs = Observation(
        int(row["frame_id"]), row["camera_id"], "world", pose,
        q_area=q_area, q_fit=q_fit, q_time=q_time,
        rot_sigma=getattr(cfg, f"{detector}_rot_sigma"),
        trans_sigma=getattr(cfg, f"{detector}_trans_sigma"),
        covariance=row.get("covariance") if adaptive else None,
        detector=row.get("detector", "foundationpose"), capture_time=capture,
    )
    if cfg.covariance_calibration_path:
        raise ValueError("External covariance calibration is outside this frozen baseline protocol")
    covariance = QualityScorer(noise_cfg).pose_covariance(obs)
    return Measurement(str(row["camera_id"]), pose, positive_covariance(covariance))


def canonical_groups(observations: list[dict], max_span: float) -> list[list[dict]]:
    """Last-arrival replacement, then deterministic object/group/sensor ordering.

    Conflicting same-camera duplicates with tied arrival times are rejected:
    there is no recorded ordering evidence to choose one reproducibly.
    """
    latest: dict[tuple, dict] = {}
    for row in observations:
        key = (row["object_name"], int(row["frame_id"]), str(row["camera_id"]))
        arrival = float(row.get("arrival_time", row["t_host"]))
        if not math.isfinite(arrival):
            raise ValueError("Invalid arrival time")
        old = latest.get(key)
        if old is not None:
            previous = float(old.get("arrival_time", old["t_host"]))
            if arrival == previous and row != old:
                raise ValueError(f"Ambiguous tied duplicate: {key}")
            if arrival < previous:
                continue
        latest[key] = row
    groups: dict[tuple, list[dict]] = {}
    for key, row in sorted(latest.items()):
        groups.setdefault(key[:2], []).append(row)
    result = list(groups.values())
    for rows in result:
        times = [float(r.get("capture_time", r["t_host"])) for r in rows]
        if not np.isfinite(times).all() or max(times) - min(times) > max_span:
            raise ValueError("Capture group exceeds permitted time span")
    result.sort(key=lambda rows: (min(float(r.get("capture_time", r["t_host"])) for r in rows),
                                 rows[0]["object_name"], int(rows[0]["frame_id"])))
    return result


def pose_factor(key: int, obs: Measurement, huber: float) -> gtsam.CustomFactor:
    gaussian = gtsam.noiseModel.Gaussian.Covariance(obs.covariance)
    noise = (gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(huber), gaussian)
             if huber > 0 else gaussian)

    def error(_factor, values, jacobians):
        delta = obs.pose.between(values.atPose3(key))
        if jacobians is not None:
            jacobians[0] = gtsam.Pose3.LogmapDerivative(delta)
        return gtsam.Pose3.Logmap(delta)

    return gtsam.CustomFactor(noise, [key], error)


def batch_pose(measurements: list[Measurement], cfg: BaselineConfig,
               huber: float) -> tuple[gtsam.Pose3, np.ndarray, dict]:
    if not measurements:
        raise ValueError("Cannot initialize without observations")
    key = gtsam.symbol("x", 0)
    graph = gtsam.NonlinearFactorGraph()
    for obs in measurements:
        graph.add(pose_factor(key, obs, huber))
    # A deterministic measurement medoid under this exact objective, not GT.
    def seed_cost(obs):
        values = gtsam.Values()
        values.insert(key, obs.pose)
        return (graph.error(values), obs.sensor)
    seed = min(measurements, key=seed_cost)
    initial = gtsam.Values()
    initial.insert(key, seed.pose)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(cfg.max_iterations)
    params.setRelativeErrorTol(cfg.relative_error_tol)
    params.setAbsoluteErrorTol(cfg.absolute_error_tol)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    pose = result.atPose3(key)
    cost = float(graph.error(result))
    if not math.isfinite(cost) or not np.isfinite(pose.matrix()).all():
        raise ValueError("Nonfinite batch solution")
    covariance = positive_covariance(gtsam.Marginals(graph, result).marginalCovariance(key))
    return pose, covariance, {"iterations": optimizer.iterations(), "objective": cost,
                             "hit_iteration_limit": optimizer.iterations() >= cfg.max_iterations,
                             "initial_sensor": seed.sensor}


def motion_jacobian(velocity: np.ndarray, dt: float) -> np.ndarray:
    tangent = velocity * dt
    transition = np.eye(12)
    transition[:6, :6] = gtsam.Pose3.Expmap(-tangent).AdjointMap()
    transition[:6, 6:] = gtsam.Pose3.ExpmapDerivative(tangent) * dt
    return transition


_QUAD_X, _QUAD_W = np.polynomial.legendre.leggauss(5)


def process_covariance(velocity: np.ndarray, dt: float, cfg: BaselineConfig) -> np.ndarray:
    """Five-point quadrature of continuous body-twist white-acceleration noise.

    Integrates Phi(tau) L Qa L.T Phi(tau).T, preserving pose/velocity cross terms.
    It is exact for zero velocity and an approximation for nonzero motion.
    """
    density = np.diag([cfg.angular_accel_density**2]*3 + [cfg.linear_accel_density**2]*3)
    result = np.zeros((12, 12))
    for node, weight in zip(_QUAD_X, _QUAD_W):
        tau = .5*dt*(node + 1.)
        transport = np.vstack((gtsam.Pose3.ExpmapDerivative(velocity*tau)*tau, np.eye(6)))
        result += .5*dt*weight * (transport @ density @ transport.T)
    return result


def innovation(pose: gtsam.Pose3, obs: Measurement) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = pose.between(obs.pose)
    residual = gtsam.Pose3.Logmap(delta)
    log_jac = gtsam.Pose3.LogmapDerivative(delta)
    # -d Log(Exp(-dx) X^-1 Z)/d dx; measurement noise is right-local at Z.
    h = np.zeros((6, 12))
    h[:, :6] = log_jac @ delta.inverse().AdjointMap()
    noise = log_jac @ obs.covariance @ log_jac.T
    return residual, h, positive_covariance(noise)


class PoseVelocityFilter:
    def __init__(self, cfg: BaselineConfig, gated: bool):
        self.cfg, self.gated = cfg, gated
        self.pose: gtsam.Pose3 | None = None
        self.velocity = np.zeros(6)
        self.covariance = np.eye(12)
        self.timestamp: float | None = None
        self.last_observed: float | None = None

    def step(self, observations: list[Measurement], timestamp: float) -> tuple[gtsam.Pose3 | None, dict]:
        if not math.isfinite(timestamp) or (self.timestamp is not None and timestamp <= self.timestamp):
            raise ValueError("Filter capture times must be finite and strictly increasing")
        expired = (self.last_observed is not None
                   and timestamp - self.last_observed > self.cfg.max_prediction_gap_s)
        if self.pose is None or expired:
            if not observations:
                self.timestamp = timestamp
                return None, {"prediction_only": False, "status": "uninitialized_or_expired"}
            self.pose, pose_cov, init = batch_pose(observations, self.cfg, self.cfg.huber_k)
            self.velocity = np.zeros(6)
            self.covariance = np.zeros((12, 12))
            self.covariance[:6, :6] = pose_cov
            self.covariance[6:, 6:] = np.diag(
                [self.cfg.initial_angular_velocity_sigma**2]*3 +
                [self.cfg.initial_linear_velocity_sigma**2]*3)
            self.timestamp = self.last_observed = timestamp
            # Initialization consumes the observations once: no second Kalman update.
            return self.pose, {"prediction_only": False, "status": "reinitialized" if expired else "initialized",
                               "accepted": [o.sensor for o in observations], "rejected": [], "initializer": init}
        dt = timestamp - self.timestamp
        transition = motion_jacobian(self.velocity, dt)
        self.covariance = positive_covariance(transition @ self.covariance @ transition.T
                                             + process_covariance(self.velocity, dt, self.cfg))
        self.pose = self.pose.compose(gtsam.Pose3.Expmap(self.velocity*dt))
        self.timestamp = timestamp
        residuals, jacobians, noises, accepted, rejected, nis_values = [], [], [], [], [], {}
        for obs in observations:
            residual, h, noise = innovation(self.pose, obs)
            s = h @ self.covariance @ h.T + noise
            nis = float(residual @ np.linalg.solve(s, residual))
            nis_values[obs.sensor] = nis
            if self.gated and nis > self.cfg.nis_threshold:
                rejected.append(obs.sensor)
                continue
            residuals.append(residual)
            jacobians.append(h)
            noises.append(noise)
            accepted.append(obs.sensor)
        if accepted:
            h = np.vstack(jacobians)
            residual = np.concatenate(residuals)
            noise = np.zeros((len(residual), len(residual)))
            for i, block in enumerate(noises):
                noise[6*i:6*i+6, 6*i:6*i+6] = block
            p = self.covariance
            s = h @ p @ h.T + noise
            gain = np.linalg.solve(s, h @ p).T
            correction = gain @ residual
            joseph = np.eye(12) - gain @ h
            posterior = joseph @ p @ joseph.T + gain @ noise @ gain.T
            self.pose = self.pose.compose(gtsam.Pose3.Expmap(correction[:6]))
            self.velocity += correction[6:]
            reset = np.eye(12)
            reset[:6, :6] = gtsam.Pose3.ExpmapDerivative(correction[:6])
            self.covariance = positive_covariance(reset @ posterior @ reset.T)
            self.last_observed = timestamp
        return self.pose, {"prediction_only": not bool(accepted), "status": "updated" if accepted else "predicted",
                           "accepted": accepted, "rejected": rejected, "nis": nis_values,
                           "observation_age_s": timestamp - self.last_observed}


def run_baseline(observations: list[dict], method: str, cfg: BaselineConfig | None = None,
                 noise_cfg: FusionConfig | None = None) -> tuple[list[dict], dict]:
    if method not in METHODS:
        raise ValueError(f"Unknown independent method: {method}")
    cfg, noise_cfg = cfg or BaselineConfig(), noise_cfg or FusionConfig()
    started = time.perf_counter()
    groups = canonical_groups(observations, cfg.max_group_span_s)
    filters, predictions, compute = {}, [], []
    for rows in groups:
        first = rows[0]
        timestamp = min(float(r.get("capture_time", r["t_host"])) for r in rows)
        tick = time.perf_counter()
        try:
            measures = [measurement(r, noise_cfg, method.endswith("quality")) for r in rows]
            if method.startswith("se3_"):
                pose, cov, detail = batch_pose(measures, cfg, 0. if method == "se3_gaussian_quality" else cfg.huber_k)
                detail.update(prediction_only=False, accepted=[o.sensor for o in measures], rejected=[])
            else:
                filt = filters.setdefault(first["object_name"], PoseVelocityFilter(cfg, "gated" in method))
                pose, detail = filt.step(measures, timestamp)
                cov = filt.covariance[:6, :6]
            if pose is not None:
                if not np.isfinite(pose.matrix()).all():
                    raise ValueError("Nonfinite filter pose")
                predictions.append({"frame_id": int(first["frame_id"]), "object_name": first["object_name"],
                                    "pose_world": pose.matrix().tolist(), "covariance": cov.tolist(),
                                    "capture_time_s": timestamp, "observation_count": len(detail["accepted"]),
                                    "active_sensors": detail["accepted"], "prediction_only": detail["prediction_only"],
                                    "diagnostics": detail})
        except Exception as exc:
            raise BaselineRunError(str(exc), predictions, {
                "frame_id": int(first["frame_id"]), "object_name": first["object_name"],
                "exception": type(exc).__name__, "error": str(exc),
                "policy": "stop method; completed prefix retained; current group and suffix missing",
            }) from exc
        compute.append(1000*(time.perf_counter() - tick))
    return predictions, {"n_groups": len(groups), "elapsed_s": time.perf_counter() - started,
                         "compute_ms_mean_per_group": float(np.mean(compute)) if compute else None,
                         "compute_ms_p95_per_group": float(np.percentile(compute, 95)) if compute else None,
                         "timing_scope": "Offline final-group estimator including noise mapping, excluding cache I/O and frontend; not live latency"}
