# managers/fusion_manager.py
# pyright: reportMissingTypeArgument=false

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from common.structured_log import slog

import numpy as np

from config import BackendConfig, FusionConfig

from core.types import DetectionResult, TimeStampedPacket
from core.pose_consensus import SensorConsistencyHistory, select_pose_consensus


# ---------------------------------------------------------------------------
# GTSAM import (lazy — only needed when fusion is enabled)
# ---------------------------------------------------------------------------

try:
    import gtsam
    from gtsam import symbol

    _GTSAM_AVAILABLE = True
except ImportError:
    gtsam = None  # type: ignore[assignment]
    symbol = None  # type: ignore[assignment]
    _GTSAM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Observation — bridges upstream messages to GTSAM
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """
    A heterogeneous observation on the tracked object (tool/frame).

    Corresponds to a Pose3 measurement of the object in the sensor frame.
    Maps from the upstream DetectionResult / dict message format.
    """

    time_k: int  # discrete time step (k), monotonic integer
    sensor_id: str  # source identifier, e.g. "hl2", "ots", "rs_serial/dope"
    sensor_type: str  # "world" | "fixed" | "unknown" | "mobile"

    # If sensor_type == "world": measured_pose is T_world_tool / T_world_object
    # Otherwise: measured_pose is T_sensor_tool / T_sensor_object
    measured_pose: Any  # gtsam.Pose3 at runtime; dict or list before init

    # ---- Quality terms ----
    q_area: float = 1.0  # detection area / feature count quality
    q_fit: float = 1.0  # model fit / reprojection quality
    q_time: float = 1.0  # temporal consistency quality
    beta: float = 1.0  # base scaling
    rot_sigma: float | None = None
    trans_sigma: float | None = None

    # ---- Motion terms (for async uncertainty) ----
    omega_norm: float = 0.0  # angular velocity magnitude [rad/s]
    v_norm: float = 0.0  # linear velocity magnitude [m/s]
    delta_t: float = 0.0  # time since last measurement [s]
    t_host: float | None = None
    capture_time: float | None = None
    covariance: np.ndarray | None = None
    detector: str | None = None

    # ---- Fixed sensor: known world_T_sensor ----
    known_extrinsic: Any = None  # gtsam.Pose3 or None

    # ---- Mobile sensor: which camera-pose key to use ----
    mobile_cam_state_name: Optional[str] = None
    imu_delta_pose: Any = None
    imu_delta_covariance: np.ndarray | None = None


@dataclass
class _WorldStateRevision:
    observations: list[Observation]
    seed: Any
    motion: tuple[int, Any, float] | None


# ---------------------------------------------------------------------------
# Quality scorer — adaptive covariance from quality terms
# ---------------------------------------------------------------------------


class QualityScorer:
    """
    Converts quality metrics (area, fit, time) into an adaptive noise
    covariance for GTSAM Pose3 factors.
    """

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self._calibrated_sigmas: dict[str, tuple[float, float]] = {}
        self._calibrated_covariances: dict[str, np.ndarray] = {}
        path = str(getattr(cfg, "covariance_calibration_path", "") or "").strip()
        if path:
            try:
                import json
                from pathlib import Path

                data = json.loads(Path(path).read_text(encoding="utf-8"))
                for key, value in (data.get("sensors", data) or {}).items():
                    if not isinstance(value, dict):
                        continue
                    rot = float(value.get("rot_sigma", 0.0))
                    trans = float(value.get("trans_sigma", 0.0))
                    if rot > 0.0 and trans > 0.0:
                        self._calibrated_sigmas[str(key)] = (rot, trans)
                    covariance = np.asarray(
                        value.get("residual_covariance"), dtype=np.float64
                    )
                    if covariance.shape == (6, 6) and np.isfinite(covariance).all():
                        self._calibrated_covariances[str(key)] = covariance
            except Exception:
                # Calibration files are optional; retain deterministic defaults
                # and let the run manifest report the configured path.
                self._calibrated_sigmas = {}

    def compute_gamma(self, obs: Observation) -> float:
        if not bool(getattr(self.cfg, "use_adaptive_covariance", True)):
            return 1.0
        gamma = (
            obs.beta
            * (max(obs.q_area, 1e-6) ** self.cfg.w_area)
            * (max(obs.q_fit, 1e-6) ** self.cfg.w_fit)
            * (max(obs.q_time, 1e-6) ** self.cfg.w_time)
        )
        # The quality terms are intentionally bounded.  A detector score is
        # not a calibrated probability and must not be allowed to create an
        # arbitrarily confident factor.
        return float(
            np.clip(
                gamma,
                max(float(getattr(self.cfg, "covariance_min_scale", 0.25)), 1e-6),
                max(float(getattr(self.cfg, "covariance_max_scale", 100.0)), 1.0),
            )
        )

    def pose_sigmas(self, obs: Observation) -> np.ndarray:
        if obs.covariance is not None:
            covariance = np.asarray(obs.covariance, dtype=np.float64)
            if covariance.shape == (6, 6) and np.isfinite(covariance).all():
                diagonal = np.maximum(np.diag(covariance), 0.0)
                floors = np.array(
                    [
                        float(getattr(self.cfg, "covariance_floor_rot_rad", 1e-4)),
                    ]
                    * 3
                    + [
                        float(getattr(self.cfg, "covariance_floor_trans_m", 1e-4)),
                    ]
                    * 3
                )
                return np.sqrt(np.maximum(diagonal, floors**2))
        gamma = self.compute_gamma(obs)

        base_rot = self.cfg.sigma_R if obs.rot_sigma is None else obs.rot_sigma
        base_trans = self.cfg.sigma_t if obs.trans_sigma is None else obs.trans_sigma
        for key in (obs.sensor_id, obs.detector or ""):
            if key in self._calibrated_sigmas:
                base_rot, base_trans = self._calibrated_sigmas[key]
                break
        rot_var = (base_rot**2 + (obs.omega_norm * obs.delta_t) ** 2) / gamma
        trans_var = (base_trans**2 + (obs.v_norm * obs.delta_t) ** 2) / gamma

        sigmas = np.array(
            [
                math.sqrt(rot_var),
                math.sqrt(rot_var),
                math.sqrt(rot_var),
                math.sqrt(trans_var),
                math.sqrt(trans_var),
                math.sqrt(trans_var),
            ]
        )
        floors = np.array(
            [
                float(getattr(self.cfg, "covariance_floor_rot_rad", 1e-4)),
            ]
            * 3
            + [
                float(getattr(self.cfg, "covariance_floor_trans_m", 1e-4)),
            ]
            * 3
        )
        return np.maximum(sigmas, floors)

    def pose_covariance(self, obs: Observation) -> np.ndarray:
        """Return a finite, positive-semidefinite 6x6 observation covariance."""
        if obs.covariance is not None:
            covariance = np.asarray(obs.covariance, dtype=np.float64)
            if covariance.shape == (6, 6) and np.isfinite(covariance).all():
                covariance = 0.5 * (covariance + covariance.T)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                floors = np.array(
                    [
                        float(getattr(self.cfg, "covariance_floor_rot_rad", 1e-4)) ** 2,
                    ]
                    * 3
                    + [
                        float(getattr(self.cfg, "covariance_floor_trans_m", 1e-4)) ** 2,
                    ]
                    * 3
                )
                eigenvalues = np.maximum(eigenvalues, float(np.min(floors)))
                return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        for key in (obs.sensor_id, obs.detector or ""):
            covariance = self._calibrated_covariances.get(key)
            if covariance is not None:
                gamma = self.compute_gamma(obs)
                covariance = 0.5 * (covariance + covariance.T) / gamma
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                eigenvalues = np.maximum(eigenvalues, 1e-8)
                return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        sigmas = self.pose_sigmas(obs)
        return np.diag(sigmas**2)

    def make_pose_noise(self, obs: Observation) -> Any:
        """Return a robust GTSAM pose noise model conditioned on quality."""
        covariance = self.pose_covariance(obs)
        gaussian = gtsam.noiseModel.Gaussian.Covariance(covariance)
        huber_k = float(getattr(self.cfg, "robust_huber_k", 0.0))
        if huber_k <= 0.0:
            return gaussian
        robust = gtsam.noiseModel.mEstimator.Huber.Create(huber_k)
        return gtsam.noiseModel.Robust.Create(robust, gaussian)


# ---------------------------------------------------------------------------
# Integrity monitor — NIS and health state
# ---------------------------------------------------------------------------


class IntegrityMonitor:
    """Monitors estimation integrity via covariance trace and NIS."""

    _CHI2_95: Dict[int, float] = {
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.0705,
        6: 12.5916,
    }
    _CHI2_99: Dict[int, float] = {
        1: 6.6349,
        2: 9.2103,
        3: 11.3449,
        4: 13.2767,
        5: 15.0863,
        6: 16.8119,
    }

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg

    def uncertainty_metric(self, Sigma_post: np.ndarray) -> float:
        return float(np.trace(Sigma_post))

    def health_state(self, q: float, nis: float | None = None) -> str:
        """Classify the estimate using both uncertainty and innovation."""
        if q < self.cfg.delta_safe and (
            nis is None or nis <= self.chi2_threshold(6)
        ):
            return "Reliable"
        if q < self.cfg.delta_alarm and (
            nis is None or nis <= 2.0 * self.chi2_threshold(6)
        ):
            return "Degraded"
        return "Unsafe"

    def nis(self, residual: np.ndarray, S: np.ndarray) -> float:
        residual = np.asarray(residual, dtype=np.float64).reshape(-1, 1)
        S = np.asarray(S, dtype=np.float64)
        S = 0.5 * (S + S.T)
        try:
            solved = np.linalg.solve(S, residual)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(S) @ residual
        return float((residual.T @ solved).item())

    def chi2_threshold_95(self, dof: int = 6) -> float:
        return self._CHI2_95.get(dof, 12.5916)

    def chi2_threshold(self, dof: int = 6) -> float:
        probability = float(getattr(self.cfg, "nis_gate_probability", 0.99))
        if probability >= 0.99:
            return self._CHI2_99.get(dof, 16.8119)
        return self._CHI2_95.get(dof, 12.5916)


# ---------------------------------------------------------------------------
# HeteroFusionSystem — GTSAM iSAM2-based heterogeneous fusion
# ---------------------------------------------------------------------------


class HeteroFusionSystem:
    """
    iSAM2-based pose-graph fusion for heterogeneous sensors.

    Supports three sensor types:
      fixed    — known extrinsic (e.g. OptiTrack). Extrinsic is a fixed prior.
      unknown  — extrinsic is an online-calibrated variable in the graph.
      mobile   — camera pose at each step is a variable (e.g. RealSense + VO).
    """

    def __init__(self, cfg: FusionConfig):
        if not _GTSAM_AVAILABLE:
            raise ImportError(
                "gtsam is required for HeteroFusionSystem. Install: pip install gtsam"
            )

        self.cfg = cfg
        self.quality = QualityScorer(cfg)
        self.monitor = IntegrityMonitor(cfg)

        isam_params = gtsam.ISAM2Params()
        isam_params.setRelinearizeThreshold(cfg.relinearize_threshold)
        isam_params.relinearizeSkip = cfg.relinearize_skip
        self.isam = gtsam.ISAM2(isam_params)

        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()

        self.initialized = False

        self._tool_step: int = 0
        self._unknown_keys: Dict[str, int] = {}
        self._mobile_keys: Dict[tuple, int] = {}
        self._fixed_keys: Dict[str, int] = {}
        self._mobile_initialized: set[str] = set()
        self._unknown_initialized: set[str] = set()
        self._temporal_pairs: set[tuple[int, int]] = set()
        self._mobile_temporal_pairs: set[tuple[str, int, int]] = set()
        self._unknown_initial_poses: Dict[str, Any] = {}
        self._world_factor_indices: dict[int, list[int]] = {}
        self._pending_world_states: dict[int, _WorldStateRevision] = {}
        # One checkpoint per object, not one per historical frame. It is valid
        # only until another state or a camera/extrinsic update is committed.
        self._world_revision_checkpoint: tuple[int, Any] | None = None

    # ---- key helpers ----

    def tool_key(self, k: int) -> int:
        return symbol("x", k)

    def _cam_idx(self, sensor_name: str) -> int:
        key = (sensor_name,)
        if key not in self._mobile_keys:
            self._mobile_keys[key] = len(self._mobile_keys) + 1
        return self._mobile_keys[key]

    def _ext_idx(self, sensor_name: str) -> int:
        if sensor_name not in self._unknown_keys:
            self._unknown_keys[sensor_name] = len(self._unknown_keys) + 1
        return self._unknown_keys[sensor_name]

    def cam_key(self, sensor_name: str, k: int) -> int:
        key = (sensor_name, k)
        if key not in self._mobile_keys:
            self._mobile_keys[key] = len(self._mobile_keys) + 1
        return symbol("m", self._mobile_keys[key])

    def ext_key(self, sensor_name: str) -> int:
        idx = self._ext_idx(sensor_name)
        return symbol("c", idx)

    def fixed_key(self, sensor_name: str) -> int:
        """Return a process-stable GTSAM key without Python hash randomization."""
        if sensor_name not in self._fixed_keys:
            self._fixed_keys[sensor_name] = len(self._fixed_keys) + 1
        return symbol("f", self._fixed_keys[sensor_name])

    # ---- initialization helpers ----

    def _pose3_from_matrix(self, T: np.ndarray) -> Any:
        R = gtsam.Rot3(T[:3, :3])
        t = gtsam.Point3(T[:3, 3])
        return gtsam.Pose3(R, t)

    def _prior_noise(self) -> Any:
        sigmas = np.array(
            [
                self.cfg.prior_rot_sigma,
                self.cfg.prior_rot_sigma,
                self.cfg.prior_rot_sigma,
                self.cfg.prior_trans_sigma,
                self.cfg.prior_trans_sigma,
                self.cfg.prior_trans_sigma,
            ]
        )
        return gtsam.noiseModel.Diagonal.Sigmas(sigmas)

    def initialize_tool_pose(self, k: int, init_pose: Any) -> None:
        key = self.tool_key(k)
        if not self.new_values.exists(key):
            self.new_values.insert(key, init_pose)
        self.new_factors.add(
            gtsam.PriorFactorPose3(key, init_pose, self._prior_noise())
        )
        self.initialized = True
    def ensure_tool_state(self, k: int, init_pose: Any | None = None) -> None:
        """
        Ensure x_k exists.

        Only the first initialized tool state receives a prior.
        Later states only receive an initial value, not an extra prior.
        """
        key = self.tool_key(k)
        current = self.current_values()

        if current.exists(key) or self.new_values.exists(key):
            return

        if init_pose is None and k > 0:
            prev_key = self.tool_key(k - 1)

            if current.exists(prev_key):
                init_pose = current.atPose3(prev_key)
            elif self.new_values.exists(prev_key):
                init_pose = self.new_values.atPose3(prev_key)

        if init_pose is None:
            init_pose = gtsam.Pose3()

        self.new_values.insert(key, init_pose)

        if not self.initialized:
            self.new_factors.add(
                gtsam.PriorFactorPose3(key, init_pose, self._prior_noise())
            )
            self.initialized = True
    def initialize_mobile_cam(self, sensor_name: str, k: int, init_pose: Any) -> None:
        key = self.cam_key(sensor_name, k)
        if not self.new_values.exists(key):
            self.new_values.insert(key, init_pose)
        # Only the first pose of a mobile trajectory is anchored.  Adding a
        # prior at every frame would silently disable the motion/IMU factors.
        if sensor_name not in self._mobile_initialized:
            self.new_factors.add(
                gtsam.PriorFactorPose3(key, init_pose, self._prior_noise())
            )
            self._mobile_initialized.add(sensor_name)

    def initialize_unknown_extrinsic(
        self, sensor_name: str, init_pose: Any, prior_noise: Any | None = None
    ) -> None:
        key = self.ext_key(sensor_name)
        if not self.new_values.exists(key):
            self.new_values.insert(key, init_pose)
        if sensor_name in self._unknown_initialized:
            return
        self._unknown_initial_poses[sensor_name] = init_pose
        if prior_noise is None:
            loose = np.array(
                [
                    float(getattr(self.cfg, "online_extrinsic_prior_rot_sigma", 0.35)),
                ]
                * 3
                + [
                    float(getattr(self.cfg, "online_extrinsic_prior_trans_sigma", 0.10)),
                ]
                * 3
            )
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(loose)
        self.new_factors.add(gtsam.PriorFactorPose3(key, init_pose, prior_noise))
        self._unknown_initialized.add(sensor_name)

    # ---- motion factors ----

    def add_tool_motion(
        self,
        k_prev: int,
        k_curr: int,
        rel_pose: Any,
        delta_t: float | None = None,
    ) -> None:
        if (k_prev, k_curr) in self._temporal_pairs:
            return
        key_prev = self.tool_key(k_prev)
        key_curr = self.tool_key(k_curr)

        current = self.current_values()
        if not current.exists(key_curr) and not self.new_values.exists(key_curr):
            guess = (
                current.atPose3(key_prev).compose(rel_pose)
                if current.exists(key_prev)
                else rel_pose
            )
            self.new_values.insert(key_curr, guess)

        dt_scale = max(float(delta_t or 0.0), 1e-3)
        dt_scale = max(dt_scale, 1.0 / 60.0)
        motion_scale = math.sqrt(dt_scale / (1.0 / 60.0))
        motion_scale *= float(getattr(self.cfg, "temporal_process_scale", 1.0))
        motion_sigmas = np.array(
            [
                self.cfg.motion_rot_sigma * motion_scale,
                self.cfg.motion_rot_sigma * motion_scale,
                self.cfg.motion_rot_sigma * motion_scale,
                self.cfg.motion_trans_sigma * motion_scale,
                self.cfg.motion_trans_sigma * motion_scale,
                self.cfg.motion_trans_sigma * motion_scale,
            ]
        )
        self.new_factors.add(
            gtsam.BetweenFactorPose3(
                key_prev,
                key_curr,
                rel_pose,
                gtsam.noiseModel.Diagonal.Sigmas(motion_sigmas),
            )
        )
        self._temporal_pairs.add((k_prev, k_curr))

    def add_mobile_cam_motion(
        self,
        sensor_name: str,
        k_prev: int,
        k_curr: int,
        rel_pose: Any,
        delta_t: float | None = None,
        covariance: np.ndarray | None = None,
    ) -> None:
        edge_key = (sensor_name, k_prev, k_curr)
        if edge_key in self._mobile_temporal_pairs:
            return
        key_prev = self.cam_key(sensor_name, k_prev)
        key_curr = self.cam_key(sensor_name, k_curr)

        current = self.current_values()
        if not current.exists(key_prev) and not self.new_values.exists(key_prev):
            self.initialize_mobile_cam(sensor_name, k_prev, gtsam.Pose3())

        if not current.exists(key_curr) and not self.new_values.exists(key_curr):
            base = (
                current.atPose3(key_prev)
                if current.exists(key_prev)
                else self.new_values.atPose3(key_prev)
            )
            self.new_values.insert(key_curr, base.compose(rel_pose))

        dt_scale = max(float(delta_t or 0.0), 1e-3)
        dt_scale = max(dt_scale, 1.0 / 60.0)
        motion_scale = math.sqrt(dt_scale / (1.0 / 60.0))
        motion_scale *= float(getattr(self.cfg, "temporal_process_scale", 1.0))
        if covariance is not None and np.asarray(covariance).shape == (6, 6):
            covariance = np.asarray(covariance, dtype=np.float64)
            noise = gtsam.noiseModel.Gaussian.Covariance(
                0.5 * (covariance + covariance.T)
            )
        else:
            motion_sigmas = np.array(
                [
                    self.cfg.cam_motion_rot_sigma * motion_scale,
                    self.cfg.cam_motion_rot_sigma * motion_scale,
                    self.cfg.cam_motion_rot_sigma * motion_scale,
                    self.cfg.cam_motion_trans_sigma * motion_scale,
                    self.cfg.cam_motion_trans_sigma * motion_scale,
                    self.cfg.cam_motion_trans_sigma * motion_scale,
                ]
            )
            noise = gtsam.noiseModel.Diagonal.Sigmas(motion_sigmas)
        self.new_factors.add(
            gtsam.BetweenFactorPose3(
                key_prev,
                key_curr,
                rel_pose,
                noise,
            )
        )
        self._mobile_temporal_pairs.add(edge_key)

    # ---- observation factor ----

    def add_observation(self, obs: Observation) -> None:
        """
        Add a Pose3 factor for the tracked object.

        World_T_tool = World_T_sensor * Sensor_T_tool
        => Between(World_T_sensor, World_T_tool) = Sensor_T_tool
        """
        tool_k = self.tool_key(obs.time_k)
        current = self.current_values()

        # Ensure tool state exists
        initial_tool_pose = None
        if obs.sensor_type == "world":
            initial_tool_pose = obs.measured_pose
        elif obs.sensor_type in ("fixed", "unknown"):
            if obs.known_extrinsic is not None:
                initial_tool_pose = obs.known_extrinsic.compose(obs.measured_pose)
        elif obs.sensor_type == "mobile":
            initial_tool_pose = obs.measured_pose
        self.ensure_tool_state(obs.time_k, initial_tool_pose)

        noise = self.quality.make_pose_noise(obs)

        
        if obs.sensor_type == "world":
            # Direct world-frame pose observation:
            # x_k = T_world_tool is directly measured by this camera stream.
            self.new_factors.add(
                gtsam.PriorFactorPose3(tool_k, obs.measured_pose, noise)
            )

        elif obs.sensor_type == "fixed":
            if obs.known_extrinsic is None:
                raise ValueError("Fixed sensor requires known_extrinsic (gtsam.Pose3).")

            # Add a tiny-fixed node for the known extrinsic
            fixed_key = self.fixed_key(obs.sensor_id)
            if not current.exists(fixed_key) and not self.new_values.exists(fixed_key):
                self.new_values.insert(fixed_key, obs.known_extrinsic)
                tiny = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-9)
                self.new_factors.add(
                    gtsam.PriorFactorPose3(fixed_key, obs.known_extrinsic, tiny)
                )

            self.new_factors.add(
                gtsam.BetweenFactorPose3(fixed_key, tool_k, obs.measured_pose, noise)
            )

        elif obs.sensor_type == "unknown":
            if bool(getattr(self.cfg, "enable_online_calibration", True)):
                ext_k = self.ext_key(obs.sensor_id)
                if not current.exists(ext_k) and not self.new_values.exists(ext_k):
                    self.initialize_unknown_extrinsic(
                        obs.sensor_id,
                        obs.known_extrinsic
                        if obs.known_extrinsic is not None
                        else gtsam.Pose3(),
                    )

                self.new_factors.add(
                    gtsam.BetweenFactorPose3(ext_k, tool_k, obs.measured_pose, noise)
                )
            else:
                fixed_key = self.fixed_key(obs.sensor_id + "_fixed")
                fixed_pose = (
                    obs.known_extrinsic
                    if obs.known_extrinsic is not None
                    else gtsam.Pose3()
                )
                if not current.exists(fixed_key) and not self.new_values.exists(
                    fixed_key
                ):
                    self.new_values.insert(fixed_key, fixed_pose)
                    tiny = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-9)
                    self.new_factors.add(
                        gtsam.PriorFactorPose3(fixed_key, fixed_pose, tiny)
                    )
                self.new_factors.add(
                    gtsam.BetweenFactorPose3(
                        fixed_key, tool_k, obs.measured_pose, noise
                    )
                )

        elif obs.sensor_type == "mobile":
            cam_k = self.cam_key(obs.sensor_id, obs.time_k)
            if not current.exists(cam_k) and not self.new_values.exists(cam_k):
                self.initialize_mobile_cam(obs.sensor_id, obs.time_k, gtsam.Pose3())

            self.new_factors.add(
                gtsam.BetweenFactorPose3(cam_k, tool_k, obs.measured_pose, noise)
            )
        else:
            raise ValueError(f"Unknown sensor_type: {obs.sensor_type}")

    # ---- update & query ----

    def revise_world_state(self, k: int, observations: list[Observation], seed: Any,
                           motion: tuple[int, Any, float] | None = None) -> None:
        """Replace the complete measurement set, including any provisional choice."""
        key = self.tool_key(k)
        if not self.current_values().exists(key):
            if self.new_values.exists(key):
                self.new_values.update(key, seed)
            else:
                self.new_values.insert(key, seed)
        # World measurements anchor the graph themselves; do not double-count
        # the first camera as an additional permanent initialization prior.
        self.initialized = True
        self._pending_world_states[k] = _WorldStateRevision(observations, seed, motion)

    def _prepare_world_revision(self) -> tuple[Any, int | None]:
        """Remove provisional solver history without buffering sensor input.

        A copied pre-state solver preserves all earlier factors and estimates.
        Revisions use the latest complete measurement set and consensus seed,
        so an excluded first view cannot leave a different linearization.
        Mixed/batched updates and late edits to older states keep the ordinary
        factor-replacement path; they must never rewind unrelated updates.
        """
        if (not self.cfg.enable_world_revision_checkpoint
                or len(self._pending_world_states) != 1
                or self.new_factors.size() != 0):
            self._world_revision_checkpoint = None
            return self.isam, None
        k = next(iter(self._pending_world_states))
        key = self.tool_key(k)
        if any(value_key != key for value_key in self.new_values.keys()):
            self._world_revision_checkpoint = None
            return self.isam, None
        checkpoint = self._world_revision_checkpoint
        if checkpoint is not None and checkpoint[0] == k:
            self.new_values = gtsam.Values()
            self.new_values.insert(key, self._pending_world_states[k].seed)
            # Keep the last successful solver/indices until the revised update
            # succeeds. An optimizer exception must not erase published state.
            return gtsam.ISAM2(checkpoint[1]), k
        elif not self.current_values().exists(key):
            self._world_revision_checkpoint = (k, gtsam.ISAM2(self.isam))
        else:
            self._world_revision_checkpoint = None
        return self.isam, None

    def update(self) -> None:
        solver, rewound_step = self._prepare_world_revision()
        remove = []
        slots: dict[int, list[int]] = {}
        for k, revision in self._pending_world_states.items():
            if k != rewound_step:
                remove.extend(self._world_factor_indices.get(k, []))
            slots[k] = []
            for obs in revision.observations:
                slots[k].append(self.new_factors.size())
                self.new_factors.add(gtsam.PriorFactorPose3(
                    self.tool_key(k), obs.measured_pose, self.quality.make_pose_noise(obs)))
            if revision.motion is not None:
                previous, relative, dt = revision.motion
                scale = math.sqrt(max(dt, 1.0 / 60.0) * 60) * self.cfg.temporal_process_scale
                sigmas = np.array([self.cfg.motion_rot_sigma] * 3 + [self.cfg.motion_trans_sigma] * 3) * scale
                noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
                if self.cfg.robust_huber_k > 0:
                    noise = gtsam.noiseModel.Robust.Create(
                        gtsam.noiseModel.mEstimator.Huber.Create(self.cfg.robust_huber_k), noise)
                slots[k].append(self.new_factors.size())
                self.new_factors.add(gtsam.BetweenFactorPose3(self.tool_key(previous), self.tool_key(k), relative, noise))
        result = solver.update(self.new_factors, self.new_values, remove)
        self.isam = solver
        indices = list(result.getNewFactorsIndices())
        for k, positions in slots.items():
            self._world_factor_indices[k] = [indices[i] for i in positions]
        self._pending_world_states.clear()
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()

    def reset_pending(self) -> None:
        self._world_revision_checkpoint = None
        self._pending_world_states.clear()
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()

    def current_values(self) -> Any:
        return self.isam.calculateEstimate()

    def extrinsic_estimates(self) -> Dict[str, np.ndarray]:
        """Return current online-calibrated world-to-sensor estimates."""
        current = self.current_values()
        output: Dict[str, np.ndarray] = {}
        for sensor_name in self._unknown_initialized:
            key = self.ext_key(sensor_name)
            if current.exists(key):
                output[sensor_name] = current.atPose3(key).matrix()
        return output

    def extrinsic_corrections(self) -> Dict[str, Dict[str, float]]:
        current = self.current_values()
        output: Dict[str, Dict[str, float]] = {}
        for sensor_name, initial in self._unknown_initial_poses.items():
            key = self.ext_key(sensor_name)
            if not current.exists(key):
                continue
            delta = initial.between(current.atPose3(key))
            output[sensor_name] = {
                "translation_m": float(np.linalg.norm(delta.translation())),
                "rotation_deg": float(
                    np.degrees(np.linalg.norm(gtsam.Rot3.Logmap(delta.rotation())))
                ),
            }
        return output

    def marginal_covariance(self, key: int) -> np.ndarray:
        return np.array(self.isam.marginalCovariance(key))

    def estimate_tool_pose(self, k: int) -> Optional[np.ndarray]:
        current = self.current_values()
        key = self.tool_key(k)
        if not current.exists(key):
            return None
        pose: Any = current.atPose3(key)
        return pose.matrix()

    def integrity_report(
        self, k: int, nis_values: list[float] | None = None
    ) -> Dict[str, Any]:
        if not bool(getattr(self.cfg, "enable_integrity_monitoring", True)):
            return {"enabled": False}
        key = self.tool_key(k)
        Sigma = self.marginal_covariance(key)
        q = self.monitor.uncertainty_metric(Sigma)
        finite_nis = [float(v) for v in (nis_values or []) if np.isfinite(v)]
        max_nis = max(finite_nis) if finite_nis else None
        health = self.monitor.health_state(q, max_nis)
        return {
            "enabled": True,
            "Q": q,
            "health": health,
            "trace_covariance": q,
            "position_std_m": float(np.sqrt(max(np.trace(Sigma[3:, 3:]), 0.0))),
            "orientation_std_rad": float(
                np.sqrt(max(np.trace(Sigma[:3, :3]), 0.0))
            ),
            "max_nis": max_nis,
            "nis_threshold": self.monitor.chi2_threshold(6),
        }

    def nis_report(self, obs: Observation) -> Dict[str, float]:
        if not bool(getattr(self.cfg, "enable_integrity_monitoring", True)):
            return {"enabled": 0.0}
        current = self.current_values()
        tool_k = self.tool_key(obs.time_k)
        if not current.exists(tool_k):
            raise RuntimeError("Tool state not yet estimated.")

        world_T_tool = current.atPose3(tool_k)

        if obs.sensor_type == "world":
            pred = world_T_tool
        elif obs.sensor_type == "fixed":
            pred = obs.known_extrinsic.inverse().compose(world_T_tool)
        elif obs.sensor_type == "unknown":
            world_T_sensor = current.atPose3(self.ext_key(obs.sensor_id))
            pred = world_T_sensor.inverse().compose(world_T_tool)
        elif obs.sensor_type == "mobile":
            world_T_sensor = current.atPose3(self.cam_key(obs.sensor_id, obs.time_k))
            pred = world_T_sensor.inverse().compose(world_T_tool)
        else:
            raise ValueError(obs.sensor_type)

        err_pose = pred.between(obs.measured_pose)
        residual = np.array(gtsam.Pose3.Logmap(err_pose)).reshape(6, 1)
        Sigma_x = self.marginal_covariance(tool_k)
        sigmas = self.quality.pose_sigmas(obs)
        R = np.diag(sigmas**2)
        S = Sigma_x + R
        nis_val = self.monitor.nis(residual, S)
        threshold = self.monitor.chi2_threshold(6)
        return {
            "nis": nis_val,
            "threshold_95": threshold,
            "consistent": float(nis_val < threshold),
        }


# ---------------------------------------------------------------------------
# FusionManager — EventBus wrapper around HeteroFusionSystem
# ---------------------------------------------------------------------------


class FusionManager:
    """
    Async wrapper that bridges the EventBus to GTSAM HeteroFusionSystem.

    Subscribes to upstream pose topics and feeds them into the pose graph.
    Publishes fused tool pose to the configured fused topic on every update.

    Supported sensor types for GTSAM fusion:
      mobile  — RealSense / HL2 camera with unknown motion (default)
      fixed   — OptiTrack or similar with known extrinsic
      unknown — online-calibrated extrinsic (e.g. AprilTag on world frame)
    """

    def __init__(self, bus, loop, cfg: BackendConfig):
        self.bus = bus
        self.loop = loop
        self.cfg = cfg
        self.fusion_cfg: FusionConfig = cfg.fusion

        # ---- upstream queues ----
        self._q_hl2 = bus.subscribe(cfg.fusion.topic_hl2)
        self._q_ots = bus.subscribe(cfg.fusion.topic_optitrack)
        self._obs_topic = "tracking/detections"
        if getattr(cfg, "frontend", None) is not None:
            topic_override = str(
                getattr(cfg.frontend, "topic_observations", "") or ""
            ).strip()
            if topic_override:
                self._obs_topic = topic_override
        self._q_tracking = bus.subscribe(self._obs_topic)
        self._q_realsense = bus.subscribe("realsense_stream")
        # Optional inertial streams.  Existing producers need not publish
        # these topics; empty queues preserve the old RGB-D-only behavior.
        self._q_imu = [
            bus.subscribe("quest/imu"),
            bus.subscribe("hl2/imu/gyro"),
            bus.subscribe("hl2/imu/accel"),
        ]

        # ---- GTSAM system ----
        if not _GTSAM_AVAILABLE:
            raise ImportError(
                "gtsam is not installed in the mr_nav environment. "
                "Run: pip install gtsam"
            )
        # per-object GTSAM fusion instances
        self._gtsam_instances: Dict[str, HeteroFusionSystem] = {}

        # per-object discrete time step counters
        self._steps: Dict[str, int] = {}
        self._time_bins: Dict[str, tuple[float, int]] = {}
        self._time_bins_by_object: Dict[
            str, list[tuple[float, int, set[str]]]
        ] = {}
        self._time_bin_sensors: Dict[str, set[str]] = {}
        self._step_timestamps: Dict[tuple[str, int], list[float]] = {}
        self._step_sync_groups: Dict[tuple[str, int], int] = {}
        self._step_observation_counts: Dict[tuple[str, int], int] = {}
        self._step_observations: Dict[tuple[str, int], list[Observation]] = {}
        self._best_state_sigma: Dict[tuple[str, int], float] = {}
        self._last_object_step: Dict[str, int] = {}
        self._last_object_time: Dict[str, float] = {}
        self._object_pose_history: Dict[str, list[tuple[int, float, Any]]] = {}
        self._temporal_added: set[tuple[str, int, int]] = set()
        self._last_sensor_capture: Dict[tuple[str, str], float] = {}
        self._sensor_integrity_failures: Dict[str, int] = {}
        self._sensor_integrity_successes: Dict[str, int] = {}
        self._sensor_health: Dict[str, str] = {}
        self._imu_state: Dict[str, dict[str, Any]] = {}
        self._world_candidates: dict[tuple[str, int], dict[str, Observation]] = {}
        self._consensus_reports: dict[tuple[str, int], dict[str, Any]] = {}
        self._consistency_history = SensorConsistencyHistory(
            self.fusion_cfg.integrity_failures_before_unsafe,
            self.fusion_cfg.integrity_recovery_before_reliable)

        # track latest message per sensor (for motion factor injection)
        self._last_sensor_T: Dict[tuple[str, str], Any] = {}
        self._last_sensor_step: Dict[tuple[str, str], int] = {}

        self._perf_interval = 20
        self._perf_loop_count = 0
        self._perf_loop_ms_sum = 0.0
        self._perf_obs_count = 0
        self._perf_obs_sched_lag_ms_sum = 0.0
        self._perf_t0_wall: float | None = None
        self._perf_t0_host: float | None = None
        self._offline_replay: bool = False
        self._last_obs_debug: Dict[str, Any] = {}

        # ---- load tag-cube calibration for fixed / unknown sensors ----
        self._known_extrinsics: Dict[str, Any] = {}  # serial → gtsam.Pose3
        self._sensor_modes: Dict[
            str, str
        ] = {}  # serial → "fixed" | "unknown" | "mobile"
        self._load_calibration()

    def configure_runtime(self, *, offline_replay: bool = False) -> None:
        self._offline_replay = bool(offline_replay)

    @staticmethod
    def _extract_object_name(sensor_id: str) -> str:
        """
        Extract object name from sensor_id like 'realsense/xxx/dope/Tuna' → 'Tuna'.
        Non-DOPE sources get '__default__'.
        """
        for marker in ("/foundationpose/", "/dope/", "/m3t/", "/megapose/"):
            parts = sensor_id.split(marker)
            if len(parts) == 2 and parts[1]:
                return parts[1]
        return "__default__"

    def _assign_step(
        self,
        object_name: str,
        sensor_id: str,
        t_host: float | None,
        sync_group: int | None = None,
    ) -> int | None:
        # Explicit capture-time grouping is authoritative. It remains stable
        # even when GPU inference completes out of order.
        if sync_group is not None:
            step = int(sync_group)
            self._steps[object_name] = max(self._steps.get(object_name, 0), step + 1)
            return step
        previous = self._time_bins.get(object_name)
        if t_host is not None and bool(
            getattr(self.fusion_cfg, "enable_time_compensation", True)
        ):
            bins = self._time_bins_by_object.setdefault(object_name, [])
            max_reorder = max(
                float(getattr(self.fusion_cfg, "max_reorder_s", 0.25)),
                float(self.fusion_cfg.sync_window_s),
            )
            if previous is not None and t_host < previous[0] - max_reorder:
                return None
            candidates = [
                item
                for item in bins
                if abs(t_host - item[0]) <= float(self.fusion_cfg.sync_window_s)
                and sensor_id not in item[2]
            ]
            if candidates:
                chosen = min(candidates, key=lambda item: abs(t_host - item[0]))
                chosen[2].add(sensor_id)
                self._time_bins[object_name] = (max(previous[0], t_host), chosen[1]) if previous else (t_host, chosen[1])
                self._time_bin_sensors[object_name] = set(chosen[2])
                return chosen[1]

        step = self._steps.get(object_name, 0)
        self._steps[object_name] = step + 1
        if t_host is not None:
            self._time_bins[object_name] = (t_host, step)
            self._time_bin_sensors[object_name] = {sensor_id}
            bins = self._time_bins_by_object.setdefault(object_name, [])
            bins.append((t_host, step, {sensor_id}))
            max_reorder = max(
                float(getattr(self.fusion_cfg, "max_reorder_s", 0.25)),
                float(self.fusion_cfg.sync_window_s),
            )
            cutoff = t_host - max_reorder
            self._time_bins_by_object[object_name] = [
                item for item in bins if item[0] >= cutoff
            ]
        return step

    def _get_system(self, object_name: str) -> HeteroFusionSystem:
        if object_name not in self._gtsam_instances:
            self._gtsam_instances[object_name] = HeteroFusionSystem(self.fusion_cfg)
            slog(
                component=self.__class__.__name__,
                event="new_fusion_instance",
                object_name=object_name,
            )
        return self._gtsam_instances[object_name]

    @staticmethod
    def _drain_queue_nowait(q: asyncio.Queue) -> list[Any]:
        out: list[Any] = []
        while True:
            try:
                out.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
        return out

    async def _update_and_publish(self, object_name: str, current_step: int) -> None:
        gtsam = self._gtsam_instances[object_name]
        try:
            gtsam.update()
        except Exception as exc:
            slog(
                component=self.__class__.__name__,
                event="gtsam_update_error",
                error=str(exc),
                step=int(current_step),
                object_name=object_name,
                last_obs=self._last_obs_debug,
            )
            gtsam.reset_pending()
            return

        fused = gtsam.estimate_tool_pose(current_step)
        if fused is None:
            return
        try:
            fused_pose = gtsam.current_values().atPose3(gtsam.tool_key(current_step))
            history = self._object_pose_history.setdefault(object_name, [])
            history_item = (
                int(current_step),
                float(
                    min(
                        self._step_timestamps.get(
                            (object_name, current_step), [time.time()]
                        )
                    )
                ),
                fused_pose,
            )
            history[:] = [item for item in history if item[0] != int(current_step)] + [history_item]
            history.sort(key=lambda item: (item[1], item[0]))
            if len(history) > 8:
                del history[:-8]
        except Exception:
            pass

        covariance = None
        try:
            covariance = gtsam.marginal_covariance(gtsam.tool_key(current_step))
        except Exception:
            covariance = None

        nis_values: list[float] = []
        sensor_health: dict[str, str] = {}
        consensus_report = self._consensus_reports.get((object_name, current_step))
        if bool(getattr(self.fusion_cfg, "enable_integrity_monitoring", True)) and consensus_report is None:
            for obs in self._step_observations.get((object_name, current_step), []):
                try:
                    nis_report = gtsam.nis_report(obs)
                    nis_value = float(nis_report.get("nis", float("nan")))
                    if np.isfinite(nis_value):
                        nis_values.append(nis_value)
                    threshold = float(nis_report.get("threshold_95", float("inf")))
                    bad = not bool(nis_report.get("consistent", 0.0))
                    if bad or nis_value > threshold:
                        self._sensor_integrity_failures[obs.sensor_id] = (
                            self._sensor_integrity_failures.get(obs.sensor_id, 0) + 1
                        )
                        self._sensor_integrity_successes[obs.sensor_id] = 0
                    else:
                        self._sensor_integrity_successes[obs.sensor_id] = (
                            self._sensor_integrity_successes.get(obs.sensor_id, 0) + 1
                        )
                        self._sensor_integrity_failures[obs.sensor_id] = 0
                    failures = self._sensor_integrity_failures.get(obs.sensor_id, 0)
                    successes = self._sensor_integrity_successes.get(obs.sensor_id, 0)
                    if failures >= int(
                        getattr(self.fusion_cfg, "integrity_failures_before_unsafe", 2)
                    ):
                        sensor_health[obs.sensor_id] = "Unsafe"
                    elif successes >= int(
                        getattr(self.fusion_cfg, "integrity_recovery_before_reliable", 3)
                    ):
                        sensor_health[obs.sensor_id] = "Reliable"
                    else:
                        sensor_health[obs.sensor_id] = "Degraded"
                    self._sensor_health[obs.sensor_id] = sensor_health[obs.sensor_id]
                except Exception as exc:
                    slog(
                        component=self.__class__.__name__,
                        event="integrity_sensor_check_error",
                        sensor_id=obs.sensor_id,
                        error=str(exc),
                    )
            try:
                max_t = float(
                    getattr(self.fusion_cfg, "online_extrinsic_max_correction_m", 0.25)
                )
                max_r = float(
                    getattr(
                        self.fusion_cfg, "online_extrinsic_max_correction_deg", 30.0
                    )
                )
                for sensor_name, correction in gtsam.extrinsic_corrections().items():
                    if (
                        correction["translation_m"] > max_t
                        or correction["rotation_deg"] > max_r
                    ):
                        sensor_health[sensor_name] = "Unsafe"
                        self._sensor_health[sensor_name] = "Unsafe"
                        slog(
                            component=self.__class__.__name__,
                            event="online_extrinsic_correction_exceeded",
                            sensor_id=sensor_name,
                            correction=correction,
                        )
            except Exception as exc:
                slog(
                    component=self.__class__.__name__,
                    event="online_extrinsic_check_error",
                    error=str(exc),
                )

        out = {
            "object_name": object_name,
            "T": fused.tolist(),
            "step": current_step,
            "sync_group": self._step_sync_groups.get((object_name, current_step)),
            "t_capture": min(
                self._step_timestamps.get((object_name, current_step), [time.time()])
            ),
            "t_publish": time.time(),
            "observation_count": self._step_observation_counts.get(
                (object_name, current_step), 0
            ),
            "active_sensors": sorted(
                {
                    obs.sensor_id
                    for obs in self._step_observations.get(
                        (object_name, current_step), []
                    )
                }
            ),
        }
        if covariance is not None:
            out["covariance"] = covariance.tolist()
        try:
            extrinsics = gtsam.extrinsic_estimates()
            if extrinsics:
                out["online_extrinsics"] = {
                    name: matrix.tolist() for name, matrix in extrinsics.items()
                }
                out["online_extrinsic_corrections"] = gtsam.extrinsic_corrections()
        except Exception:
            pass
        # Backward-compatible alias; new consumers should use t_publish.
        out["stamp"] = out["t_publish"]
        if bool(getattr(self.fusion_cfg, "enable_integrity_monitoring", True)):
            try:
                if consensus_report is not None:
                    out["integrity"] = {"enabled": True, **consensus_report}
                    out["integrity"]["trace_covariance"] = float(np.trace(covariance)) if covariance is not None else None
                else:
                    out["integrity"] = gtsam.integrity_report(current_step, nis_values)
                    out["integrity"]["sensor_health"] = sensor_health
            except Exception:
                pass
        if consensus_report is not None:
            out["candidate_count"] = consensus_report["candidate_count"]
            out["excluded_sensors"] = consensus_report["excluded_sensors"]
            # Preserve prediction provenance even when detailed integrity
            # reporting is disabled by a caller or an ablation.
            out["prediction_only"] = bool(consensus_report.get("prediction_only", False))
        await self.bus.publish(self.cfg.fusion.topic_fused, out)

    def _revise_world_observation(self, object_name: str, step: int, obs: Observation,
                                   sync_group: int | None) -> tuple[str, int]:
        system = self._get_system(object_name)
        state_key = (object_name, step)
        candidates = self._world_candidates.setdefault(state_key, {})
        candidates[obs.sensor_id] = obs  # duplicate source/state replaces, not counts twice
        observations = [candidates[sensor] for sensor in sorted(candidates)]
        timestamp = min(o.capture_time for o in observations if o.capture_time is not None)
        history = [item for item in self._object_pose_history.get(object_name, []) if item[1] < timestamp - 1e-6]
        history_guard = bool(getattr(self.fusion_cfg, "consensus_supported_history_only", False))
        excluded_history = 0
        if history_guard:
            # Consult final replaceable state reports, not provisional per-input
            # votes. A singleton must not certify its own prediction repeatedly.
            supported = []
            for item in history:
                report = self._consensus_reports.get((object_name, item[0]), {})
                if report.get("support_count", 0) >= 2 and not report.get("ambiguous", True):
                    supported.append(item)
            excluded_history = len(history) - len(supported)
            history = supported
        prediction, motion = None, None
        if history and bool(getattr(self.fusion_cfg, "enable_consensus_history_prediction", True)):
            previous, previous_time, previous_pose = history[-1]
            gap = timestamp - previous_time
            if 0 < gap <= self.fusion_cfg.max_prediction_gap_s:
                delta = gtsam.Pose3()
                if self.fusion_cfg.temporal_constant_velocity and len(history) >= 2:
                    past_dt = previous_time - history[-2][1]
                    if past_dt > 1e-6:
                        delta = gtsam.Pose3.Expmap(gtsam.Pose3.Logmap(history[-2][2].between(previous_pose)) * gap / past_dt)
                prediction = previous_pose.compose(delta)
                if self.fusion_cfg.enable_temporal_factors:
                    motion = (previous, delta, gap)
        covariances = [system.quality.pose_covariance(o) for o in observations]
        selection = select_pose_consensus(
            [o.measured_pose for o in observations], covariances,
            trans_gate_m=self.fusion_cfg.innovation_gate_trans_m,
            rot_gate_deg=self.fusion_cfg.innovation_gate_rot_deg,
            nis_threshold=system.monitor.chi2_threshold(6) if (
                self.fusion_cfg.enable_mahalanobis_gate and self.fusion_cfg.consensus_use_nis) else None,
            huber_k=self.fusion_cfg.robust_huber_k, prediction=prediction,
            qualities=([o.q_fit for o in observations]
                       if bool(getattr(self.fusion_cfg, "enable_consensus_quality_override", False))
                       else None),
            quality_override_ratio=float(getattr(
                self.fusion_cfg, "consensus_quality_override_ratio", 1.5)))
        selected = [observations[i] for i in selection.selected]
        # A history constraint is not another independent camera observation.
        # Use only past multi-view-confirmed states; prediction-only states
        # cannot refresh this evidence and prolong their own lifetime.
        guarded_motion = None
        guard_detail = None
        if (bool(getattr(self.fusion_cfg, "enable_singleton_history_guard", False))
                and self.fusion_cfg.enable_temporal_factors and len(selected) == 1):
            anchors = []
            for item in self._object_pose_history.get(object_name, []):
                report = self._consensus_reports.get((object_name, item[0]), {})
                stamps = self._step_timestamps.get((object_name, item[0]), [])
                # The latest input used by the anchor must also precede this
                # state. A late revision must never consult future evidence.
                if (item[0] != step and stamps and max(stamps) < timestamp - 1e-6
                        and report.get("support_count", 0) >= 2
                        and not report.get("ambiguous", True)):
                    anchors.append(item)
            if anchors:
                previous, previous_time, previous_pose = anchors[-1]
                age = timestamp - previous_time
                if 0 < age <= self.fusion_cfg.singleton_history_guard_max_age_s:
                    delta = gtsam.Pose3()
                    if self.fusion_cfg.temporal_constant_velocity and len(anchors) >= 2:
                        past_dt = previous_time - anchors[-2][1]
                        if 1e-6 < past_dt <= self.fusion_cfg.max_prediction_gap_s:
                            delta = gtsam.Pose3.Expmap(gtsam.Pose3.Logmap(
                                anchors[-2][2].between(previous_pose)) * age / past_dt)
                    guarded_prediction = previous_pose.compose(delta)
                    innovation = guarded_prediction.between(selection.pose)
                    distance = float(np.linalg.norm(innovation.translation()))
                    angle = float(np.linalg.norm(gtsam.Rot3.Logmap(innovation.rotation())))
                    growth = math.sqrt(max(age, 1. / 60.) * 60) * self.fusion_cfg.temporal_process_scale
                    trans_limit = self.fusion_cfg.innovation_gate_trans_m + 3*self.fusion_cfg.motion_trans_sigma*growth
                    rot_limit = math.radians(self.fusion_cfg.innovation_gate_rot_deg) + 3*self.fusion_cfg.motion_rot_sigma*growth
                    guard_detail = {"anchor_step": previous, "anchor_age_s": age,
                                    "innovation_m": distance, "innovation_deg": math.degrees(angle),
                                    "limit_m": trans_limit, "limit_deg": math.degrees(rot_limit),
                                    "prediction_only": distance > trans_limit or angle > rot_limit}
                    if guard_detail["prediction_only"]:
                        selected = []
                        selection.pose = guarded_prediction
                        guarded_motion = (previous, delta, age)
        # A mutually supported new hypothesis can restart the motion prior;
        # past tracker drift must not prevent future re-detection.
        reacquired = False
        if prediction is not None:
            innovation = prediction.between(selection.pose)
            if (np.linalg.norm(innovation.translation()) > self.fusion_cfg.innovation_gate_trans_m
                    or np.degrees(np.linalg.norm(gtsam.Rot3.Logmap(innovation.rotation()))) > self.fusion_cfg.innovation_gate_rot_deg):
                motion = None
                reacquired = len(selected) >= 2
        if guarded_motion is not None:
            motion = guarded_motion
            reacquired = False
        system.revise_world_state(step, selected, selection.pose, motion)
        self._step_observations[state_key] = selected
        self._step_observation_counts[state_key] = len(selected)
        self._step_timestamps[state_key] = [o.capture_time for o in observations if o.capture_time is not None]
        if sync_group is not None:
            self._step_sync_groups[state_key] = sync_group
        selected_ids = {o.sensor_id for o in selected}
        sensor_health = {}
        for o in observations:
            vote = (o.sensor_id in selected_ids) if len(selected) >= 2 and not selection.ambiguous else None
            sensor_health[o.sensor_id] = self._consistency_history.update(object_name, o.sensor_id, step, vote)
        calibrated = bool(selected) and all(o.covariance is not None or o.sensor_id in system.quality._calibrated_sigmas
                         or (o.detector or "") in system.quality._calibrated_sigmas for o in selected)
        sufficient = len(selected) >= 2 and not selection.ambiguous
        health = "Reliable" if sufficient and calibrated and not reacquired else "Degraded"
        if len(observations) >= 3 and not sufficient:
            health = "Unsafe"
        self._consensus_reports[state_key] = {
            "health": health, "candidate_count": len(observations), "support_count": len(selected),
            "excluded_sensors": sorted(set(candidates) - selected_ids), "ambiguous": selection.ambiguous,
            "evidence": ("prediction_only_recent_multiview" if guarded_motion is not None else
                         "multi_view_consistent" if sufficient else "single_view_or_conflicting"),
            "covariance_calibrated": calibrated, "accuracy_guaranteed": False,
            "sensor_health": sensor_health, "motion_prior_active": motion is not None,
            "reacquired": reacquired,
            "supported_history_only": history_guard,
            "excluded_history_states": excluded_history,
            "singleton_history_guard": guard_detail,
            "prediction_only": guarded_motion is not None,
        }
        return object_name, step

    def _load_calibration(self) -> None:
        """
        Load tag-cube calibration result and per-camera sensor modes.

        For each camera in the calibration result:
          - T_world_cam is the estimated world-to-camera transform.
          - This is loaded directly as known extrinsic for fixed sensors.
        """
        import json
        from pathlib import Path

        cfg = getattr(self.cfg, "calibration", None)
        if cfg is None:
            slog(
                component=self.__class__.__name__,
                event="calib",
                status="no_calibration_config",
                note="all sensors mobile",
            )
            return

        result_path = cfg.tag_cube.result_path
        if not result_path:
            return

        path = Path(result_path)
        if not path.exists():
            slog(
                component=self.__class__.__name__,
                event="calib",
                status="calib_file_not_found",
                path=str(path),
                note="all sensors mobile",
            )
            return

        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            slog(
                component=self.__class__.__name__,
                event="calib",
                status="calib_parse_error",
                error=str(exc),
            )
            return

        from core.coordinate_frames import transform_to_y_down

        coordinate_system = data.get(
            "coordinate_system", data.get("coord", "y_down")
        )
        T_world_cam: dict[str, np.ndarray] = {}
        for serial, T_mat in data.get("T_world_cam", {}).items():
            if T_mat is None:
                continue
            T = transform_to_y_down(
                np.array(T_mat, dtype=float), coordinate_system
            )
            R = gtsam.Rot3(T[:3, :3])
            t = gtsam.Point3(T[:3, 3])
            T_gtsam = gtsam.Pose3(R, t)
            T_world_cam[serial] = T_gtsam
            self._known_extrinsics[serial] = T_gtsam

        # Per-camera mode from config
        per_camera_mode = getattr(cfg, "per_camera_mode", {}) or {}
        for serial in T_world_cam:
            self._sensor_modes[serial] = per_camera_mode.get(serial, "fixed")

        slog(
            component=self.__class__.__name__,
            event="calib_loaded",
            status="ok",
            cameras=list(T_world_cam.keys()),
            modes=self._sensor_modes,
        )

    def get_sensor_mode(self, serial: str) -> str:
        """Return sensor type: fixed (known extrinsic) / unknown / mobile."""
        return self._sensor_modes.get(serial, "mobile")

    def get_extrinsic(self, serial: str) -> Any | None:
        """Return gtsam.Pose3 of T_world_camera, or None if not known."""
        return self._known_extrinsics.get(serial)

    # -------------------------------------------------------------------------
    # Public async entry point (matches existing backend_main signature)
    # -------------------------------------------------------------------------

    async def run(self):
        """
        Main async loop. Drains upstream queues and feeds GTSAM.
        Runs at the configured fusion rate (dt_fuse).
        """
        dt = self.fusion_cfg.dt_fuse

        slog(
            component=self.__class__.__name__,
            event="run",
            fusion_type="GTSAM_iSAM2_HeteroFusion",
            dt_fuse=dt,
            topic_observations=self._obs_topic,
            offline_replay=bool(self._offline_replay),
        )

        try:
            while True:
                t_loop_start = time.perf_counter()
                msgs: list[Any] = []
                queues = [
                    self._q_hl2,
                    self._q_ots,
                    self._q_tracking,
                    self._q_realsense,
                    *self._q_imu,
                ]

                if self._offline_replay:
                    # Replay/offline mode: drain as much as possible, process message-by-message.
                    for q in queues:
                        msgs.extend(self._drain_queue_nowait(q))

                    if not msgs:
                        get_tasks = [asyncio.create_task(q.get()) for q in queues]
                        try:
                            done, _pending = await asyncio.wait(
                                get_tasks,
                                timeout=max(dt, 0.01),
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                        finally:
                            for t in get_tasks:
                                if not t.done():
                                    t.cancel()
                        for t in done:
                            msgs.append(t.result())
                        for q in queues:
                            msgs.extend(self._drain_queue_nowait(q))

                    for msg in msgs:
                        if msg is None:
                            continue
                        if self._is_imu_message(msg):
                            await self._on_imu_message(msg)
                            continue
                        result = await self._on_pose_message(msg)
                        if result is None:
                            continue
                        obj_name, current_step = result
                        await self._update_and_publish(obj_name, current_step)
                else:
                    get_tasks = [asyncio.create_task(q.get()) for q in queues]
                    try:
                        done, _pending = await asyncio.wait(
                            get_tasks, timeout=dt, return_when=asyncio.FIRST_COMPLETED
                        )
                    finally:
                        for t in get_tasks:
                            if not t.done():
                                t.cancel()

                    for t in done:
                        msg = t.result()
                        if msg is None:
                            continue
                        if self._is_imu_message(msg):
                            await self._on_imu_message(msg)
                            continue
                        result = await self._on_pose_message(msg)
                        if result is None:
                            continue
                        obj_name, current_step = result
                        await self._update_and_publish(obj_name, current_step)

                loop_ms = (time.perf_counter() - t_loop_start) * 1000.0
                self._perf_loop_count += 1
                self._perf_loop_ms_sum += loop_ms

                if self._perf_loop_count % self._perf_interval == 0:
                    n = float(self._perf_interval)
                    obs_lag_ms_avg = (
                        self._perf_obs_sched_lag_ms_sum / float(self._perf_obs_count)
                        if self._perf_obs_count > 0
                        else 0.0
                    )
                    slog(
                        component=self.__class__.__name__,
                        event="perf",
                        samples=int(self._perf_interval),
                        fusion_loop_ms_avg=self._perf_loop_ms_sum / n,
                        obs_to_fusion_ms_avg=obs_lag_ms_avg,
                    )
                    self._perf_loop_ms_sum = 0.0
                    self._perf_obs_count = 0
                    self._perf_obs_sched_lag_ms_sum = 0.0
        finally:
            rem = self._perf_loop_count % self._perf_interval
            if rem > 0:
                n = float(rem)
                obs_lag_ms_avg = (
                    self._perf_obs_sched_lag_ms_sum / float(self._perf_obs_count)
                    if self._perf_obs_count > 0
                    else 0.0
                )
                slog(
                    component=self.__class__.__name__,
                    event="perf_final",
                    samples=int(rem),
                    fusion_loop_ms_avg=self._perf_loop_ms_sum / n,
                    obs_to_fusion_ms_avg=obs_lag_ms_avg,
                )

    # -------------------------------------------------------------------------
    # Bus message handler
    # -------------------------------------------------------------------------

    @staticmethod
    def _infer_serial_from_sensor_id(sensor_id: str) -> str:
        sid = str(sensor_id or "").strip()
        if not sid:
            return ""
        parts = [p for p in sid.split("/") if p]
        if not parts:
            return ""
        if parts[0] == "realsense" and len(parts) >= 2:
            return parts[1]
        if parts[0].startswith("realsense_"):
            return parts[0].replace("realsense_", "", 1)
        return parts[0]

    @staticmethod
    def _graph_sensor_id(sensor_id: str) -> str:
        """Strip detector/object suffixes so calibration is per physical sensor."""
        source = str(sensor_id or "")
        for marker in ("/foundationpose/", "/dope/", "/seg3d/", "/m3t/", "/megapose/"):
            if marker in source:
                return source.split(marker, 1)[0]
        return source

    @staticmethod
    def _is_imu_message(msg: Any) -> bool:
        if isinstance(msg, TimeStampedPacket):
            source = str(msg.source).lower()
            payload = msg.payload if isinstance(msg.payload, dict) else {}
            return "imu" in source or str(payload.get("stream", "")).startswith("imu")
        if isinstance(msg, dict):
            source = str(msg.get("source", "")).lower()
            return bool(
                msg.get("kind") == "imu"
                or "imu" in source
                or any(k in msg for k in ("gyro", "angular_velocity", "accel"))
            )
        return False

    async def _on_imu_message(self, msg: Any) -> None:
        """Cache optional IMU samples and preintegrated deltas.

        The recorder/replay path is not required to emit IMU packets.  When
        it does, pose messages consume the accumulated delta for the same
        sensor.  Raw gyroscope samples are integrated into a rotation-only
        delta; a producer may provide a full ``delta_pose`` when translation
        preintegration is available.
        """
        if isinstance(msg, TimeStampedPacket):
            source = str(msg.source)
            payload = msg.payload if isinstance(msg.payload, dict) else {}
            timestamp = float(msg.t_capture if msg.t_capture is not None else msg.t_host)
        elif isinstance(msg, dict):
            source = str(msg.get("source", "imu"))
            payload = msg
            timestamp = float(msg.get("t_capture", msg.get("t_host", time.time())))
        else:
            return
        stream = str(payload.get("stream", ""))
        sensor_id = source
        if "hl2" in source.lower():
            sensor_id = "hl2"
        elif "quest" in source.lower():
            sensor_id = "quest"
        state = self._imu_state.setdefault(
            sensor_id,
            {
                "last_t": None,
                "pending_delta": np.eye(4, dtype=np.float64),
                "omega_norm": 0.0,
                "accel_norm": 0.0,
            },
        )
        delta = payload.get("delta_pose", payload.get("T_delta", payload.get("imu_delta_pose")))
        if delta is not None:
            try:
                matrix = np.asarray(delta, dtype=np.float64)
                if matrix.shape == (4, 4) and np.isfinite(matrix).all():
                    state["pending_delta"] = state["pending_delta"] @ matrix
            except (TypeError, ValueError):
                pass
        vector = payload.get("gyro", payload.get("angular_velocity"))
        if vector is None and isinstance(payload.get("imu_payload"), dict):
            nested = payload["imu_payload"]
            vector = nested.get("gyro", nested.get("angular_velocity"))
        if vector is not None:
            try:
                gyro = np.asarray(vector, dtype=np.float64).reshape(-1)[:3]
                if gyro.size == 3 and np.isfinite(gyro).all():
                    state["omega_norm"] = float(np.linalg.norm(gyro))
                    last_t = state.get("last_t")
                    if last_t is not None and timestamp > float(last_t):
                        dt = min(timestamp - float(last_t), 0.25)
                        rot = gtsam.Rot3.Expmap(gyro * dt).matrix()
                        delta_matrix = np.eye(4, dtype=np.float64)
                        delta_matrix[:3, :3] = rot
                        state["pending_delta"] = state["pending_delta"] @ delta_matrix
            except (TypeError, ValueError):
                pass
        accel = payload.get("accel", payload.get("linear_acceleration"))
        if accel is not None:
            try:
                vector_accel = np.asarray(accel, dtype=np.float64).reshape(-1)[:3]
                if vector_accel.size == 3 and np.isfinite(vector_accel).all():
                    state["accel_norm"] = float(np.linalg.norm(vector_accel))
            except (TypeError, ValueError):
                pass
        state["last_t"] = timestamp
        state["stream"] = stream

    def _consume_imu(self, sensor_id: str) -> dict[str, Any]:
        candidates = [sensor_id]
        if "quest" in sensor_id.lower():
            candidates.append("quest")
        if "hl2" in sensor_id.lower():
            candidates.append("hl2")
        for candidate in candidates:
            state = self._imu_state.get(candidate)
            if state is None:
                continue
            delta = np.asarray(state.get("pending_delta", np.eye(4)), dtype=np.float64)
            state["pending_delta"] = np.eye(4, dtype=np.float64)
            return {
                "delta_pose": delta,
                "omega_norm": float(state.get("omega_norm", 0.0)),
                "accel_norm": float(state.get("accel_norm", 0.0)),
            }
        return {"delta_pose": None, "omega_norm": 0.0, "accel_norm": 0.0}

    async def _on_pose_message(self, msg: Any) -> tuple[str, int] | None:
        """
        Parse incoming DetectionResult or dict message and add to GTSAM graph.

        Fuses world-frame DOPE, FoundationPose, and segmentation-3D detections.
        AprilTag cube poses are recorded but excluded from fusion.

        Returns the object_name of the processed detection, or None if skipped.
        """
        # Detect format
        msg_t_host: float | None = None
        msg_capture_time: float | None = None
        msg_sync_group: int | None = None
        pose_covariance: np.ndarray | None = None
        imu_delta_pose_mat: np.ndarray | None = None
        imu_delta_covariance: np.ndarray | None = None
        sensor_type = "world"
        extrinsic = None
        detector_name: str | None = None

        if isinstance(msg, DetectionResult):
            sensor_id = str(msg.source)
            source_lower = sensor_id.lower()
            if not any(
                marker in source_lower
                for marker in ("dope", "foundationpose", "seg3d", "m3t", "megapose")
            ):
                return None
            object_name = msg.object_name or self._extract_object_name(sensor_id)
            if msg.pose_sensor is not None:
                pose_mat = np.array(msg.pose_sensor, dtype=float)
                sensor_type = str(msg.sensor_type or "fixed")
            else:
                pose_mat = np.array(msg.pose_world, dtype=float)
                sensor_type = str(msg.sensor_type or "world")
            msg_t_host = float(msg.t_host)
            msg_capture_time = float(
                msg.capture_time if msg.capture_time is not None else msg.t_host
            )
            msg_sync_group = msg.sync_group
            detector_name = msg.detector
            quality_metrics = dict(msg.quality_metrics or {})
            q_area = float(
                max(
                    quality_metrics.get("q_area", quality_metrics.get("score", msg.score)),
                    0.0,
                )
            )
            q_fit = float(
                np.exp(
                    -max(
                        float(
                            quality_metrics.get(
                                "reproj_error", msg.reproj_error or 0.0
                            )
                        ),
                        0.0,
                    )
                    / max(self.fusion_cfg.sigma_reproj, 1e-6)
                )
            )
            if "q_fit" in quality_metrics:
                q_fit = float(quality_metrics["q_fit"])
            elif quality_metrics.get("depth_inlier_fraction") is not None:
                q_fit = float(quality_metrics["depth_inlier_fraction"])
            q_time = float(quality_metrics.get("q_time", 1.0))
            serial = self._infer_serial_from_sensor_id(sensor_id)
            pose_covariance = msg.covariance
            imu_delta_pose_mat = msg.imu_delta_pose
            imu_delta_covariance = msg.imu_delta_covariance
            if msg.known_extrinsic is not None:
                ext_mat = np.asarray(msg.known_extrinsic, dtype=float)
                extrinsic = gtsam.Pose3(
                    gtsam.Rot3(ext_mat[:3, :3]), gtsam.Point3(ext_mat[:3, 3])
                )
            else:
                extrinsic = self.get_extrinsic(serial)

        elif isinstance(msg, dict):
            sensor_id = str(msg.get("source", "unknown"))

            source_lower = sensor_id.lower()
            if not any(
                marker in source_lower
                for marker in ("dope", "foundationpose", "seg3d", "m3t", "megapose")
            ):
                return None
            object_name = str(msg.get("object_name") or self._extract_object_name(sensor_id))

            if msg.get("pose_sensor") is not None or msg.get("T_sensor_object") is not None:
                pose_mat = np.array(
                    msg.get("pose_sensor") if msg.get("pose_sensor") is not None else msg["T_sensor_object"], dtype=float
                )
                sensor_type = str(msg.get("sensor_type") or "fixed")
            elif "T" in msg:
                pose_mat = np.array(msg["T"], dtype=float)
            elif "pose_world" in msg:
                pose_mat = np.array(msg["pose_world"], dtype=float)
            else:
                slog(
                    component=self.__class__.__name__,
                    event="bad_pose_message",
                    reason="missing_T_or_pose_world",
                    keys=list(msg.keys()),
                )
                return None
            raw_t_host = msg.get("t_host")
            if raw_t_host is not None:
                try:
                    msg_t_host = float(raw_t_host)
                except (TypeError, ValueError):
                    msg_t_host = None
            raw_capture_time = msg.get("t_capture", msg.get("capture_time"))
            if raw_capture_time is not None:
                try:
                    msg_capture_time = float(raw_capture_time)
                except (TypeError, ValueError):
                    msg_capture_time = msg_t_host
            else:
                msg_capture_time = msg_t_host
            raw_sync_group = msg.get("sync_group")
            if raw_sync_group is not None:
                try:
                    msg_sync_group = int(raw_sync_group)
                except (TypeError, ValueError):
                    msg_sync_group = None
            qm = msg.get("quality_metrics", {}) or {}

            score_raw = qm.get("score", msg.get("score", qm.get("num_features", 1.0)))
            q_area = float(max(score_raw if score_raw is not None else 1.0, 0.0))

            reproj_raw = qm.get("reproj_error", msg.get("reproj_error", 0.0))
            if reproj_raw is None:
                reproj_raw = 0.0

            q_fit = float(
                np.exp(
                    -max(float(reproj_raw), 0.0)
                    / max(self.fusion_cfg.sigma_reproj, 1e-6)
                )
            )

            if "q_fit" in qm:
                q_fit = float(qm["q_fit"])
            elif qm.get("depth_inlier_fraction") is not None:
                q_fit = float(qm["depth_inlier_fraction"])

            q_time = float(qm.get("q_time", msg.get("q_time", 1.0)))
            detector_name = str(msg.get("detector") or "") or None

            serial = self._infer_serial_from_sensor_id(sensor_id)
            sensor_type = str(msg.get("sensor_type") or sensor_type)
            ext_data = msg.get("known_extrinsic")
            extrinsic = self.get_extrinsic(serial)
            if ext_data is not None:
                ext_mat = np.array(ext_data, dtype=float)
                R = gtsam.Rot3(ext_mat[:3, :3])
                t = gtsam.Point3(ext_mat[:3, 3])
                extrinsic = gtsam.Pose3(R, t)
            covariance_data = msg.get("covariance", qm.get("covariance"))
            if covariance_data is not None:
                try:
                    pose_covariance = np.asarray(covariance_data, dtype=float)
                except (TypeError, ValueError):
                    pose_covariance = None
            imu_data = msg.get("imu_delta_pose", msg.get("T_imu_delta"))
            if imu_data is not None:
                try:
                    imu_delta_pose_mat = np.asarray(imu_data, dtype=float)
                except (TypeError, ValueError):
                    imu_delta_pose_mat = None
            imu_cov_data = msg.get("imu_delta_covariance")
            if imu_cov_data is not None:
                try:
                    imu_delta_covariance = np.asarray(imu_cov_data, dtype=float)
                except (TypeError, ValueError):
                    imu_delta_covariance = None

        else:
            return None

        graph_sensor_id = self._graph_sensor_id(sensor_id)

        # Get or create per-object fusion system
        gtsam_sys = self._get_system(object_name)
        group_time = (
            msg_capture_time
            if bool(getattr(self.fusion_cfg, "enable_time_compensation", True))
            else msg_t_host
        )
        obj_step = self._assign_step(
            object_name, graph_sensor_id, group_time, sync_group=msg_sync_group
        )
        if obj_step is None:
            slog(
                component=self.__class__.__name__,
                event="stale_observation_rejected",
                object_name=object_name,
                sensor_id=sensor_id,
                t_host=msg_t_host,
            )
            return None

        if msg_t_host is not None:
            t_now_wall = time.time()
            if self._perf_t0_wall is None or self._perf_t0_host is None:
                self._perf_t0_wall = t_now_wall
                self._perf_t0_host = msg_t_host

            expected_wall = self._perf_t0_wall + (msg_t_host - self._perf_t0_host)
            sched_lag_ms = (t_now_wall - expected_wall) * 1000.0
            self._perf_obs_count += 1
            self._perf_obs_sched_lag_ms_sum += sched_lag_ms

        if sensor_type != "world":
            if sensor_type == "unknown" and not bool(
                getattr(self.fusion_cfg, "enable_online_calibration", True)
            ):
                sensor_type = "fixed"
                if extrinsic is None:
                    extrinsic = gtsam.Pose3()

            if sensor_type == "fixed" and extrinsic is None:
                sensor_type = "unknown"

        self._last_obs_debug = {
            "sensor_id": sensor_id,
            "object_name": object_name,
            "serial": serial,
            "sensor_type": sensor_type,
            "step": int(obj_step),
        }

        # Convert to gtsam Pose3
        # RGB-D poses and text extrinsics have small roundoff errors. Rot3's
        # matrix constructor does not project onto SO(3); repeated predictions
        # can amplify those errors until the graph becomes singular.
        rotation = np.asarray(pose_mat[:3, :3], dtype=np.float64)
        u, singular_values, vt = np.linalg.svd(rotation)
        if np.linalg.det(rotation) <= 0 or np.max(np.abs(singular_values - 1.0)) > 1e-2:
            raise ValueError(f"Invalid pose rotation from {sensor_id}: not a rigid transform")
        R_mat = gtsam.Rot3(u @ vt)
        t_vec = gtsam.Point3(pose_mat[:3, 3])
        measured_pose = gtsam.Pose3(R_mat, t_vec)
        imu_info = self._consume_imu(sensor_id)
        if imu_delta_pose_mat is None and imu_info.get("delta_pose") is not None:
            candidate_delta = np.asarray(imu_info["delta_pose"], dtype=np.float64)
            if candidate_delta.shape == (4, 4) and not np.allclose(
                candidate_delta, np.eye(4), atol=1e-12
            ):
                imu_delta_pose_mat = candidate_delta

        if "foundationpose" in sensor_id.lower():
            rot_sigma = float(self.fusion_cfg.foundationpose_rot_sigma)
            trans_sigma = float(self.fusion_cfg.foundationpose_trans_sigma)
        elif "megapose" in sensor_id.lower():
            rot_sigma = float(self.fusion_cfg.megapose_rot_sigma)
            trans_sigma = float(self.fusion_cfg.megapose_trans_sigma)
        elif "m3t" in sensor_id.lower():
            rot_sigma = float(self.fusion_cfg.m3t_rot_sigma)
            trans_sigma = float(self.fusion_cfg.m3t_trans_sigma)
        elif "seg3d" in sensor_id.lower():
            rot_sigma = float(self.fusion_cfg.seg3d_rot_sigma)
            trans_sigma = float(self.fusion_cfg.seg3d_trans_sigma)
        else:
            rot_sigma = float(self.fusion_cfg.dope_rot_sigma)
            trans_sigma = float(self.fusion_cfg.dope_trans_sigma)
        sensor_history_key = (object_name, graph_sensor_id)
        last_capture = self._last_sensor_capture.get(sensor_history_key)
        delta_t = (
            max(float(msg_capture_time - last_capture), 0.0)
            if msg_capture_time is not None and last_capture is not None
            else 0.0
        )
        if msg_t_host is not None and msg_capture_time is not None:
            latency_s = abs(float(msg_t_host) - float(msg_capture_time))
            q_time = min(
                q_time,
                math.exp(
                    -latency_s
                    / max(float(self.fusion_cfg.sync_window_s), 1e-3)
                ),
            )
        obs = Observation(
            time_k=obj_step,
            sensor_id=graph_sensor_id,
            sensor_type=sensor_type,
            measured_pose=measured_pose,
            q_area=q_area,
            q_fit=q_fit,
            q_time=q_time,
            known_extrinsic=extrinsic,
            rot_sigma=rot_sigma,
            trans_sigma=trans_sigma,
            delta_t=delta_t,
            t_host=msg_t_host,
            capture_time=msg_capture_time,
            covariance=pose_covariance,
            detector=detector_name,
            imu_delta_covariance=imu_delta_covariance,
            omega_norm=float(imu_info.get("omega_norm", 0.0)),
            v_norm=0.0,
        )
        if sensor_type == "world" and self.fusion_cfg.enable_revisable_consensus:
            if obs.capture_time is None:
                obs.capture_time = time.time()
            return self._revise_world_observation(object_name, obj_step, obs, msg_sync_group)
        if (
            bool(getattr(self.fusion_cfg, "enable_sensor_fault_isolation", True))
            and self._sensor_health.get(graph_sensor_id) == "Unsafe"
        ):
            slog(
                component=self.__class__.__name__,
                event="sensor_observation_isolated",
                sensor_id=graph_sensor_id,
                object_name=object_name,
                reason="integrity_monitor_unsafe",
            )
            return None
        if imu_delta_pose_mat is not None and imu_delta_pose_mat.shape == (4, 4):
            obs.imu_delta_pose = gtsam.Pose3(
                gtsam.Rot3(imu_delta_pose_mat[:3, :3]),
                gtsam.Point3(imu_delta_pose_mat[:3, 3]),
            )
        effective_sigma = float(gtsam_sys.quality.pose_sigmas(obs)[3])

        # Reject a contradictory second-camera measurement for the same time
        # state. The first measurement remains protected by the robust factor.
        current = gtsam_sys.current_values()
        state_key = gtsam_sys.tool_key(obj_step)
        if current.exists(state_key) and self._step_observation_counts.get(
            (object_name, obj_step), 0
        ) > 0:
            predicted = current.atPose3(state_key)
            measured_world = measured_pose
            if sensor_type in ("fixed", "unknown") and extrinsic is not None:
                measured_world = extrinsic.compose(measured_pose)
            elif sensor_type == "mobile":
                cam_key = gtsam_sys.cam_key(graph_sensor_id, obj_step)
                if current.exists(cam_key):
                    measured_world = current.atPose3(cam_key).compose(measured_pose)
            delta = predicted.between(measured_world)
            trans_delta = float(np.linalg.norm(delta.translation()))
            rot_delta_deg = float(
                np.degrees(np.linalg.norm(gtsam.Rot3.Logmap(delta.rotation())))
            )
            nis_bad = False
            nis_value = None
            if bool(getattr(self.fusion_cfg, "enable_mahalanobis_gate", True)):
                try:
                    sigma_x = gtsam_sys.marginal_covariance(state_key)
                    residual = np.asarray(gtsam.Pose3.Logmap(delta)).reshape(6, 1)
                    nis_value = gtsam_sys.monitor.nis(
                        residual, sigma_x + gtsam_sys.quality.pose_covariance(obs)
                    )
                    nis_bad = nis_value > gtsam_sys.monitor.chi2_threshold(6)
                except Exception:
                    nis_bad = False
            if (
                trans_delta > float(self.fusion_cfg.innovation_gate_trans_m)
                or rot_delta_deg > float(self.fusion_cfg.innovation_gate_rot_deg)
                or nis_bad
            ):
                previous_sigma = self._best_state_sigma.get(
                    (object_name, obj_step), float("inf")
                )
                if effective_sigma >= previous_sigma:
                    slog(
                        component=self.__class__.__name__,
                        event="innovation_rejected",
                        sensor_id=sensor_id,
                        object_name=object_name,
                        translation_m=trans_delta,
                        rotation_deg=rot_delta_deg,
                        effective_sigma=effective_sigma,
                        best_sigma=previous_sigma,
                        nis=nis_value,
                    )
                    return None

        # Add one temporal state-transition factor for each new object state.
        # The previous implementation only added detection-derived motion for
        # some sensor types and skipped world-frame observations entirely.  A
        # constant-velocity/hold prediction is now used for every source; an
        # optional IMU delta replaces the mobile-camera hold prior.
        previous_step = self._last_object_step.get(object_name)
        if (
            bool(getattr(self.fusion_cfg, "enable_temporal_factors", True))
            and previous_step is not None
            and obj_step > previous_step
            and (object_name, previous_step, obj_step) not in self._temporal_added
        ):
            delta_seconds = (
                max(float(msg_capture_time - self._last_object_time[object_name]), 0.0)
                if msg_capture_time is not None
                and object_name in self._last_object_time
                else 1.0 / 60.0
            )
            rel_tool = gtsam.Pose3()
            if bool(getattr(self.fusion_cfg, "temporal_constant_velocity", True)):
                history = self._object_pose_history.get(object_name, [])
                if len(history) >= 2:
                    history_dt = history[-1][1] - history[-2][1]
                    if history_dt > 1e-6:
                        relative = history[-2][2].between(history[-1][2])
                        # Reconstruct on SE(3), rather than recursively copying
                        # near-rotation matrices into new motion factors.
                        tangent = gtsam.Pose3.Logmap(relative)
                        rel_tool = gtsam.Pose3.Expmap(tangent * (delta_seconds / history_dt))
            if 0 < delta_seconds <= float(self.fusion_cfg.max_prediction_gap_s):
                gtsam_sys.add_tool_motion(
                    previous_step,
                    obj_step,
                    rel_tool,
                    delta_t=delta_seconds,
                )
                self._temporal_added.add((object_name, previous_step, obj_step))

        last_step = self._last_sensor_step.get(sensor_history_key)
        if sensor_type == "mobile" and last_step is not None and obj_step > last_step:
            imu_rel = obs.imu_delta_pose if obs.imu_delta_pose is not None else gtsam.Pose3()
            gtsam_sys.add_mobile_cam_motion(
                graph_sensor_id,
                last_step,
                obj_step,
                imu_rel,
                delta_t=delta_t,
                covariance=imu_delta_covariance,
            )

        self._last_sensor_step[sensor_history_key] = obj_step
        self._last_sensor_T[sensor_history_key] = measured_pose
        if msg_capture_time is not None:
            self._last_sensor_capture[sensor_history_key] = msg_capture_time

        # Add observation factor
        try:
            gtsam_sys.add_observation(obs)
        except Exception as exc:
            slog(
                component=self.__class__.__name__,
                event="add_observation_error",
                sensor_id=sensor_id,
                object_name=object_name,
                error=str(exc),
            )
            return None

        step_key = (object_name, obj_step)
        if msg_capture_time is not None:
            self._step_timestamps.setdefault(step_key, []).append(msg_capture_time)
        if msg_sync_group is not None:
            self._step_sync_groups[step_key] = msg_sync_group
        self._step_observation_counts[step_key] = (
            self._step_observation_counts.get(step_key, 0) + 1
        )
        self._step_observations.setdefault(step_key, []).append(obs)
        if previous_step is None or obj_step >= previous_step:
            self._last_object_step[object_name] = obj_step
            if msg_capture_time is not None:
                self._last_object_time[object_name] = msg_capture_time

        state_sigma_key = (object_name, obj_step)
        self._best_state_sigma[state_sigma_key] = min(
            effective_sigma,
            self._best_state_sigma.get(state_sigma_key, float("inf")),
        )

        return object_name, obj_step
