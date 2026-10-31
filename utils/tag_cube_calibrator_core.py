# utils/tag_cube_calibrator_core.py
# pyright: basic
# utils.tag_cube_calibrator_core
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Standalone calibration math — no RealSense or OpenCV SDK dependencies.
# Shared by the CLI calibrator and any downstream tools.

from __future__ import annotations

import json
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Cube geometry
# ---------------------------------------------------------------------------

_CUBE_HALF: float = 0.025  # 50 mm → half = 0.025 m

# Tag-centre offsets in the cube's body frame (metres).
# Convention: body origin = cube centre.
# Confirmed layout (face-pair axis alignment):
#   face 0 = +X,  face 3 = -X   (0-3 pair)
#   face 5 = +Y,  face 1 = -Y   (1-5 pair)
#   face 4 = +Z,  face 2 = -Z   (2-4 pair)
_TAG_OFFSETS: dict[int, np.ndarray] = {
    0: np.array([_CUBE_HALF, 0.0, 0.0], dtype=np.float64),  # +X
    1: np.array([0.0, -_CUBE_HALF, 0.0], dtype=np.float64),  # -Y
    2: np.array([0.0, 0.0, -_CUBE_HALF], dtype=np.float64),  # -Z
    3: np.array([-_CUBE_HALF, 0.0, 0.0], dtype=np.float64),  # -X
    4: np.array([0.0, 0.0, _CUBE_HALF], dtype=np.float64),  # +Z
    5: np.array([0.0, _CUBE_HALF, 0.0], dtype=np.float64),  # +Y
}

# Per-face in-plane basis (u, v) in cube frame.
# Corner order follows OpenCV marker convention used in this project:
#   0: (-s, +s), 1: (+s, +s), 2: (+s, -s), 3: (-s, -s)
# where point = center + x*u + y*v.
_TAG_FACE_BASIS: dict[int, tuple[np.ndarray, np.ndarray]] = {
    0: (
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ),  # +X
    1: (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ),  # -Y
    2: (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, -1.0, 0.0], dtype=np.float64),
    ),  # -Z
    3: (
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
    ),  # -X
    4: (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    ),  # +Z
    5: (
        np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ),  # +Y
}


def _tag_object_corners(tag_id: int, half: float) -> Optional[np.ndarray]:
    """Return 4x3 object points for one face in cube frame."""
    center = _TAG_OFFSETS.get(tag_id)
    basis = _TAG_FACE_BASIS.get(tag_id)
    if center is None or basis is None:
        return None

    u, v = basis
    return np.array(
        [
            center + (-half) * u + (half) * v,
            center + (half) * u + (half) * v,
            center + (half) * u + (-half) * v,
            center + (-half) * u + (-half) * v,
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def rotvec_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues vector (3,) → 3×3 rotation matrix."""
    import cv2

    rvec_arr = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    return cv2.Rodrigues(rvec_arr)[0]


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → axis-angle vector (3,)."""
    import cv2

    R_arr = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return cv2.Rodrigues(R_arr)[0].ravel()


def skew(v: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric matrix."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """
    se(3) → SE(3).  xi = [omega(3), v(3)].
    """
    omega = xi[:3]
    v = xi[3:]
    theta = float(np.linalg.norm(omega))
    if theta < 1e-8:
        rot = np.eye(3)
        trans = v.copy()
    else:
        k = omega / theta
        K = skew(k)
        rot = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
        trans = (
            np.eye(3)
            + (1 - math.cos(theta)) / theta * K
            + (1 - math.sin(theta) / theta) * (K @ K)
        ) @ v
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def T_to_components(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4×4 SE(3) → (R 3×3, t 3,)"""
    return T[:3, :3].copy(), T[:3, 3].copy()


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(R, t) → 4×4 SE(3) matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# Rotation averaging
# ---------------------------------------------------------------------------


def average_rotation(Rs: list[np.ndarray]) -> np.ndarray:
    """
    Iterative axis-angle averaging for a set of 3×3 rotation matrices.
    """
    R_mean = Rs[0].copy()
    for _ in range(20):
        log_sum = np.zeros(3)
        for R in Rs:
            log_sum += matrix_to_rotvec(R @ R_mean.T)
        delta = log_sum / len(Rs)
        R_mean = se3_exp(np.concatenate([delta, np.zeros(3)]))[:3, :3] @ R_mean
    return R_mean


def average_translation(ts: list[np.ndarray]) -> np.ndarray:
    """Arithmetic mean of translation vectors."""
    return np.mean(ts, axis=0)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass
class CubeObservation:
    """
    One frame: camera observes the cube, OptiTrack reports the cube in world.

    T_cam_cube  : camera frame → cube body frame   (from solvePnP)
    T_world_cube: world frame → cube body frame     (from OptiTrack)
    Derived: T_world_cam = T_world_cube @ inv(T_cam_cube)
    """

    camera_serial: str
    t_host: float
    T_cam_cube: np.ndarray  # 4×4 SE(3)
    T_world_cube: np.ndarray  # 4×4 SE(3) — world → cube
    visible_ids: list[int]
    reproj_error_px: float


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


class TagCubeCalibrator:
    """
    Robust extrinsic calibration using a static TAG16H5 cube observed
    by one or more RealSense cameras simultaneously.

    World frame is defined by OptiTrack (or identity if OTS is disabled).
    Each camera independently computes T_world_cam via:
      T_world_cam = mean(T_world_cube) @ inv(mean(T_cam_cube))
    """

    def __init__(
        self,
        cube_size_mm: float = 50.0,
        min_frames: int = 20,
        min_visible_faces: int = 2,
        consensus_thresh_mm: float = 10.0,
        max_reproj_error_px: float = 3.0,
    ):
        self.cube_size_mm = cube_size_mm
        self._half = cube_size_mm / 2000.0  # mm → m, half
        self.min_frames = min_frames
        self.min_visible_faces = min_visible_faces
        self._thresh_mm = consensus_thresh_mm
        self.max_reproj_error_px = max_reproj_error_px

        # Per-camera observations
        self._obs: dict[str, list[CubeObservation]] = {}
        self._stats: dict[str, dict[str, int]] = {}
        self._last_reason: dict[str, str] = {}

    def _init_stats(self, camera_serial: str) -> dict[str, int]:
        stats = self._stats.setdefault(
            camera_serial,
            {
                "received": 0,
                "accepted": 0,
                "reject_too_few_faces": 0,
                "reject_no_valid_ids": 0,
                "reject_solvepnp": 0,
                "reject_non_finite_pose": 0,
                "reject_non_finite_world": 0,
                "reject_non_finite_reproj": 0,
                "reject_high_reproj": 0,
                "pnp_rotation_fallback": 0,
            },
        )
        self._last_reason.setdefault(camera_serial, "idle")
        return stats

    def register_camera(self, camera_serial: str) -> None:
        self._init_stats(camera_serial)

    def _set_reason(self, camera_serial: str, reason: str) -> None:
        self._last_reason[camera_serial] = reason

    def get_last_reason(self, camera_serial: str) -> str:
        return self._last_reason.get(camera_serial, "idle")

    @property
    def cameras(self) -> list[str]:
        cams = list(self._stats.keys())
        for serial in self._obs.keys():
            if serial not in self._stats:
                cams.append(serial)
        return cams

    def add_frame(
        self,
        camera_serial: str,
        t_host: float,
        corners_per_id: dict[int, np.ndarray],
        K: np.ndarray,
        dist: np.ndarray,
        T_world_cube: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Process one frame from one camera.

        Returns True if the frame contributed a valid observation.
        """
        import cv2

        stats = self._init_stats(camera_serial)
        stats["received"] += 1

        if len(corners_per_id) < self.min_visible_faces:
            stats["reject_too_few_faces"] += 1
            self._set_reason(camera_serial, "rej_too_few_faces")
            return False

        half = self._half
        face_entries: list[tuple[int, np.ndarray, np.ndarray]] = []

        for tag_id, corners in corners_per_id.items():
            objp = _tag_object_corners(tag_id, half)
            if objp is None:
                continue
            imgp = corners.reshape(4, 2).astype(np.float64)
            face_entries.append((tag_id, objp, imgp))

        if len(face_entries) < self.min_visible_faces:
            stats["reject_no_valid_ids"] += 1
            self._set_reason(camera_serial, "rej_no_valid_ids")
            return False

        def _compute_reproj(
            objpts_local: np.ndarray,
            imgpts_local: np.ndarray,
            rvec_local: np.ndarray,
            tvec_local: np.ndarray,
        ) -> float:
            projected, _ = cv2.projectPoints(
                objpts_local, rvec_local, tvec_local, K, dist
            )
            projected_2d = projected.reshape(-1, 2)
            imgpts_2d = imgpts_local.reshape(-1, 2)
            err = np.linalg.norm(projected_2d - imgpts_2d, axis=1)
            return float(np.mean(err))

        def _try_solve_entries(
            solve_entries: list[tuple[int, np.ndarray, np.ndarray]],
            rotations: tuple[int, ...],
        ):
            objpts_all: list[np.ndarray] = []
            imgpts_all: list[np.ndarray] = []
            ids_used: list[int] = []
            for (tag_id, objp, imgp), r in zip(solve_entries, rotations):
                objp_rot = np.roll(objp, -int(r), axis=0) if r else objp
                objpts_all.append(objp_rot)
                imgpts_all.append(imgp)
                ids_used.append(tag_id)

            objpts_local = np.vstack(objpts_all)
            imgpts_local = np.vstack(imgpts_all)

            try:
                ok_local, rvec_local, tvec_local, _ = cv2.solvePnPRansac(
                    objpts_local,
                    imgpts_local,
                    K,
                    dist,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    confidence=0.99,
                    reprojectionError=2.0,
                )
            except Exception:
                return None

            if (not ok_local) or rvec_local is None or tvec_local is None:
                return None

            reproj_local = _compute_reproj(
                objpts_local, imgpts_local, rvec_local, tvec_local
            )
            if not np.isfinite(reproj_local):
                return None

            return (
                objpts_local,
                imgpts_local,
                rvec_local,
                tvec_local,
                ids_used,
                reproj_local,
            )

        # Robust subset search:
        # - prefer 3-face subsets (more constrained)
        # - fallback to 2-face subsets (avoid one bad face poisoning the solve)
        n_all = len(face_entries)
        subset_sizes: list[int] = []
        if n_all >= 3:
            subset_sizes.append(3)
        if n_all >= 2:
            subset_sizes.append(2)

        best_result = None
        best_reproj = float("inf")
        used_fallback = False

        for k_subset in subset_sizes:
            for subset_idx in itertools.combinations(range(n_all), k_subset):
                solve_entries = [face_entries[i] for i in subset_idx]

                # 1) canonical corner order
                zero_rot = tuple(0 for _ in range(k_subset))
                cand0 = _try_solve_entries(solve_entries, zero_rot)
                if cand0 is not None:
                    reproj0 = float(cand0[5])
                    if reproj0 < best_reproj:
                        best_reproj = reproj0
                        best_result = cand0
                        used_fallback = False

                    # Good enough => no need to test all rotations for this subset.
                    if reproj0 <= self.max_reproj_error_px:
                        continue

                # 2) rotation fallback: per-face 0/90/180/270 in-plane rotations
                for rot_combo in itertools.product(range(4), repeat=k_subset):
                    if rot_combo == zero_rot:
                        continue
                    cand = _try_solve_entries(solve_entries, rot_combo)
                    if cand is None:
                        continue
                    reproj_c = float(cand[5])
                    if reproj_c < best_reproj:
                        best_reproj = reproj_c
                        best_result = cand
                        used_fallback = True

        if best_result is None:
            stats["reject_solvepnp"] += 1
            self._set_reason(camera_serial, "rej_solvepnp")
            return False

        objpts, imgpts, rvec, tvec, ids_valid, reproj = best_result
        if used_fallback:
            stats["pnp_rotation_fallback"] += 1

        if (not np.isfinite(rvec).all()) or (not np.isfinite(tvec).all()):
            stats["reject_non_finite_pose"] += 1
            self._set_reason(camera_serial, "rej_non_finite_pose")
            return False

        R = rotvec_to_matrix(rvec)
        if not np.isfinite(R).all():
            stats["reject_non_finite_pose"] += 1
            self._set_reason(camera_serial, "rej_non_finite_pose")
            return False

        T_cam_cube = make_T(R, tvec.ravel())
        if not np.isfinite(T_cam_cube).all():
            stats["reject_non_finite_pose"] += 1
            self._set_reason(camera_serial, "rej_non_finite_pose")
            return False

        # Default: cube at world origin
        if T_world_cube is None:
            T_world_cube = np.eye(4, dtype=np.float64)
        elif not np.isfinite(T_world_cube).all():
            stats["reject_non_finite_world"] += 1
            self._set_reason(camera_serial, "rej_non_finite_world")
            return False

        projected, _ = cv2.projectPoints(objpts, rvec, tvec, K, dist)
        projected_2d = projected.reshape(-1, 2)
        imgpts_2d = imgpts.reshape(-1, 2)
        per_point_err = np.linalg.norm(projected_2d - imgpts_2d, axis=1)
        reproj = float(np.mean(per_point_err))
        if not np.isfinite(reproj):
            stats["reject_non_finite_reproj"] += 1
            self._set_reason(camera_serial, "rej_non_finite_reproj")
            return False
        if reproj > self.max_reproj_error_px:
            stats["reject_high_reproj"] += 1
            self._set_reason(camera_serial, "rej_high_reproj")
            return False

        obs = CubeObservation(
            camera_serial=camera_serial,
            t_host=t_host,
            T_cam_cube=T_cam_cube,
            T_world_cube=T_world_cube.copy(),
            visible_ids=ids_valid,
            reproj_error_px=reproj,
        )
        self._obs.setdefault(camera_serial, []).append(obs)
        stats["accepted"] += 1
        self._set_reason(camera_serial, "ok")
        return True

    def estimate_world_cam(self, camera_serial: str) -> Optional[np.ndarray]:
        """
        Estimate T_world_cam for one camera.

        T_world_cam = mean(T_world_cube) @ inv(mean(T_cam_cube))
        """
        obs_list = self._obs.get(camera_serial, [])
        if len(obs_list) < 3:
            return None

        # Average T_cam_cube
        Rs = [o.T_cam_cube[:3, :3] for o in obs_list]
        ts = [o.T_cam_cube[:3, 3] for o in obs_list]
        R_cam_cube = average_rotation(Rs)
        t_cam_cube = average_translation(ts)
        T_cam_cube = make_T(R_cam_cube, t_cam_cube)

        # Average T_world_cube
        Rs_w = [o.T_world_cube[:3, :3] for o in obs_list]
        ts_w = [o.T_world_cube[:3, 3] for o in obs_list]
        R_world_cube = average_rotation(Rs_w)
        t_world_cube = average_translation(ts_w)
        T_world_cube = make_T(R_world_cube, t_world_cube)

        # Compose
        T_world_cam = T_world_cube @ np.linalg.inv(T_cam_cube)
        return T_world_cam

    def estimate_all(self) -> dict[str, Optional[np.ndarray]]:
        """Compute T_world_cam for every camera."""
        return {serial: self.estimate_world_cam(serial) for serial in self.cameras}

    def calibration_report(self) -> dict[str, Any]:
        """Full report with ndarrays (keep as ndarray for in-memory use)."""
        T_world_cam = self.estimate_all()
        return {
            "cube_size_mm": self.cube_size_mm,
            "n_frames_per_camera": {
                s: self._stats.get(s, {}).get("accepted", 0) for s in self.cameras
            },
            "frame_stats": self.frame_stats(),
            "T_world_cam": T_world_cam,
        }

    def frame_stats(self) -> dict[str, dict[str, int]]:
        return {s: dict(v) for s, v in self._stats.items()}

    @staticmethod
    def _T_to_list(T: Optional[np.ndarray]) -> Optional[list]:
        if T is None:
            return None
        return T.tolist()

    def save(self, path: str | Path) -> None:
        """Save calibration to JSON (converts ndarrays to lists)."""
        path = Path(path)
        report = self.calibration_report()
        save_data = {
            "cube_size_mm": report["cube_size_mm"],
            "n_frames_per_camera": report["n_frames_per_camera"],
            "frame_stats": report["frame_stats"],
            "T_world_cam": {
                s: self._T_to_list(T) for s, T in report["T_world_cam"].items()
            },
        }
        path.write_text(json.dumps(save_data, indent=2))
        print(f"[*] Calibration saved → {path}")
