#!/usr/bin/env python3
"""
Replay AprilTag cube observations and estimate camera extrinsics.

Coordinate convention
---------------------
All transforms use column vectors:
    X_target = T_target_source @ X_source_h

For every accepted frame, solvePnP estimates:
    T_cam_cube: cube/tag-object coordinates -> camera coordinates

For a static cube and static cameras, the final camera pose w.r.t. the cube is:
    T_cube_cam = inv(T_cam_cube)

If you define the cube frame as the world frame, then T_world_cam == T_cube_cam.

Default cube layout
-------------------
The default face layout matches the user's provided table:
    tag 0: +X face, center [+half, 0, 0]
    tag 1: -Y face, center [0, -half, 0]
    tag 2: -Z face, center [0, 0, -half]
    tag 3: -X face, center [-half, 0, 0]
    tag 4: +Z face, center [0, 0, +half]
    tag 5: +Y face, center [0, +half, 0]

The in-plane basis vectors define the physical orientation of each printed tag on the cube.
If the top direction of each tag is unknown, keep --auto-corner-rolls enabled and inspect
"corner_roll_histogram" in the output. For production use, replace the automatic rolls with
fixed rolls using --fixed-corner-rolls once the correct values are known.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np


# In-plane basis vectors for each tag face.
# For a face center c, object corners are generated as:
#   c + (-tag_half) * u + (+tag_half) * v
#   c + (+tag_half) * u + (+tag_half) * v
#   c + (+tag_half) * u + (-tag_half) * v
#   c + (-tag_half) * u + (-tag_half) * v
# followed by an optional cyclic roll.
# These match the original script's basis assumptions.
_DEFAULT_TAG_FACE_BASIS: dict[int, tuple[np.ndarray, np.ndarray]] = {
    0: (np.array([0.0, 1.0, 0.0], dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64)),   # +X
    1: (np.array([1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64)),   # -Y
    2: (np.array([1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, -1.0, 0.0], dtype=np.float64)),  # -Z
    3: (np.array([0.0, 1.0, 0.0], dtype=np.float64), np.array([0.0, 0.0, -1.0], dtype=np.float64)),  # -X
    4: (np.array([1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64)),   # +Z
    5: (np.array([-1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64)),  # +Y
}


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))[0].ravel()


def rotvec_to_matrix(rvec: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]


def se3_exp_rot(delta: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(delta))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = np.asarray(delta, dtype=np.float64).reshape(3) / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def average_rotation(Rs: list[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
    if not Rs:
        raise ValueError("average_rotation requires at least one rotation")
    Rm = np.asarray(Rs[0], dtype=np.float64).reshape(3, 3).copy()
    if weights is None:
        w = np.ones(len(Rs), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w) != len(Rs):
            raise ValueError("weights length must match rotations length")
        w = np.maximum(w, 1e-12)
    w = w / np.sum(w)

    for _ in range(50):
        d = np.zeros(3, dtype=np.float64)
        for wi, R in zip(w, Rs):
            d += wi * matrix_to_rotvec(np.asarray(R, dtype=np.float64).reshape(3, 3) @ Rm.T)
        if np.linalg.norm(d) < 1e-12:
            break
        Rm = se3_exp_rot(d) @ Rm
    return Rm


def average_translation(ts: list[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.vstack([np.asarray(t, dtype=np.float64).reshape(1, 3) for t in ts])
    if weights is None:
        return np.mean(arr, axis=0)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.maximum(w, 1e-12)
    w = w / np.sum(w)
    return np.sum(arr * w[:, None], axis=0)


def cube_face_center(tag_id: int, cube_half_m: float) -> Optional[np.ndarray]:
    centers = {
        0: np.array([+cube_half_m, 0.0, 0.0], dtype=np.float64),
        1: np.array([0.0, -cube_half_m, 0.0], dtype=np.float64),
        2: np.array([0.0, 0.0, -cube_half_m], dtype=np.float64),
        3: np.array([-cube_half_m, 0.0, 0.0], dtype=np.float64),
        4: np.array([0.0, 0.0, +cube_half_m], dtype=np.float64),
        5: np.array([0.0, +cube_half_m, 0.0], dtype=np.float64),
    }
    return centers.get(tag_id)


def tag_object_corners(
    tag_id: int,
    cube_half_m: float,
    tag_half_m: float,
    corner_roll: int = 0,
) -> Optional[np.ndarray]:
    c = cube_face_center(tag_id, cube_half_m)
    basis = _DEFAULT_TAG_FACE_BASIS.get(tag_id)
    if c is None or basis is None:
        return None
    u, v = basis
    obj = np.array(
        [
            c + (-tag_half_m) * u + (+tag_half_m) * v,
            c + (+tag_half_m) * u + (+tag_half_m) * v,
            c + (+tag_half_m) * u + (-tag_half_m) * v,
            c + (-tag_half_m) * u + (-tag_half_m) * v,
        ],
        dtype=np.float64,
    )
    # The legacy physical layout above was authored with +Y up. Normalize the
    # cube/world object points before PnP so emitted T_world_cam is Y-down on
    # both sides of the transform contract.
    obj[:, 1] *= -1.0
    r = int(corner_roll) % 4
    return np.roll(obj, -r, axis=0) if r else obj


def parse_fixed_corner_rolls(text: str) -> dict[int, int]:
    """Parse strings like '0:0,1:2,2:1,3:0,4:3,5:0'."""
    text = str(text or "").strip()
    if not text:
        return {}
    out: dict[int, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad --fixed-corner-rolls item: {item!r}; expected tag_id:roll")
        k, v = item.split(":", 1)
        tid = int(k.strip())
        roll = int(v.strip()) % 4
        if tid < 0 or tid > 5:
            raise ValueError(f"Unsupported tag id in --fixed-corner-rolls: {tid}")
        out[tid] = roll
    return out


@dataclass
class PoseQuality:
    mean_px: float
    rms_px: float
    max_px: float
    n_points: int
    n_inliers: int


@dataclass
class Obs:
    T_cam_cube: np.ndarray
    quality: PoseQuality
    face_ids: list[int]
    corner_rolls: dict[int, int]
    frame_index: int


@dataclass
class CamStats:
    processed: int = 0
    detected_any_tag: int = 0
    detected_valid_cube_tag: int = 0
    accepted: int = 0
    reject_no_tag: int = 0
    reject_faces: int = 0
    reject_pnp: int = 0
    reject_reproj: int = 0
    rejected_pose_outlier: int = 0

    def asdict(self) -> dict[str, int]:
        return {
            "processed": self.processed,
            "detected_any_tag": self.detected_any_tag,
            "detected_valid_cube_tag": self.detected_valid_cube_tag,
            "accepted": self.accepted,
            "reject_no_tag": self.reject_no_tag,
            "reject_faces": self.reject_faces,
            "reject_pnp": self.reject_pnp,
            "reject_reproj": self.reject_reproj,
            "rejected_pose_outlier": self.rejected_pose_outlier,
        }


@dataclass
class SolveCandidate:
    rv: np.ndarray
    tv: np.ndarray
    quality: PoseQuality
    rolls: dict[int, int]
    inliers: Optional[np.ndarray] = None


class Calibrator:
    def __init__(
        self,
        cube_size_mm: float,
        tag_size_mm: float,
        max_reproj_error_px: float,
        max_point_error_px: float,
        ransac_reproj_error_px: float,
        min_faces: int = 2,
        auto_corner_rolls: bool = True,
        fixed_corner_rolls: Optional[dict[int, int]] = None,
        min_observations: int = 3,
        pose_outlier_rot_deg: float = 5.0,
        pose_outlier_trans_m: float = 0.03,
    ):
        self.cube_half_m = float(cube_size_mm) / 2000.0
        self.tag_half_m = float(tag_size_mm) / 2000.0
        if self.cube_half_m <= 0:
            raise ValueError("cube_size_mm must be positive")
        if self.tag_half_m <= 0:
            raise ValueError("tag_size_mm must be positive")
        if self.tag_half_m > self.cube_half_m + 1e-12:
            raise ValueError("tag_size_mm cannot be larger than cube_size_mm for a face-mounted tag")

        self.max_reproj_error_px = float(max_reproj_error_px)
        self.max_point_error_px = float(max_point_error_px)
        self.ransac_reproj_error_px = float(ransac_reproj_error_px)
        self.min_faces = max(2, int(min_faces))
        self.auto_corner_rolls = bool(auto_corner_rolls)
        self.fixed_corner_rolls = dict(fixed_corner_rolls or {})
        self.min_observations = max(1, int(min_observations))
        self.pose_outlier_rot_deg = float(pose_outlier_rot_deg)
        self.pose_outlier_trans_m = float(pose_outlier_trans_m)

        self.obs: dict[str, list[Obs]] = defaultdict(list)
        self.stats: dict[str, CamStats] = defaultdict(CamStats)
        self.corner_roll_hist: dict[str, Counter[str]] = defaultdict(Counter)

    def _frame_entries(self, corners_per_id: dict[int, np.ndarray]) -> list[tuple[int, np.ndarray, np.ndarray]]:
        entries: list[tuple[int, np.ndarray, np.ndarray]] = []
        for tid in sorted(corners_per_id.keys()):
            base_obj = tag_object_corners(tid, self.cube_half_m, self.tag_half_m, corner_roll=0)
            if base_obj is None:
                continue
            img = np.asarray(corners_per_id[tid], dtype=np.float64).reshape(4, 2)
            entries.append((tid, base_obj, img))
        return entries

    @staticmethod
    def _reproj_errors(objp: np.ndarray, imgp: np.ndarray, rv: np.ndarray, tv: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
        proj, _ = cv2.projectPoints(objp, rv, tv, K, dist)
        return np.linalg.norm(proj.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1)

    def _solve_with_rolls(
        self,
        entries: list[tuple[int, np.ndarray, np.ndarray]],
        rolls: tuple[int, ...],
        K: np.ndarray,
        dist: np.ndarray,
    ) -> Optional[SolveCandidate]:
        obj_all: list[np.ndarray] = []
        img_all: list[np.ndarray] = []
        roll_map: dict[int, int] = {}
        for (tid, obj, img), r in zip(entries, rolls):
            r_int = int(r) % 4
            obj_use = np.roll(obj, -r_int, axis=0) if r_int else obj
            obj_all.append(obj_use)
            img_all.append(img)
            roll_map[tid] = r_int

        objp = np.vstack(obj_all).astype(np.float64)
        imgp = np.vstack(img_all).astype(np.float64)
        if objp.shape[0] < 6:
            return None

        try:
            ok, rv, tv, inliers = cv2.solvePnPRansac(
                objp,
                imgp,
                K,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
                confidence=0.999,
                reprojectionError=self.ransac_reproj_error_px,
                iterationsCount=200,
            )
        except cv2.error:
            return None

        if not ok or rv is None or tv is None:
            return None

        # Refine on all points with the RANSAC estimate as the initial guess.
        try:
            ok2, rv2, tv2 = cv2.solvePnP(
                objp,
                imgp,
                K,
                dist,
                rvec=rv,
                tvec=tv,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok2 and rv2 is not None and tv2 is not None:
                rv, tv = rv2, tv2
        except cv2.error:
            pass

        errs = self._reproj_errors(objp, imgp, rv, tv, K, dist)
        if not np.all(np.isfinite(errs)):
            return None

        q = PoseQuality(
            mean_px=float(np.mean(errs)),
            rms_px=float(np.sqrt(np.mean(errs * errs))),
            max_px=float(np.max(errs)),
            n_points=int(objp.shape[0]),
            n_inliers=0 if inliers is None else int(len(inliers)),
        )
        return SolveCandidate(rv=rv, tv=tv, quality=q, rolls=roll_map, inliers=inliers)

    def add_processed_frame(self, serial: str) -> None:
        self.stats[serial].processed += 1

    def add_no_tag_frame(self, serial: str) -> None:
        st = self.stats[serial]
        st.reject_no_tag += 1

    def add_frame(
        self,
        serial: str,
        frame_index: int,
        corners_per_id: dict[int, np.ndarray],
        K: np.ndarray,
        dist: np.ndarray,
    ) -> bool:
        st = self.stats[serial]
        if corners_per_id:
            st.detected_valid_cube_tag += 1

        entries = self._frame_entries(corners_per_id)
        if len(entries) < self.min_faces:
            st.reject_faces += 1
            return False

        candidate_rolls: list[tuple[int, ...]] = []
        if self.auto_corner_rolls:
            ranges: list[list[int]] = []
            for tid, _, _ in entries:
                if tid in self.fixed_corner_rolls:
                    ranges.append([self.fixed_corner_rolls[tid]])
                else:
                    ranges.append([0, 1, 2, 3])
            candidate_rolls = list(itertools.product(*ranges))
        else:
            rolls = []
            for tid, _, _ in entries:
                rolls.append(self.fixed_corner_rolls.get(tid, 0))
            candidate_rolls = [tuple(rolls)]

        best: Optional[SolveCandidate] = None
        for rolls in candidate_rolls:
            c = self._solve_with_rolls(entries, rolls, K, dist)
            if c is None:
                continue
            if best is None or c.quality.rms_px < best.quality.rms_px:
                best = c

        if best is None:
            st.reject_pnp += 1
            return False

        if best.quality.rms_px > self.max_reproj_error_px or best.quality.max_px > self.max_point_error_px:
            st.reject_reproj += 1
            return False

        R = rotvec_to_matrix(best.rv)
        T = make_T(R, best.tv.ravel())
        face_ids = [int(tid) for tid, _, _ in entries]
        self.obs[serial].append(
            Obs(
                T_cam_cube=T,
                quality=best.quality,
                face_ids=face_ids,
                corner_rolls=best.rolls,
                frame_index=int(frame_index),
            )
        )
        for tid, roll in best.rolls.items():
            self.corner_roll_hist[serial][f"tag{tid}:roll{roll}"] += 1
        st.accepted += 1
        return True

    def _pose_outlier_mask(self, obs: list[Obs]) -> np.ndarray:
        n = len(obs)
        if n < 4:
            return np.ones(n, dtype=bool)
        Rs = [o.T_cam_cube[:3, :3] for o in obs]
        ts = [o.T_cam_cube[:3, 3] for o in obs]
        reproj = np.array([max(o.quality.rms_px, 1e-3) for o in obs], dtype=np.float64)
        w = 1.0 / reproj
        R0 = average_rotation(Rs, weights=w)
        t0 = average_translation(ts, weights=w)

        rot_err = np.array([np.linalg.norm(matrix_to_rotvec(R @ R0.T)) * 180.0 / math.pi for R in Rs], dtype=np.float64)
        trans_err = np.array([np.linalg.norm(t - t0) for t in ts], dtype=np.float64)

        rot_med = float(np.median(rot_err))
        trans_med = float(np.median(trans_err))
        rot_mad = float(np.median(np.abs(rot_err - rot_med)))
        trans_mad = float(np.median(np.abs(trans_err - trans_med)))

        # Use both robust and absolute gates. Absolute gates are important because static cube/camera
        # observations should be tightly clustered.
        rot_gate = min(self.pose_outlier_rot_deg, rot_med + max(3.0 * 1.4826 * rot_mad, 1.0))
        trans_gate = min(self.pose_outlier_trans_m, trans_med + max(3.0 * 1.4826 * trans_mad, 0.005))
        mask = (rot_err <= rot_gate) & (trans_err <= trans_gate)
        if int(np.count_nonzero(mask)) < max(2, min(self.min_observations, n)):
            return np.ones(n, dtype=bool)
        return mask

    def estimate_all(self) -> dict[str, Optional[dict]]:
        out: dict[str, Optional[dict]] = {}
        all_serials = sorted(set(self.stats.keys()) | set(self.obs.keys()))
        for serial in all_serials:
            obs_all = self.obs.get(serial, [])
            if len(obs_all) < self.min_observations:
                out[serial] = None
                continue

            mask = self._pose_outlier_mask(obs_all)
            used = [o for o, keep in zip(obs_all, mask) if bool(keep)]
            rejected = len(obs_all) - len(used)
            self.stats[serial].rejected_pose_outlier += int(rejected)
            if len(used) < self.min_observations:
                out[serial] = None
                continue

            Rs = [o.T_cam_cube[:3, :3] for o in used]
            ts = [o.T_cam_cube[:3, 3] for o in used]
            rms = np.array([max(o.quality.rms_px, 1e-3) for o in used], dtype=np.float64)
            weights = 1.0 / rms
            R = average_rotation(Rs, weights=weights)
            t = average_translation(ts, weights=weights)
            T_cam_cube = make_T(R, t)
            T_cube_cam = np.linalg.inv(T_cam_cube)

            out[serial] = {
                "T_cam_cube": T_cam_cube,
                "T_cube_cam": T_cube_cam,
                "n_observations_total": int(len(obs_all)),
                "n_observations_used": int(len(used)),
                "n_observations_pose_outliers": int(rejected),
                "quality": {
                    "mean_rms_reproj_px": float(np.mean([o.quality.rms_px for o in used])),
                    "median_rms_reproj_px": float(np.median([o.quality.rms_px for o in used])),
                    "max_rms_reproj_px": float(np.max([o.quality.rms_px for o in used])),
                    "mean_max_point_reproj_px": float(np.mean([o.quality.max_px for o in used])),
                    "max_point_reproj_px": float(np.max([o.quality.max_px for o in used])),
                    "mean_faces_per_frame": float(np.mean([len(o.face_ids) for o in used])),
                },
            }
        return out


def _load_camera_meta(metadata_json: Path) -> dict[str, dict]:
    if not metadata_json.exists():
        raise RuntimeError(f"camera metadata not found: {metadata_json}")
    try:
        data = json.loads(metadata_json.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse camera metadata: {metadata_json}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"camera metadata root must be a JSON object: {metadata_json}")
    return data


def _make_guessed_intrinsics(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    fx = fy = float(max(w, h)) * 0.9
    cx = float(w) * 0.5
    cy = float(h) * 0.5
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.zeros((5,), dtype=np.float64)
    return K, D


def _extract_intrinsics(
    meta: dict[str, dict],
    serial: str,
    w: int,
    h: int,
    allow_guessed_intrinsics: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    m = meta.get(serial, {}) if isinstance(meta, dict) else {}
    K = m.get("K") if isinstance(m, dict) else None
    dist = m.get("dist") if isinstance(m, dict) else None

    K_np = None
    if K is not None:
        try:
            K_np = np.array(K, dtype=np.float64).reshape(3, 3)
        except Exception as exc:
            raise RuntimeError(f"invalid K for camera serial={serial}: {exc}") from exc

    if K_np is None:
        if allow_guessed_intrinsics:
            print(f"[WARNING] {serial}: missing K; using guessed intrinsics. Results may be invalid.")
            return _make_guessed_intrinsics(w, h)
        raise RuntimeError(
            f"missing camera intrinsics for serial={serial}. "
            "Provide camera_metadata.json or pass --allow-guessed-intrinsics for debugging only."
        )

    if K_np[0, 0] <= 0 or K_np[1, 1] <= 0 or abs(K_np[2, 2] - 1.0) > 1e-6:
        raise RuntimeError(f"invalid K values for serial={serial}: {K_np}")

    if dist is not None:
        try:
            D_np = np.array(dist, dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise RuntimeError(f"invalid dist for camera serial={serial}: {exc}") from exc
    else:
        D_np = np.zeros((5,), dtype=np.float64)
    return K_np, D_np


def detect_markers(gray: np.ndarray, ar_dict, ar_params):
    try:
        detector = cv2.aruco.ArucoDetector(ar_dict, ar_params)
        return detector.detectMarkers(gray)
    except AttributeError:
        return cv2.aruco.detectMarkers(gray, ar_dict, parameters=ar_params)


def make_aruco_params():
    params = cv2.aruco.DetectorParameters()
    # Subpixel refinement improves pose stability. Attribute names vary by OpenCV version,
    # so use guarded assignment.
    try:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01
    except Exception:
        pass
    return params


def convert_to_gray(frame: np.ndarray, color_order: str) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"unsupported image shape: {frame.shape}")
    if color_order == "rgb":
        return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)
    if color_order == "bgr":
        return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    raise ValueError(f"unsupported color order: {color_order}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay AprilTag 16h5 tag-cube calibrator with robust diagnostics")
    ap.add_argument("--session-dir", required=True, help="Replay session directory containing rgbd/*.h5")
    ap.add_argument("--metadata-json", default="", help="Camera metadata json path; default: session_dir/camera_metadata.json")
    ap.add_argument("--cube-size-mm", "--cube-size", dest="cube_size_mm", type=float, default=50.0, help="Physical cube edge length in mm")
    ap.add_argument(
        "--tag-size-mm",
        "--tag-size",
        "--tag_size",
        "--tag_size_mm",
        dest="tag_size_mm",
        type=float,
        default=0.0,
        help="AprilTag outer edge length in mm; default equals cube size",
    )
    ap.add_argument("--max-reproj-error-px", type=float, default=5.0, help="Maximum accepted per-frame RMS reprojection error")
    ap.add_argument("--max-point-error-px", type=float, default=12.0, help="Maximum accepted single-corner reprojection error")
    ap.add_argument("--ransac-reproj-error-px", type=float, default=2.0, help="solvePnPRansac inlier threshold")
    ap.add_argument("--min-faces", type=int, default=2, help="Minimum cube faces required per frame; forced to at least 2")
    ap.add_argument("--min-observations", type=int, default=3, help="Minimum accepted frames required per camera")
    ap.add_argument("--reference-serial", default="", help="Reference camera serial for T_ref_cam outputs; default first valid serial")
    ap.add_argument("--stride", type=int, default=3, help="Frame stride for replay processing")
    ap.add_argument("--max-frames", type=int, default=0, help="Per-camera max frames to process; 0 means all")
    ap.add_argument("--color-order", choices=["rgb", "bgr"], default="bgr", help="Color dataset channel order before grayscale conversion")
    ap.add_argument("--allow-guessed-intrinsics", action="store_true", help="Debug only: allow guessed intrinsics when metadata is missing")
    ap.add_argument("--auto-corner-rolls", dest="auto_corner_rolls", action="store_true", default=True, help="Try cyclic corner rolls for unknown physical tag orientation")
    ap.add_argument("--no-auto-corner-rolls", dest="auto_corner_rolls", action="store_false", help="Disable automatic corner-roll search")
    ap.add_argument("--fixed-corner-rolls", default="", help="Fixed rolls, e.g. '0:0,1:2,2:1,3:0,4:3,5:0'")
    ap.add_argument("--pose-outlier-rot-deg", type=float, default=5.0, help="Absolute rotation gate for pose outlier rejection")
    ap.add_argument("--pose-outlier-trans-m", type=float, default=0.03, help="Absolute translation gate for pose outlier rejection")
    ap.add_argument("--show", action="store_true", help="Visualize detections")
    ap.add_argument("--out", default="", help="Output json path; default session_dir/tag_cube_extrinsics_fixed.json")
    args = ap.parse_args()

    session_dir = Path(args.session_dir)
    rgbd_dir = session_dir / "rgbd"
    if not rgbd_dir.exists() or not rgbd_dir.is_dir():
        raise RuntimeError(f"rgbd directory not found: {rgbd_dir}")

    h5_files = sorted(rgbd_dir.glob("*.h5"))
    if not h5_files:
        raise RuntimeError(f"No h5 files found under: {rgbd_dir}")

    metadata_json = Path(args.metadata_json) if str(args.metadata_json).strip() else (session_dir / "camera_metadata.json")
    meta = _load_camera_meta(metadata_json)

    out_json = Path(args.out) if str(args.out).strip() else (session_dir / "tag_cube_extrinsics_fixed.json")
    tag_size_mm = float(args.tag_size_mm) if float(args.tag_size_mm) > 0 else float(args.cube_size_mm)
    fixed_rolls = parse_fixed_corner_rolls(args.fixed_corner_rolls)

    cal = Calibrator(
        cube_size_mm=float(args.cube_size_mm),
        tag_size_mm=tag_size_mm,
        max_reproj_error_px=float(args.max_reproj_error_px),
        max_point_error_px=float(args.max_point_error_px),
        ransac_reproj_error_px=float(args.ransac_reproj_error_px),
        min_faces=int(args.min_faces),
        auto_corner_rolls=bool(args.auto_corner_rolls),
        fixed_corner_rolls=fixed_rolls,
        min_observations=int(args.min_observations),
        pose_outlier_rot_deg=float(args.pose_outlier_rot_deg),
        pose_outlier_trans_m=float(args.pose_outlier_trans_m),
    )

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is not available. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, "DICT_APRILTAG_16h5"):
        raise RuntimeError("This OpenCV build does not expose DICT_APRILTAG_16h5. Install/update opencv-contrib-python.")

    ar_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    ar_params = make_aruco_params()

    start = time.time()
    for p in h5_files:
        serial = p.stem
        with h5py.File(str(p), "r") as f:
            if "color" not in f:
                print(f"[!] {serial}: no color dataset, skip")
                continue
            color = f["color"]
            if color.ndim != 4 or color.shape[0] <= 0:
                print(f"[!] {serial}: bad color shape {color.shape}, skip")
                continue

            h, w = int(color.shape[1]), int(color.shape[2])
            K, D = _extract_intrinsics(meta, serial, w=w, h=h, allow_guessed_intrinsics=bool(args.allow_guessed_intrinsics))

            n = int(color.shape[0])
            stride = max(1, int(args.stride))
            max_frames = int(args.max_frames)
            seen = 0

            print(f"[i] {serial}: frames={n}, size={w}x{h}, stride={stride}, K=({K[0,0]:.2f},{K[1,1]:.2f},{K[0,2]:.2f},{K[1,2]:.2f})")

            for i in range(0, n, stride):
                if max_frames > 0 and seen >= max_frames:
                    break
                seen += 1
                cal.add_processed_frame(serial)

                frame = np.asarray(color[i])
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

                try:
                    gray = convert_to_gray(frame, args.color_order)
                except Exception as exc:
                    print(f"[!] {serial}: failed to convert frame {i} to gray: {exc}")
                    continue

                corners, ids, _ = detect_markers(gray, ar_dict, ar_params)
                if ids is None or len(ids) == 0:
                    cal.add_no_tag_frame(serial)
                    continue

                cal.stats[serial].detected_any_tag += 1
                corners_per_id: dict[int, np.ndarray] = {}
                for c, id_arr in zip(corners, ids):
                    tid = int(id_arr[0])
                    if 0 <= tid <= 5:
                        corners_per_id[tid] = np.asarray(c, dtype=np.float64).reshape(4, 2)

                cal.add_frame(serial, frame_index=i, corners_per_id=corners_per_id, K=K, dist=D)

                if args.show:
                    vis = frame.copy()
                    if args.color_order == "rgb":
                        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                    try:
                        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                    except Exception:
                        pass
                    st = cal.stats[serial]
                    cv2.putText(
                        vis,
                        f"{serial} frame {i+1}/{n} acc={st.accepted}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(f"calib::{serial}", vis)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

    if args.show:
        cv2.destroyAllWindows()

    estimates = cal.estimate_all()

    cameras_out: dict[str, dict] = {}
    for serial, est in estimates.items():
        if est is None:
            cameras_out[serial] = {
                "valid": False,
                "T_cam_cube": None,
                "T_cube_cam": None,
                "T_world_cam": None,
                "note": "Not enough valid observations after filtering",
            }
            continue
        T_cam_cube = np.asarray(est["T_cam_cube"], dtype=np.float64)
        T_cube_cam = np.asarray(est["T_cube_cam"], dtype=np.float64)
        cameras_out[serial] = {
            "valid": True,
            "T_cam_cube": T_cam_cube.tolist(),
            "T_cube_cam": T_cube_cam.tolist(),
            # Alias for compatibility with old consumers. This is only correct because
            # this script defines the static cube frame as the world frame.
            "T_world_cam": T_cube_cam.tolist(),
            "world_frame_definition": "cube frame; therefore T_world_cam is an alias of T_cube_cam",
            "n_observations_total": est["n_observations_total"],
            "n_observations_used": est["n_observations_used"],
            "n_observations_pose_outliers": est["n_observations_pose_outliers"],
            "quality": est["quality"],
        }

    # Camera-to-camera extrinsics.
    # T_ref_cam maps points from each camera coordinate frame into the reference camera frame:
    #     X_ref = T_ref_cam @ X_cam
    # Because each camera pose is estimated in the same static cube/world frame:
    #     T_ref_cam = inv(T_cube_ref) @ T_cube_cam = T_ref_cube @ T_cube_cam
    valid_serials = [s for s in sorted(cameras_out.keys()) if cameras_out[s].get("valid", False)]
    requested_ref = str(args.reference_serial).strip()
    if requested_ref:
        if requested_ref not in cameras_out or not cameras_out[requested_ref].get("valid", False):
            raise RuntimeError(f"--reference-serial {requested_ref!r} is not a valid calibrated camera. Valid: {valid_serials}")
        reference_serial = requested_ref
    else:
        reference_serial = valid_serials[0] if valid_serials else ""

    T_ref_cam: dict[str, Optional[list]] = {}
    T_cam_ref: dict[str, Optional[list]] = {}
    pairwise_T_target_source: dict[str, dict[str, Optional[list]]] = {}
    if reference_serial:
        T_cube_ref = np.asarray(cameras_out[reference_serial]["T_cube_cam"], dtype=np.float64)
        T_ref_cube = np.linalg.inv(T_cube_ref)
        for serial, cam in cameras_out.items():
            if not cam.get("valid", False):
                T_ref_cam[serial] = None
                T_cam_ref[serial] = None
                continue
            T_cube_cam_i = np.asarray(cam["T_cube_cam"], dtype=np.float64)
            T = T_ref_cube @ T_cube_cam_i
            T_ref_cam[serial] = T.tolist()
            T_cam_ref[serial] = np.linalg.inv(T).tolist()
            cam["T_ref_cam"] = T.tolist()
            cam["T_cam_ref"] = np.linalg.inv(T).tolist()
        for target in valid_serials:
            pairwise_T_target_source[target] = {}
            T_cube_target = np.asarray(cameras_out[target]["T_cube_cam"], dtype=np.float64)
            T_target_cube = np.linalg.inv(T_cube_target)
            for source in valid_serials:
                T_cube_source = np.asarray(cameras_out[source]["T_cube_cam"], dtype=np.float64)
                pairwise_T_target_source[target][source] = (T_target_cube @ T_cube_source).tolist()

    out = {
        "script": "tag_cube_calibrator_replay_standalone_with_tag_size.py",
        "coordinate_system": "y_down",
        "convention": "column vectors; X_target = T_target_source @ X_source_h",
        "unit": "meter",
        "reference_serial": reference_serial,
        "T_ref_cam": T_ref_cam,
        "T_cam_ref": T_cam_ref,
        "pairwise_T_target_source": pairwise_T_target_source,
        "tag_family": "DICT_APRILTAG_16h5",
        "cube_size_mm": float(args.cube_size_mm),
        "tag_size_mm": float(tag_size_mm),
        "tag_fills_face": abs(float(args.cube_size_mm) - float(tag_size_mm)) < 1e-9,
        "session_dir": str(session_dir),
        "metadata_json": str(metadata_json),
        "color_order": args.color_order,
        "min_faces": max(2, int(args.min_faces)),
        "min_observations": max(1, int(args.min_observations)),
        "max_reproj_error_px_rms": float(args.max_reproj_error_px),
        "max_point_error_px": float(args.max_point_error_px),
        "ransac_reproj_error_px": float(args.ransac_reproj_error_px),
        "auto_corner_rolls": bool(args.auto_corner_rolls),
        "fixed_corner_rolls": fixed_rolls,
        "default_face_layout": {
            "0": "+X center [+half,0,0]",
            "1": "-Y center [0,-half,0]",
            "2": "-Z center [0,0,-half]",
            "3": "-X center [-half,0,0]",
            "4": "+Z center [0,0,+half]",
            "5": "+Y center [0,+half,0]",
        },
        "stats": {s: st.asdict() for s, st in sorted(cal.stats.items())},
        "corner_roll_histogram": {s: dict(counter) for s, counter in sorted(cal.corner_roll_hist.items())},
        "cameras": cameras_out,
        # Backward-compatible top-level output used by the old script.
        "T_world_cam": {s: cam["T_world_cam"] for s, cam in cameras_out.items()},
        "duration_sec": float(time.time() - start),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")
    if reference_serial:
        print(f"[i] Reference camera for multi-camera extrinsics: {reference_serial}")
        print("[i] Use T_ref_cam[serial] as X_ref = T_ref_cam @ X_cam")
    for serial, cam in sorted(cameras_out.items()):
        st = cal.stats[serial].asdict() if serial in cal.stats else {}
        if cam["valid"]:
            q = cam["quality"]
            print(
                f"[OK] {serial}: used={cam['n_observations_used']}/{cam['n_observations_total']}, "
                f"rms_mean={q['mean_rms_reproj_px']:.3f}px, max_point={q['max_point_reproj_px']:.3f}px, "
                f"accepted={st.get('accepted', 0)}"
            )
        else:
            print(f"[FAIL] {serial}: {cam['note']}; stats={st}")


if __name__ == "__main__":
    main()
