"""Single-camera M3T region+depth tracking; no cross-camera optimization here."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np


def load_native():
    if os.environ.get("CONDA_DEFAULT_ENV") != "mr_nav":
        raise RuntimeError("M3T must run in conda mr_nav")
    if "_m3t_native" in sys.modules:
        return sys.modules["_m3t_native"]
    directory = Path(__file__).parent / "m3t_native/build/Release"
    matches = list(directory.glob("_m3t_native*.pyd"))
    if len(matches) != 1:
        raise RuntimeError("Build M3T first: .\\tools\\build_m3t.ps1")
    # Retain the handle for the entire lifetime of the native extension.
    handle = os.add_dll_directory(str(Path(sys.prefix) / "Library/bin"))
    spec = importlib.util.spec_from_file_location("_m3t_native", matches[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._dll_directory = handle
    sys.modules["_m3t_native"] = module
    return module


def depth_fit(observed: np.ndarray, rendered: np.ndarray) -> dict:
    """Same residual definitions as the FP adapter; renderers differ, no GT use."""
    observed = np.asarray(observed, dtype=np.float32)
    rendered = np.asarray(rendered, dtype=np.float32)
    if observed.shape != rendered.shape or observed.ndim != 2:
        raise ValueError("Depth images must have matching HxW shapes")
    mask = np.isfinite(rendered) & (rendered > 1e-6)
    overlap = mask & np.isfinite(observed) & (observed >= 1e-3)
    count = int(mask.sum())
    result = {"status": "ok", "rendered_pixel_count": count,
              "observed_overlap_fraction": float(overlap.sum() / count) if count else None,
              "depth_inlier_fraction": None, "depth_abs_error_median_m": None,
              "depth_abs_error_p95_m": None, "occluded_or_depth_mismatch_fraction": None}
    if overlap.any():
        residual = np.abs(observed[overlap] - rendered[overlap])
        inlier = float(np.mean(residual <= np.maximum(0.02, 0.03 * rendered[overlap])))
        result.update(depth_inlier_fraction=inlier, depth_abs_error_median_m=float(np.median(residual)),
                      depth_abs_error_p95_m=float(np.percentile(residual, 95)),
                      occluded_or_depth_mismatch_fraction=1 - inlier)
    return result


def validate_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-3)):
        raise ValueError("Expected finite rigid object-to-camera SE(3) pose")
    return pose


class M3TRunner:
    def __init__(self, mesh_path: Path, cache_root: Path, *, threads: int = 4):
        self.native = load_native()
        self.mesh_path = Path(mesh_path).resolve()
        key = hashlib.sha256(self.mesh_path.read_bytes()).hexdigest()[:16]
        self.cache = Path(cache_root).resolve() / f"{self.mesh_path.stem}_{key}"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.threads = threads
        self.trackers = {}
        self.intrinsics = {}

    def process(self, camera: str, color: np.ndarray, depth_m: np.ndarray, K: np.ndarray,
                initial_pose: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        color = np.asarray(color)
        depth_m = np.asarray(depth_m)
        K = np.asarray(K, dtype=np.float32)
        if (color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3
                or depth_m.shape != color.shape[:2] or not np.isfinite(depth_m).all()
                or (depth_m < 0).any() or (depth_m > 65.535).any()
                or K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0):
            raise ValueError("Invalid native-resolution registered RGB-D input or intrinsics")
        color = np.ascontiguousarray(color)
        depth = np.ascontiguousarray(np.rint(depth_m * 1000), dtype=np.uint16)
        if camera not in self.trackers:
            if initial_pose is None:
                raise ValueError("M3T needs an explicit first-frame pose; no automatic GT fallback")
            pose = validate_pose(initial_pose)
            self.trackers[camera] = self.native.SingleCameraTracker(
                str(self.mesh_path), str(self.cache), K, color, depth, pose.astype(np.float32), self.threads)
            self.intrinsics[camera] = K.copy()
        else:
            if initial_pose is not None:
                raise ValueError("Do not feed annotations into subsequent tracking frames")
            if not np.array_equal(self.intrinsics[camera], K):
                raise ValueError("Camera intrinsics changed within take")
            pose = validate_pose(self.trackers[camera].track(color, depth))
        quality = depth_fit(depth_m, self.trackers[camera].rendered_depth())
        return pose, quality
