from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh


@dataclass(frozen=True)
class HopeModel:
    name: str
    mesh_path: Path
    scale_to_m: float


@dataclass(frozen=True)
class FoundationPoseScore:
    """Diagnostic rank logit, never a calibrated observation probability."""

    raw_logit: float | None
    status: str
    elapsed_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class FoundationPoseEstimate:
    object_name: str
    pose_cam: np.ndarray
    score: float
    mode: str
    scoring: FoundationPoseScore | None = None


class HopeModelRepository:
    """Discovers the HOPE object meshes and normalizes their units to metres."""

    def __init__(self, root: str | Path, scale_to_m: float = 0.01) -> None:
        self.root = Path(root).expanduser().resolve()
        self.scale_to_m = float(scale_to_m)
        if self.scale_to_m <= 0:
            raise ValueError("scale_to_m must be positive")

    def names(self) -> list[str]:
        return sorted(self._models())

    def resolve(self, name: str) -> HopeModel:
        models = self._models()
        key = str(name).strip().casefold()
        matches = [model for model in models.values() if model.name.casefold() == key]
        if not matches:
            raise KeyError(
                f"Unknown HOPE object {name!r}; available: {', '.join(sorted(models))}"
            )
        return matches[0]

    def load_mesh(self, name: str) -> tuple[HopeModel, trimesh.Trimesh]:
        model = self.resolve(name)
        mesh = trimesh.load(model.mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"HOPE mesh did not load as Trimesh: {model.mesh_path}")
        mesh = mesh.copy()
        mesh.apply_scale(model.scale_to_m)
        return model, mesh

    def _models(self) -> dict[str, HopeModel]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"HOPE models directory not found: {self.root}")
        found: dict[str, HopeModel] = {}
        for mesh_path in sorted(self.root.glob("*/google_16k/textured_simple.obj")):
            name = mesh_path.parents[1].name
            found[name] = HopeModel(name, mesh_path.resolve(), self.scale_to_m)
        if not found:
            raise FileNotFoundError(
                f"No */google_16k/textured_simple.obj meshes found under {self.root}"
            )
        return found


class FoundationPoseRunner:
    """Synchronous, thread-safe model-based FoundationPose runner for HOPE meshes."""

    def __init__(
        self,
        foundationpose_root: str | Path,
        hope_models_root: str | Path | None = None,
        *,
        model_repository: object | None = None,
        hope_scale_to_m: float = 0.01,
        register_iterations: int = 5,
        track_iterations: int = 2,
        score_tracking: bool = False,
        warp_cache_path: str | Path | None = None,
    ) -> None:
        self.foundationpose_root = Path(foundationpose_root).expanduser().resolve()
        if model_repository is not None:
            self.models = model_repository
        elif hope_models_root is not None:
            self.models = HopeModelRepository(hope_models_root, hope_scale_to_m)
        else:
            raise ValueError("Either hope_models_root or model_repository is required")
        self.register_iterations = max(1, int(register_iterations))
        self.track_iterations = max(1, int(track_iterations))
        # Optional diagnostic: the ranking network's output has not been
        # calibrated across cameras/frames and must not become a fusion weight.
        self.score_tracking = bool(score_tracking)
        self.warp_cache_path = Path(
            warp_cache_path or self.foundationpose_root / ".cache" / "warp"
        ).resolve()
        self._lock = threading.Lock()
        self._loaded = False
        self._estimators: dict[tuple[str, str], object] = {}
        self._mesh_cache: dict[str, trimesh.Trimesh] = {}

    def available_objects(self) -> list[str]:
        return self.models.names()

    def model_bounds(self, object_name: str) -> np.ndarray:
        """Return the metric object-space AABB as [[min_xyz], [max_xyz]]."""
        with self._lock:
            resolved = self.models.resolve(object_name).name
            mesh = self._mesh_cache.get(resolved)
            if mesh is None:
                _, mesh = self.models.load_mesh(resolved)
                self._mesh_cache[resolved] = mesh
            return np.asarray(mesh.bounds, dtype=np.float64).copy()

    def prepare(self, object_names: list[str] | None = None) -> None:
        """Load networks and requested CAD meshes before the first camera frame."""
        with self._lock:
            self._ensure_loaded()
            for name in object_names or []:
                resolved = self.models.resolve(name).name
                if resolved not in self._mesh_cache:
                    _, self._mesh_cache[resolved] = self.models.load_mesh(resolved)

    def reset(self, source: str | None = None, object_name: str | None = None) -> None:
        with self._lock:
            keys = list(self._estimators)
            for key in keys:
                if source is not None and key[0] != source:
                    continue
                if object_name is not None and key[1] != object_name:
                    continue
                self._estimators.pop(key, None)

    def process(
        self,
        *,
        source: str,
        object_name: str,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        camera_matrix: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> FoundationPoseEstimate | None:
        with self._lock:
            self._ensure_loaded()
            color = np.asarray(color_bgr)
            depth = np.asarray(depth_m, dtype=np.float32)
            K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
            if color.ndim != 3 or color.shape[2] < 3:
                raise ValueError("color_bgr must have shape HxWx3")
            if depth.shape != color.shape[:2]:
                raise ValueError("depth and color resolutions must match")
            if mask is not None and np.asarray(mask).shape[:2] != color.shape[:2]:
                raise ValueError("mask and color resolutions must match")
            rgb = cv2.cvtColor(color[..., :3], cv2.COLOR_BGR2RGB)
            key = (str(source), self.models.resolve(object_name).name)
            estimator = self._estimators.get(key)
            scoring = FoundationPoseScore(None, "disabled")

            if estimator is None or getattr(estimator, "pose_last", None) is None:
                if mask is None:
                    return None
                estimator = self._get_estimator(key)
                pose = estimator.register(
                    K=K,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=np.asarray(mask).astype(bool),
                    iteration=self.register_iterations,
                )
                mode = "register"
                score = 1.0
                scoring = self._registration_score(estimator)
            elif mask is not None:
                pose = estimator.register(
                    K=K,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=np.asarray(mask).astype(bool),
                    iteration=self.register_iterations,
                )
                mode = "register"
                score = 1.0
                scoring = self._registration_score(estimator)
            else:
                pose = estimator.track_one(
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    iteration=self.track_iterations,
                )
                mode = "track"
                score = 0.9
                if self.score_tracking:
                    scoring = self._track_score(estimator, rgb, depth, K)

            pose_cam = np.asarray(pose, dtype=np.float64).reshape(4, 4)
            if not np.isfinite(pose_cam).all():
                raise RuntimeError("FoundationPose returned a non-finite pose")
            return FoundationPoseEstimate(key[1], pose_cam, score, mode, scoring)

    @staticmethod
    def _raw_score_logit(value: object) -> float:
        """Undo ScorePredictor.predict's +100 offset after moving off CUDA."""
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        if values.size != 1 or not np.isfinite(values[0]):
            raise ValueError("Expected one finite FoundationPose rank score")
        return float(values[0] - 100.0)

    def _registration_score(self, estimator: object) -> FoundationPoseScore:
        scores = getattr(estimator, "scores", None)
        if scores is None:
            return FoundationPoseScore(None, "unavailable", error="Registration returned no rank scores")
        try:
            return FoundationPoseScore(self._raw_score_logit(scores[0]), "registration_rank")
        except (IndexError, TypeError, ValueError, RuntimeError) as exc:
            return FoundationPoseScore(None, "error", error=str(exc))

    def _get_estimator(self, key: tuple[str, str]) -> object:
        """Caller holds the runner lock; scoring must not initialize tracking."""
        estimator = self._estimators.get(key)
        if estimator is None:
            mesh = self._mesh_cache.get(key[1])
            if mesh is None:
                _, mesh = self.models.load_mesh(key[1])
                self._mesh_cache[key[1]] = mesh
            debug_dir = self.foundationpose_root / ".cache" / "runtime" / key[0].replace("/", "_") / key[1]
            estimator = self._foundation_pose(
                model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                mesh=mesh, scorer=self._scorer, refiner=self._refiner,
                glctx=self._glctx, debug=0, debug_dir=str(debug_dir),
            )
            self._estimators[key] = estimator
        return estimator

    def score_pose(
        self, *, source: str, object_name: str, color_bgr: np.ndarray,
        depth_m: np.ndarray, camera_matrix: np.ndarray, pose_cam: np.ndarray,
    ) -> FoundationPoseScore:
        """Score an existing object-to-camera pose without register/track_one.

        Only RGB-D, intrinsics, the mesh and the supplied estimate are used.
        The vendor renders a centered mesh: T_cam_center = T_cam_object @
        Translation(model_center). This conversion must be done exactly once.
        """
        return self.rank_poses(source=source, object_name=object_name, color_bgr=color_bgr,
                               depth_m=depth_m, camera_matrix=camera_matrix,
                               poses_cam=np.asarray(pose_cam).reshape(1, 4, 4))[0]

    def measure_depth_fit(
        self, *, source: str, object_name: str, depth_m: np.ndarray,
        camera_matrix: np.ndarray, pose_cam: np.ndarray,
    ) -> dict[str, float | int | str | None]:
        """Measure CAD/rendered-depth agreement for an existing pose.

        This is an observation-quality diagnostic, not a pose optimizer.  It
        renders the centered vendor mesh at the supplied pose and compares
        rendered pixels with valid RGB-D depth.  It deliberately does not use
        object GT, masks, tracking state, or the FoundationPose rank logit.
        Occlusion and missing depth are reported through the overlap fraction;
        they are not silently treated as a good fit.
        """
        with self._lock:
            try:
                self._ensure_loaded()
                depth = np.asarray(depth_m, dtype=np.float32)
                K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
                if depth.ndim != 2 or not np.isfinite(depth).all():
                    raise ValueError("depth_m must be a finite HxW array")
                key = (str(source), self.models.resolve(object_name).name)
                estimator = self._get_estimator(key)
                pose = np.asarray(pose_cam, dtype=np.float64).reshape(4, 4).copy()
                if not np.isfinite(pose).all():
                    raise ValueError("pose_cam must be finite")
                # FoundationPose internally centers the mesh before rendering.
                pose[:3, 3] += pose[:3, :3] @ np.asarray(estimator.model_center)
                import torch

                _, rendered_depth, _ = self._render_fn(
                    K=K,
                    H=depth.shape[0],
                    W=depth.shape[1],
                    ob_in_cams=torch.as_tensor(pose[None], device="cuda", dtype=torch.float32),
                    glctx=self._glctx,
                    mesh_tensors=estimator.mesh_tensors,
                )
                rendered = rendered_depth[0].detach().cpu().numpy()
                render_mask = np.isfinite(rendered) & (rendered > 1e-6)
                observed_mask = np.isfinite(depth) & (depth >= 1e-3)
                overlap = render_mask & observed_mask
                render_count = int(render_mask.sum())
                overlap_count = int(overlap.sum())
                result: dict[str, float | int | str | None] = {
                    "status": "ok",
                    "rendered_pixel_count": render_count,
                    "observed_overlap_fraction": (overlap_count / render_count
                                                    if render_count else None),
                    "depth_inlier_fraction": None,
                    "depth_abs_error_median_m": None,
                    "depth_abs_error_p95_m": None,
                    "occluded_or_depth_mismatch_fraction": None,
                }
                if overlap_count:
                    residual = np.abs(depth[overlap] - rendered[overlap])
                    # A 2 cm floor accommodates quantized consumer RGB-D while
                    # retaining a scale-aware tolerance for farther objects.
                    tolerance = np.maximum(0.02, 0.03 * rendered[overlap])
                    inliers = residual <= tolerance
                    result.update(
                        depth_inlier_fraction=float(np.mean(inliers)),
                        depth_abs_error_median_m=float(np.median(residual)),
                        depth_abs_error_p95_m=float(np.percentile(residual, 95)),
                        occluded_or_depth_mismatch_fraction=float(1.0 - np.mean(inliers)),
                    )
                return result
            except Exception as exc:
                logging.getLogger(__name__).warning("FoundationPose depth-fit diagnostic failed: %s", exc)
                return {
                    "status": "error",
                    "rendered_pixel_count": 0,
                    "observed_overlap_fraction": None,
                    "depth_inlier_fraction": None,
                    "depth_abs_error_median_m": None,
                    "depth_abs_error_p95_m": None,
                    "occluded_or_depth_mismatch_fraction": None,
                    "error": str(exc),
                }

    def rank_poses(
        self, *, source: str, object_name: str, color_bgr: np.ndarray,
        depth_m: np.ndarray, camera_matrix: np.ndarray, poses_cam: np.ndarray,
    ) -> list[FoundationPoseScore]:
        """Rank alternative object poses jointly against the same RGB-D image."""
        with self._lock:
            self._ensure_loaded()
            key = (str(source), self.models.resolve(object_name).name)
            estimator = self._get_estimator(key)
            poses = np.asarray(poses_cam, dtype=np.float64).reshape(-1, 4, 4).copy()
            if not len(poses) or not np.isfinite(poses).all():
                raise ValueError("Expected finite candidate poses")
            poses[:, :3, 3] += poses[:, :3, :3] @ np.asarray(estimator.model_center)
            rgb = cv2.cvtColor(np.asarray(color_bgr), cv2.COLOR_BGR2RGB)
            return self._score_centered_poses(estimator, rgb, depth_m, camera_matrix, poses)

    def _track_score(
        self,
        estimator: object,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
    ) -> FoundationPoseScore:
        pose_last = getattr(estimator, "pose_last", None)
        if pose_last is None:
            return FoundationPoseScore(None, "unavailable", error="No centered tracking pose")
        return self._score_centered_poses(
            estimator, rgb, depth, K, pose_last.detach().cpu().numpy().reshape(1, 4, 4)
        )[0]

    def _score_centered_poses(
        self, estimator: object, rgb: np.ndarray, depth: np.ndarray,
        K: np.ndarray, centered_poses: np.ndarray,
    ) -> list[FoundationPoseScore]:
        started = time.perf_counter()
        try:
            scores, _ = self._scorer.predict(
                mesh=estimator.mesh,
                rgb=rgb,
                depth=depth,
                K=K,
                ob_in_cams=np.asarray(centered_poses).reshape(-1, 4, 4),
                normal_map=None,
                mesh_tensors=estimator.mesh_tensors,
                glctx=self._glctx,
                mesh_diameter=estimator.diameter,
            )
            if len(scores) != len(centered_poses):
                raise ValueError("FoundationPose score count differs from candidate count")
            logits = [self._raw_score_logit(score) for score in scores]
            elapsed = 1000 * (time.perf_counter() - started)
            status = "single_pose_rank" if len(logits) == 1 else "candidate_rank"
            return [FoundationPoseScore(logit, status, elapsed / len(logits)) for logit in logits]
        except Exception as exc:  # scorer failure must not drop a valid pose
            logging.getLogger(__name__).warning("FoundationPose track scoring failed: %s", exc)
            elapsed = 1000 * (time.perf_counter() - started)
            return [FoundationPoseScore(None, "error", elapsed / len(centered_poses), str(exc))
                    for _ in centered_poses]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.foundationpose_root.is_dir():
            raise FileNotFoundError(
                f"FoundationPose source directory not found: {self.foundationpose_root}"
            )
        self.warp_cache_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WARP_CACHE_PATH", str(self.warp_cache_path))
        root_str = str(self.foundationpose_root)
        mycpp_build = str(self.foundationpose_root / "mycpp" / "build")
        for path in (mycpp_build, root_str):
            if path not in sys.path:
                sys.path.insert(0, path)
        from estimater import FoundationPose
        from learning.training.predict_pose_refine import PoseRefinePredictor
        from learning.training.predict_score import ScorePredictor
        import nvdiffrast.torch as dr

        # FoundationPose configures the process-wide root logger at INFO during
        # import and emits dozens of lines per frame. Keep backend structured
        # logs readable without editing the vendored source.
        logging.getLogger().setLevel(logging.WARNING)

        self._foundation_pose = FoundationPose
        self._scorer = ScorePredictor()
        self._refiner = PoseRefinePredictor()
        self._glctx = dr.RasterizeCudaContext()
        from Utils import nvdiffrast_render
        self._render_fn = nvdiffrast_render
        self._loaded = True
