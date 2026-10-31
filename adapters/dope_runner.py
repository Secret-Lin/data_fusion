import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np


@dataclass(frozen=True)
class DopeDetection:
    object_name: str  # which object this detection is for
    pose_cam: np.ndarray | None  # 4x4 (camera frame)
    score: float
    vis_bgr: np.ndarray | None
    projected_points: np.ndarray | None  # 9x2 in resized image pixels
    scale_factor: float  # resized = scale_factor * original
    raw_objects: list[dict]


def _ensure_dope_importable(dope_root: str) -> None:
    # DOPE repository layout:
    # - common/{detector.py,cuboid.py,cuboid_pnp_solver.py,models.py,utils.py}
    common_dir = os.path.join(dope_root, "common")
    if not os.path.isdir(common_dir):
        raise FileNotFoundError(f"DOPE common dir not found: {common_dir}")

    if common_dir not in sys.path:
        sys.path.insert(0, common_dir)


def _quat_xyzw_to_R(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _pose_from_location_quat(location_xyz: list[float], quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_xyzw_to_R(quat_xyzw)
    T[:3, 3] = np.array(location_xyz, dtype=np.float64)
    return T


def _as_float(x) -> float | None:
    try:
        if x is None:
            return None
        # torch tensors / numpy scalars often provide .item()
        item = getattr(x, "item", None)
        if callable(item):
            x = item()
        return float(x)
    except Exception:
        return None


def _reproj_error_px(obj: dict) -> float | None:
    """Reprojection error between detected cuboid2d and PnP projected_points."""
    try:
        cub = np.array(obj.get("cuboid2d"), dtype=np.float64).reshape(-1, 2)
        proj = np.array(obj.get("projected_points"), dtype=np.float64).reshape(-1, 2)
    except Exception:
        return None
    n = min(cub.shape[0], proj.shape[0])
    if n == 0:
        return None
    cub = cub[:n]
    proj = proj[:n]
    return float(np.mean(np.linalg.norm(proj - cub, axis=1)))


def _draw_cuboid(out_bgr: np.ndarray, projected_points: np.ndarray) -> None:
    try:
        from cuboid import CuboidLineIndexes
    except Exception:
        return

    pts = np.array(projected_points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 8:
        return

    for a, b in CuboidLineIndexes:
        pa = tuple(np.round(pts[int(a)]).astype(int))
        pb = tuple(np.round(pts[int(b)]).astype(int))
        cv2.line(out_bgr, pa, pb, (0, 255, 0), 2)


class DopeRunner:
    """Multi-object DOPE runner.

    Accepts one or more (object_name, weights_path) pairs and runs all
    networks sequentially per frame, returning a list of DopeDetection.
    """

    def __init__(
        self,
        *,
        dope_root: str,
        objects: list[tuple[str, str]],  # [(object_name, weights_path), ...]
        config_path: str | None = None,
        parallel: bool = False,
        device: str = "auto",
        downscale_height: int | None = None,
    ) -> None:
        _ensure_dope_importable(dope_root)

        import torch
        import yaml

        from cuboid import Cuboid3d
        from cuboid_pnp_solver import CuboidPNPSolver
        from detector import DopeNetwork, ObjectDetector, transform

        self._torch = torch
        self._ObjectDetector = ObjectDetector
        self._transform = transform

        if config_path is None:
            config_path = os.path.join(dope_root, "config", "config_pose.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        # Device
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.input_is_rectified = bool(cfg.get("input_is_rectified", True))
        if downscale_height is None:
            self.downscale_height = int(cfg.get("downscale_height", 0) or 0)
        else:
            self.downscale_height = int(downscale_height)

        # Detection config (shared across all objects)
        self.config_detect = SimpleNamespace()
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = cfg.get("thresh_angle", 0.5)
        self.config_detect.thresh_map = cfg.get("thresh_map", 0.01)
        self.config_detect.sigma = cfg.get("sigma", 3)
        self.config_detect.thresh_points = cfg.get("thresh_points", 0.1)

        # Build one (net, pnp_solver) per object
        self._objects: list[dict] = []
        for object_name, weights_path in objects:
            if object_name not in cfg.get("dimensions", {}):
                raise KeyError(
                    f"Object '{object_name}' not found in DOPE config dimensions. config_path={config_path}"
                )

            net = DopeNetwork().to(self.device)
            state_dict = torch.load(weights_path, map_location=self.device)

            # Strip 'module.' prefix from DataParallel-saved checkpoints
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            state_dict = new_state_dict

            net.load_state_dict(state_dict)
            net.eval()

            pnp_solver = CuboidPNPSolver(
                object_name,
                cuboid3d=Cuboid3d(cfg["dimensions"][object_name]),
            )

            self._objects.append({
                "name": object_name,
                "net": net,
                "pnp_solver": pnp_solver,
                "dimensions": tuple(cfg["dimensions"][object_name]),
            })

    def infer(self, frame_bgr: np.ndarray, camera: dict | None) -> list[DopeDetection]:
        """Run all object networks on the frame. Returns a list of DopeDetection (one per object found)."""
        if frame_bgr is None:
            return []

        K = None
        dist = None
        if isinstance(camera, dict):
            K = camera.get("K")
            dist = camera.get("dist")

        if K is None:
            return []

        K = np.array(K, dtype=np.float64).reshape(3, 3)
        if dist is None:
            dist = np.zeros((5,), dtype=np.float64)
        else:
            dist = np.array(dist, dtype=np.float64).reshape(-1)

        # DOPE inference scripts feed RGB images.
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Downscale for speed (same behavior as DOPE inference.py)
        camera_matrix = K.copy()
        dist_coeffs = dist.copy().reshape(-1, 1)
        h, w, _ = img.shape
        scaling_factor = 1.0
        if self.downscale_height > 0:
            scaling_factor = float(self.downscale_height) / float(h)
            if scaling_factor < 1.0:
                camera_matrix[:2] *= scaling_factor
                img = cv2.resize(img, (int(scaling_factor * w), int(scaling_factor * h)))
            else:
                scaling_factor = 1.0

        import io
        import contextlib

        torch = self._torch
        image_tensor = self._transform(img)
        image_torch_base = image_tensor.to(self.device).unsqueeze(0)

        def _build_detection(obj_cfg: dict) -> DopeDetection | None:
            """Run a single object's network and return DopeDetection or None."""
            net = obj_cfg["net"]
            pnp_solver = obj_cfg["pnp_solver"]
            obj_name = obj_cfg["name"]

            pnp_solver.set_camera_intrinsic_matrix(np.matrix(camera_matrix, dtype="float64"))
            pnp_solver.set_dist_coeffs(np.matrix(dist_coeffs, dtype="float64"))

            with torch.no_grad():
                out, seg = net(image_torch_base)

            vertex2 = out[-1][0]
            aff = seg[-1][0]

            with contextlib.redirect_stdout(io.StringIO()):
                detected_objects = self._ObjectDetector.find_object_poses(
                    vertex2, aff, pnp_solver, self.config_detect
                )

            if not detected_objects:
                return None

            # Pick best
            best = None
            best_score = -1.0
            for obj in detected_objects:
                s = self._score_from_obj(obj)
                if s > best_score:
                    best = obj
                    best_score = s

            pose_cam = None
            projected_points = None
            if best is not None and best.get("location") is not None and best.get("quaternion") is not None:
                quat = np.array(best["quaternion"], dtype=np.float64).reshape(4)
                pose_cam = _pose_from_location_quat(best["location"], quat)
            if best is not None and best.get("projected_points") is not None:
                try:
                    projected_points = np.array(best["projected_points"], dtype=np.float64).reshape(-1, 2)
                except Exception:
                    projected_points = None

            return DopeDetection(
                object_name=obj_name,
                pose_cam=pose_cam,
                score=float(best_score if best_score >= 0.0 else 0.0),
                vis_bgr=None,  # drawn later per-object
                projected_points=projected_points,
                scale_factor=float(scaling_factor),
                raw_objects=detected_objects,
            )

        results: list[DopeDetection] = []
        for obj_cfg in self._objects:
            det = _build_detection(obj_cfg)
            if det is not None and det.pose_cam is not None:
                results.append(det)

        # If nothing detected, build a no-detection vis once
        if not results:
            vis_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(
                vis_bgr,
                "DOPE: no detection",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            # Return single no-detection result
            results.append(DopeDetection(
                object_name="",
                pose_cam=None,
                score=0.0,
                vis_bgr=vis_bgr,
                projected_points=None,
                scale_factor=scaling_factor,
                raw_objects=[],
            ))

        return results

    @staticmethod
    def _score_from_obj(obj: dict) -> float:
        """Compute a quality score for a DOPE detection object."""
        # Prefer scalar confidence if DOPE provides it.
        s = _as_float(obj.get("confidence"))
        if s is None:
            s = _as_float(obj.get("score"))
        if s is not None and s > 0.0:
            return float(s)

        # Fallback: convert reprojection error into (0,1].
        err = _reproj_error_px(obj)
        if err is None:
            return 0.0

        sigma = 5.0  # pixels; tune if needed
        return float(np.exp(-max(err, 0.0) / sigma))
