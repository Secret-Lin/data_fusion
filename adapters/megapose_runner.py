"""MegaPose RGB-D temporal refinement, independent state for each camera."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

from adapters.m3t_runner import depth_fit, validate_pose


def prepare_megapose(data_dir: Path | None = None) -> Path:
    if os.environ.get("CONDA_DEFAULT_ENV") != "mr_nav":
        raise RuntimeError("MegaPose requires conda mr_nav")
    root = Path(__file__).resolve().parents[1]
    source = root / "thirdparty/megapose6d/src"
    data = Path(data_dir or root / "artifacts/megapose_data").resolve()
    data.mkdir(parents=True, exist_ok=True)
    os.environ["MEGAPOSE_DATA_DIR"] = str(data)
    # Vendor modules and spawned renderer workers need the official source path.
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return data


class MegaPoseRunner:
    def __init__(self, mesh_path: Path, object_name: str, *, data_dir: Path | None = None,
                 iterations: int = 5, render_workers: int = 1):
        data = prepare_megapose(data_dir)
        import torch
        from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
        from megapose.utils.load_model import load_named_model
        if not torch.cuda.is_available():
            raise RuntimeError("MegaPose requires CUDA")
        for name in ("coarse-rgb-906902141", "refiner-rgbd-288182519"):
            for filename in ("config.yaml", "checkpoint.pth.tar"):
                if not (data / "megapose-models" / name / filename).is_file():
                    raise FileNotFoundError("Download weights: python -m tools.download_megapose")
        self.object_name = object_name
        self.iterations = iterations
        if iterations < 1:
            raise ValueError("iterations must be positive")
        dataset = RigidObjectDataset([RigidObject(label=object_name, mesh_path=Path(mesh_path), mesh_units="m")])
        self.estimator = load_named_model("megapose-1.0-RGBD", dataset, n_workers=render_workers, bsz_images=16)
        self.states = {}
        self.intrinsics = {}

    def _seed(self, camera, color, depth_m, K, initial_pose):
        if (color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3
                or depth_m.shape != color.shape[:2] or not np.isfinite(depth_m).all()
                or (depth_m < 0).any() or K.shape != (3, 3) or not np.isfinite(K).all()
                or K[0, 0] <= 0 or K[1, 1] <= 0):
            raise ValueError("Expected registered RGB-D and valid K")
        first = camera not in self.states
        if first:
            if initial_pose is None:
                raise ValueError("MegaPose needs an explicit first-frame pose")
            seed = validate_pose(initial_pose)
        else:
            if initial_pose is not None:
                raise ValueError("No subsequent annotation/seed inputs allowed")
            if not np.array_equal(self.intrinsics[camera], K):
                raise ValueError("Camera intrinsics changed within take")
            seed = self.states[camera]
        return first, seed.copy()

    def process(self, camera: str, color: np.ndarray, depth_m: np.ndarray, K: np.ndarray,
                initial_pose: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        import pandas as pd
        import torch
        from megapose.inference.types import ObservationTensor
        from megapose.panda3d_renderer import Panda3dLightData
        from megapose.utils.tensor_collection import PandasTensorCollection

        first, seed = self._seed(camera, color, depth_m, K, initial_pose)
        # Match M3T protocol: output shared initial pose unchanged on frame zero.
        pose = seed
        with torch.inference_mode():
            if not first:
                observation = ObservationTensor.from_numpy(np.ascontiguousarray(color[..., ::-1]), depth_m, K).cuda()
                estimates = PandasTensorCollection(
                    infos=pd.DataFrame([{"label": self.object_name, "batch_im_id": 0,
                                         "instance_id": 0, "hypothesis_id": 0}]),
                    poses=torch.as_tensor(seed[None], dtype=torch.float32, device="cuda"))
                result, _ = self.estimator.run_inference_pipeline(
                    observation, coarse_estimates=estimates, n_refiner_iterations=self.iterations)
                pose = validate_pose(result.poses[0].cpu().numpy())
            rendered = self.estimator.refiner_model.renderer.render(
                labels=[self.object_name], TCO=torch.as_tensor(pose[None], dtype=torch.float32, device="cuda"),
                K=torch.as_tensor(K[None], dtype=torch.float32, device="cuda"),
                light_datas=[[Panda3dLightData(light_type="ambient", color=(1, 1, 1, 1))]],
                resolution=color.shape[:2], render_depth=True)
            quality = depth_fit(depth_m, rendered.depths[0, 0].cpu().numpy())
            quality["q_fit"] = float(quality["depth_inlier_fraction"] or 0.0)
            quality["q_area"] = float(quality["observed_overlap_fraction"] or 0.0)
            if not first:
                quality["megapose_pose_score"] = float(result.infos.iloc[0]["pose_score"])
        self.states[camera] = pose.copy()
        self.intrinsics[camera] = K.copy()
        return pose, quality

    def close(self):
        renderers = {id(m.renderer): m.renderer for m in (self.estimator.coarse_model, self.estimator.refiner_model)}
        for renderer in renderers.values():
            renderer.stop()
