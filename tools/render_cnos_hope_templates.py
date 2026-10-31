from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image


def as_mesh(value: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Trimesh):
        return value
    meshes = [item for item in value.geometry.values() if isinstance(item, trimesh.Trimesh)]
    if not meshes:
        raise RuntimeError("CAD file contains no triangle mesh")
    return trimesh.util.concatenate(meshes)


def render_object(
    mesh_path: Path,
    output_dir: Path,
    poses: np.ndarray,
    *,
    scale_to_m: float,
    radius_m: float,
) -> None:
    mesh = as_mesh(trimesh.load(mesh_path, process=False))
    mesh.apply_scale(scale_to_m)
    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    recenter = np.eye(4, dtype=np.float64)
    recenter[:3, 3] = -center

    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    camera_pose = np.diag([1.0, -1.0, -1.0, 1.0])
    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        ambient_light=np.array([0.45, 0.45, 0.45, 1.0]),
    )
    camera = pyrender.IntrinsicsCamera(
        fx=572.4114,
        fy=573.57043,
        cx=325.2611,
        cy=242.04899,
        znear=0.05,
        zfar=5.0,
    )
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    scene.add(light, pose=camera_pose)
    node = scene.add(render_mesh, pose=np.eye(4), name="hope_cad")
    renderer = pyrender.OffscreenRenderer(640, 480)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, source_pose in enumerate(poses):
            pose = np.asarray(source_pose, dtype=np.float64).copy()
            direction = pose[:3, 3]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                raise RuntimeError(f"Invalid CNOS template pose {index}")
            pose[:3, 3] = direction / norm * radius_m
            scene.set_pose(node, pose @ recenter)
            rgba, _depth = renderer.render(
                scene, flags=pyrender.RenderFlags.RGBA
            )
            Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").save(
                output_dir / f"{index:06d}.png"
            )
    finally:
        renderer.delete()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render CNOS onboarding templates from project HOPE CAD meshes"
    )
    parser.add_argument("--models-root", type=Path, default=Path("assets/models/hope_original/cans"))
    parser.add_argument("--objects", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/cnos_templates/hope_cans"))
    parser.add_argument("--scale-to-m", type=float, default=0.01)
    parser.add_argument("--radius-m", type=float, default=0.4)
    parser.add_argument(
        "--poses",
        type=Path,
        default=Path("thirdparty/CNOS/src/poses/predefined_poses/obj_poses_level0.npy"),
    )
    args = parser.parse_args()
    poses = np.load(args.poses)
    for object_name in [item.strip() for item in args.objects.split(",") if item.strip()]:
        mesh_path = args.models_root / object_name / "google_16k" / "textured_simple.obj"
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        render_object(
            mesh_path,
            args.output_dir / object_name,
            poses,
            scale_to_m=float(args.scale_to_m),
            radius_m=float(args.radius_m),
        )
        print(f"rendered {object_name}: {len(poses)} templates")


if __name__ == "__main__":
    main()
