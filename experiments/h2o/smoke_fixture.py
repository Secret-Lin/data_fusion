from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _write_pose(path: Path, matrix: np.ndarray, object_id: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = matrix.reshape(-1)
    if object_id is not None:
        values = np.concatenate([[object_id], values])
    np.savetxt(path, values.reshape(1, -1), fmt="%.10f")


def create_smoke_sequence(root: str | Path, *, frames: int = 60) -> Path:
    """Create a tiny annotation-only H2O-compatible take for CI/smoke tests."""
    root_path = Path(root).expanduser().resolve()
    sequence_dir = root_path / "subject_smoke" / "h1" / "0"

    fixed_camera_poses: dict[str, np.ndarray] = {}
    for camera_id, xyz in {
        "cam0": (-0.7, 0.0, 0.1),
        "cam1": (-0.25, 0.1, -0.15),
        "cam2": (0.65, -0.05, 0.2),
        "cam3": (0.25, 0.08, -0.2),
    }.items():
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = xyz
        fixed_camera_poses[camera_id] = pose

    for frame_id in range(int(frames)):
        phase = frame_id / max(frames - 1, 1)
        T_world_object = np.eye(4, dtype=np.float64)
        angle = 0.35 * np.sin(phase * 2.0 * np.pi)
        cosine, sine = np.cos(angle), np.sin(angle)
        T_world_object[:3, :3] = [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ]
        T_world_object[:3, 3] = [
            0.12 * np.sin(phase * 2.0 * np.pi),
            0.04 * np.cos(phase * 2.0 * np.pi),
            0.85 + 0.06 * phase,
        ]

        mobile_pose = np.eye(4, dtype=np.float64)
        mobile_pose[:3, 3] = [
            0.25 * np.cos(phase * 2.0 * np.pi),
            -0.15,
            0.15 * np.sin(phase * 2.0 * np.pi),
        ]
        camera_poses = {**fixed_camera_poses, "cam4": mobile_pose}

        for camera_id, T_world_camera in camera_poses.items():
            T_camera_object = np.linalg.inv(T_world_camera) @ T_world_object
            stem = f"{frame_id:06d}.txt"
            _write_pose(
                sequence_dir / camera_id / "cam_pose" / stem,
                T_world_camera,
            )
            _write_pose(
                sequence_dir / camera_id / "obj_pose_RT" / stem,
                T_camera_object,
                object_id=6,
            )
    return sequence_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny H2O-format smoke fixture")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    print(create_smoke_sequence(args.output, frames=args.frames))


if __name__ == "__main__":
    main()
