from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from adapters.foundationpose_runner import FoundationPoseRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Ycbineoat-style RGB-D scene with a HOPE CAD model"
    )
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--object", required=True, dest="object_name")
    parser.add_argument("--output", type=Path, default=Path("results/foundationpose_hope"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--register-iterations", type=int, default=5)
    parser.add_argument("--track-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    scene = args.scene.resolve()
    K = np.loadtxt(scene / "cam_K.txt").reshape(3, 3)
    color_files = sorted((scene / "rgb").glob("*.png"))
    if args.max_frames > 0:
        color_files = color_files[: args.max_frames]
    if not color_files:
        raise FileNotFoundError(f"No RGB PNG files found under {scene / 'rgb'}")

    runner = FoundationPoseRunner(
        repo_root / "thirdparty" / "FoundationPose",
        repo_root / "assets" / "models" / "hope_original" / "cans",
        register_iterations=args.register_iterations,
        track_iterations=args.track_iterations,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for index, color_path in enumerate(color_files):
        depth_path = scene / "depth" / color_path.name
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            raise FileNotFoundError(f"Missing RGB-D pair: {color_path}, {depth_path}")
        mask = None
        if index == 0:
            mask = cv2.imread(str(scene / "masks" / color_path.name), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(
                    f"First-frame mask not found: {scene / 'masks' / color_path.name}"
                )
        estimate = runner.process(
            source="offline",
            object_name=args.object_name,
            color_bgr=color,
            depth_m=depth.astype(np.float32) * 0.001,
            camera_matrix=K,
            mask=mask,
        )
        if estimate is None:
            raise RuntimeError("Registration did not produce a pose")
        np.savetxt(output / f"{color_path.stem}.txt", estimate.pose_cam)
        print(f"{index + 1}/{len(color_files)} {color_path.name}: {estimate.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
