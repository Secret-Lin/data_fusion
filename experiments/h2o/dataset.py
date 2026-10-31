from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


H2O_OBJECT_NAMES: dict[int, str] = {
    0: "background",
    1: "book",
    2: "espresso",
    3: "lotion",
    4: "spray",
    5: "milk",
    6: "cocoa",
    7: "chips",
    8: "cappuccino",
}


@dataclass(frozen=True)
class H2OWorldPoseSample:
    frame_id: int
    camera_id: str
    object_id: int
    object_name: str
    T_world_camera: np.ndarray
    T_camera_object: np.ndarray
    T_world_object: np.ndarray


@dataclass(frozen=True)
class H2ORGBDFrame:
    frame_id: int
    camera_id: str
    color_bgr: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray


def _pose_folder(camera_dir: Path, *names: str) -> Path:
    children = {
        child.name.lower(): child
        for child in camera_dir.iterdir()
        if child.is_dir()
    }
    for name in names:
        found = children.get(name.lower())
        if found is not None:
            return found
    expected = ", ".join(names)
    raise FileNotFoundError(f"Missing {expected} below {camera_dir}")


def _frame_files(folder: Path) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for path in folder.glob("*.txt"):
        try:
            frame_id = int(path.stem)
        except ValueError:
            continue
        output[frame_id] = path
    return output


def _load_matrix(path: Path, *, prefix_values: int = 0) -> np.ndarray:
    values = np.asarray(np.loadtxt(path), dtype=np.float64).reshape(-1)
    expected = prefix_values + 16
    if values.size != expected:
        raise ValueError(
            f"Expected {expected} values in {path}, found {values.size}"
        )
    matrix = values[prefix_values:].reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite pose in {path}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"Invalid homogeneous transform in {path}")
    return matrix


class H2OSequence:
    """Read pose annotations from one H2O take.

    ``sequence_dir`` is the directory containing ``cam0`` ... ``cam4``, for
    example ``subject1/h1/0``. H2O stores camera-to-world transforms and
    object-to-camera transforms. The adapter exposes the project convention
    ``T_world_object = T_world_camera @ T_camera_object`` in metres.
    """

    @staticmethod
    def discover_cameras(sequence_dir: str | Path) -> list[str]:
        root = Path(sequence_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"H2O sequence not found: {root}")
        cameras: list[str] = []
        for camera_dir in sorted(root.glob("cam*"), key=lambda path: path.name):
            if not camera_dir.is_dir():
                continue
            try:
                cam_pose = _pose_folder(camera_dir, "cam_pose")
                obj_pose = _pose_folder(camera_dir, "obj_pose_RT", "obj_pose_rt")
            except FileNotFoundError:
                continue
            if _frame_files(cam_pose) and _frame_files(obj_pose):
                cameras.append(camera_dir.name)
        if not cameras:
            raise RuntimeError(f"No valid H2O cameras found below {root}")
        return cameras

    def __init__(self, sequence_dir: str | Path, cameras: Iterable[str]):
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()
        if not self.sequence_dir.is_dir():
            raise FileNotFoundError(f"H2O sequence not found: {self.sequence_dir}")

        self.cameras = tuple(
            dict.fromkeys(str(camera).strip() for camera in cameras if str(camera).strip())
        )
        if not self.cameras:
            raise ValueError("At least one H2O camera is required")

        self._cam_pose_files: dict[str, dict[int, Path]] = {}
        self._obj_pose_files: dict[str, dict[int, Path]] = {}
        self._intrinsics: dict[str, np.ndarray] = {}
        for camera_id in self.cameras:
            camera_dir = self.sequence_dir / camera_id
            if not camera_dir.is_dir():
                raise FileNotFoundError(f"H2O camera not found: {camera_dir}")
            cam_pose_dir = _pose_folder(camera_dir, "cam_pose")
            obj_pose_dir = _pose_folder(camera_dir, "obj_pose_RT", "obj_pose_rt")
            self._cam_pose_files[camera_id] = _frame_files(cam_pose_dir)
            self._obj_pose_files[camera_id] = _frame_files(obj_pose_dir)

    def available_frame_ids(self, camera_id: str) -> list[int]:
        """Frames with both camera and object pose files, including background."""
        if camera_id not in self._cam_pose_files:
            raise KeyError(f"Camera {camera_id!r} is not configured")
        return sorted(set(self._cam_pose_files[camera_id]) & set(self._obj_pose_files[camera_id]))

    def camera_intrinsics(self, camera_id: str) -> np.ndarray:
        """Return H2O's 3x3 RGB/depth intrinsic matrix.

        The official file stores ``fx fy cx cy width height``. H2O RGB and
        depth are both 1280x720, so no resize or crop is applied here.
        """
        if camera_id not in self._cam_pose_files:
            raise KeyError(f"Camera {camera_id!r} is not configured")
        if camera_id in self._intrinsics:
            return self._intrinsics[camera_id].copy()
        values = np.asarray(
            np.loadtxt(self.sequence_dir / camera_id / "cam_intrinsics.txt"),
            dtype=np.float64,
        ).reshape(-1)
        if values.size != 6 or not np.isfinite(values).all():
            raise ValueError(f"Invalid H2O intrinsics for {camera_id}")
        fx, fy, cx, cy, width, height = values
        if int(width) != 1280 or int(height) != 720:
            raise ValueError(
                f"Unexpected H2O resolution for {camera_id}: {int(width)}x{int(height)}"
            )
        matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        matrix.setflags(write=False)
        self._intrinsics[camera_id] = matrix
        return matrix.copy()

    def load_rgbd(self, camera_id: str, frame_id: int) -> H2ORGBDFrame:
        """Load an H2O RGB-D frame without resizing or color/depth cropping."""
        color_path = self.sequence_dir / camera_id / "rgb" / f"{int(frame_id):06d}.png"
        depth_path = self.sequence_dir / camera_id / "depth" / f"{int(frame_id):06d}.png"
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if color is None or depth_raw is None:
            raise FileNotFoundError(f"Missing H2O RGB-D pair: {color_path}, {depth_path}")
        if color.shape[:2] != depth_raw.shape[:2] or color.shape[:2] != (720, 1280):
            raise ValueError(
                f"Unexpected H2O RGB-D shape for {camera_id}/{frame_id}: "
                f"{color.shape}, {depth_raw.shape}"
            )
        depth_m = np.asarray(depth_raw, dtype=np.float32) * 0.001
        depth_m[~np.isfinite(depth_m)] = 0.0
        return H2ORGBDFrame(
            frame_id=int(frame_id),
            camera_id=str(camera_id),
            color_bgr=color,
            depth_m=depth_m,
            K=self.camera_intrinsics(camera_id),
        )

    def common_frame_ids(self) -> list[int]:
        common: set[int] | None = None
        for camera_id in self.cameras:
            available = set(self.available_frame_ids(camera_id))
            common = available if common is None else common & available
        return sorted(common or set())

    def load(self, camera_id: str, frame_id: int) -> H2OWorldPoseSample:
        if camera_id not in self._cam_pose_files:
            raise KeyError(f"Camera {camera_id!r} is not configured")
        cam_path = self._cam_pose_files[camera_id].get(int(frame_id))
        obj_path = self._obj_pose_files[camera_id].get(int(frame_id))
        if cam_path is None or obj_path is None:
            raise FileNotFoundError(
                f"Frame {frame_id:06d} is incomplete for {camera_id}"
            )

        object_values = np.asarray(np.loadtxt(obj_path), dtype=np.float64).reshape(-1)
        if object_values.size != 17:
            raise ValueError(
                f"Expected 17 values in {obj_path}, found {object_values.size}"
            )
        object_id = int(round(float(object_values[0])))
        if object_id not in H2O_OBJECT_NAMES:
            raise ValueError(f"Unknown H2O object id {object_id} in {obj_path}")

        T_world_camera = _load_matrix(cam_path)
        T_camera_object = _load_matrix(obj_path, prefix_values=1)
        T_world_object = T_world_camera @ T_camera_object
        return H2OWorldPoseSample(
            frame_id=int(frame_id),
            camera_id=camera_id,
            object_id=object_id,
            object_name=H2O_OBJECT_NAMES[object_id],
            T_world_camera=T_world_camera,
            T_camera_object=T_camera_object,
            T_world_object=T_world_object,
        )

    def validate(self) -> dict[str, object]:
        frame_ids = self.common_frame_ids()
        if not frame_ids:
            raise RuntimeError(
                f"No common cam_pose/obj_pose_RT frames in {self.sequence_dir}"
            )
        object_ids: set[int] = set()
        for camera_id in self.cameras:
            for frame_id in (frame_ids[0], frame_ids[-1]):
                object_ids.add(self.load(camera_id, frame_id).object_id)
        return {
            "sequence_dir": str(self.sequence_dir),
            "cameras": list(self.cameras),
            "n_common_frames": len(frame_ids),
            "first_frame": frame_ids[0],
            "last_frame": frame_ids[-1],
            "object_ids_at_ends": sorted(object_ids),
        }
