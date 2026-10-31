from __future__ import annotations

import numpy as np


# Quest/Unity camera poses use Y-up coordinates, while OpenCV pose estimators
# use Y-down coordinates.  Conjugating a rigid transform changes the basis of
# both its source and target frames while keeping a proper rotation matrix.
UNITY_OPENCV_BASIS = np.diag([1.0, -1.0, 1.0, 1.0])

Y_DOWN = "y_down"
Y_UP = "y_up"


def normalize_coordinate_system(value: object, *, default: str = Y_DOWN) -> str:
    """Normalize external coordinate metadata to ``y_down`` or ``y_up``.

    Missing metadata follows the project contract and defaults to Y-down.
    Unity-labelled inputs are explicitly treated as Y-up.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        text = default
    if text in {"y_down", "ydown", "opencv", "opencv_cam", "camera_opencv"}:
        return Y_DOWN
    if text in {"y_up", "yup", "unity", "unity_cam", "camera_unity"}:
        return Y_UP
    raise ValueError(
        f"Unsupported coordinate system {value!r}; expected Y-down/OpenCV or Y-up/Unity"
    )


def unity_opencv_transform(transform: np.ndarray) -> np.ndarray:
    """Convert a homogeneous transform between Unity and OpenCV bases.

    The conversion is its own inverse. Both the source and target coordinate
    bases are changed, so the operation is ``S @ transform @ S``.
    """
    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return UNITY_OPENCV_BASIS @ value @ UNITY_OPENCV_BASIS


def transform_to_y_down(
    transform: np.ndarray,
    source_coordinate_system: object = Y_DOWN,
) -> np.ndarray:
    """Return a transform in the project's mandatory Y-down convention."""
    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    coordinate_system = normalize_coordinate_system(source_coordinate_system)
    if coordinate_system == Y_DOWN:
        return value.copy()
    return unity_opencv_transform(value)
