"""Validated dictionaries supported by the laboratory cube models."""
from __future__ import annotations

import cv2


SUPPORTED_CUBE_DICTIONARIES = ("DICT_APRILTAG_16h5", "DICT_4X4_50")


def cube_dictionary(name: str):
    if name not in SUPPORTED_CUBE_DICTIONARIES:
        raise ValueError(
            f"Unsupported cube dictionary {name!r}; expected one of "
            f"{SUPPORTED_CUBE_DICTIONARIES}"
        )
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, name):
        raise RuntimeError(f"OpenCV does not expose cube dictionary {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def validate_cube_axis_metadata(data: dict) -> None:
    """Do not silently treat an explicitly described generator frame as Y-down."""
    axes = str(data.get("coordinate_convention", {}).get("world_axes", ""))
    if data.get("coordinate_system") is None and data.get("coord") is None and "+Y Tag 0 top" in axes:
        raise ValueError(
            "Generator cube describes +Y up but has no coordinate_system; "
            "prepare a separate Y-down model with tools.prepare_generator_cube_model. "
            "Do not only relabel its coordinate_system."
        )
