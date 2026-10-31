import numpy as np

import pytest

from core.coordinate_frames import (
    normalize_coordinate_system,
    transform_to_y_down,
    unity_opencv_transform,
)


def test_unity_opencv_transform_is_involution_and_flips_y_translation() -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [1.0, 2.0, 3.0]

    converted = unity_opencv_transform(transform)

    assert np.allclose(converted[:3, 3], [1.0, -2.0, 3.0])
    assert np.allclose(unity_opencv_transform(converted), transform)
    assert np.isclose(np.linalg.det(converted[:3, :3]), 1.0)


def test_external_coordinate_contract_defaults_to_y_down() -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[1, 3] = 2.0

    assert normalize_coordinate_system(None) == "y_down"
    assert np.allclose(transform_to_y_down(transform), transform)
    assert np.allclose(transform_to_y_down(transform, "unity_cam")[1, 3], -2.0)

    with pytest.raises(ValueError, match="Unsupported coordinate system"):
        transform_to_y_down(transform, "unknown")
