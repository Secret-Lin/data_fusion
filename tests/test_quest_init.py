import json

import cv2
import numpy as np
import yaml
import pytest

from adapters.quest_init import QuestTagInitializer


@pytest.mark.parametrize("dictionary_name", ["DICT_APRILTAG_16h5", "DICT_4X4_50"])
def test_multiframe_quest_init_produces_finite_external_transform(tmp_path, dictionary_name) -> None:
    model_path = tmp_path / "cube_model.yaml"
    model_path.write_text(
        yaml.safe_dump({"dictionary": dictionary_name, "tag_size_m": 0.04, "tags": {0: {"T_world_tag": np.eye(4).tolist()}}}),
        encoding="utf-8",
    )
    initializer = QuestTagInitializer(model_path, min_samples=3)
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name)), 0, 180
    )
    image = np.full((400, 400, 3), 255, np.uint8)
    image[110:290, 110:290] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    meta = {
        "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 200.0, "cy": 200.0},
        "cameraPose": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1]},
    }
    for _ in range(3):
        initializer.add_frame(image, meta)
    result = initializer.finish()
    assert result is not None
    assert result.shape == (4, 4)
    assert np.isfinite(result).all()


def test_quest_init_prefers_common_tag_world_reference(tmp_path) -> None:
    model_path = tmp_path / "cube_model.yaml"
    model_path.write_text(
        yaml.safe_dump(
            {"tag_size_m": 0.04, "tags": {0: {"T_world_tag": np.eye(4).tolist()}}}
        ),
        encoding="utf-8",
    )
    reference = np.eye(4)
    reference[:3, 3] = [0.1, 0.2, 0.3]
    reference_path = tmp_path / "tag_cube_extrinsics.json"
    reference_path.write_text(
        json.dumps(
            {"coordinate_system": "y_down", "T_world_tag": {"2": reference.tolist()}}
        ),
        encoding="utf-8",
    )
    initializer = QuestTagInitializer(
        model_path,
        min_samples=3,
        reference_extrinsics_path=reference_path,
    )
    assert set(initializer.transforms) == {2}
    assert np.allclose(initializer.transforms[2], reference)
    assert initializer.reference_source == str(reference_path)
