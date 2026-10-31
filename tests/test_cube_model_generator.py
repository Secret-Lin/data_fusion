from pathlib import Path

import numpy as np
import pytest
import yaml

from standalone.cube_model_generator import (
    SUPPORTED_DICTIONARIES,
    build_parser,
    load_model,
    opencv_dictionary,
    save_model,
)


def _model_parts():
    transforms = {index: np.eye(4, dtype=np.float64) for index in range(6)}
    faces = {
        0: "+Z",
        1: "+X",
        2: "+Y",
        3: "-X",
        4: "-Y",
        5: "-Z",
    }
    residuals = {index: 0.0 for index in range(6)}
    return transforms, faces, residuals


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("DICT_APRILTAG_16h5", "DICT_APRILTAG_16h5"),
        ("apriltag_16h5", "DICT_APRILTAG_16h5"),
        ("DICT_4X4_50", "DICT_4X4_50"),
        ("4x4_50", "DICT_4X4_50"),
    ],
)
def test_dictionary_cli_aliases(argument: str, expected: str) -> None:
    args = build_parser().parse_args(
        ["calibrate", "--dictionary", argument]
    )
    assert args.dictionary == expected


@pytest.mark.parametrize("dictionary", SUPPORTED_DICTIONARIES)
def test_selected_dictionary_is_available_in_opencv(dictionary: str) -> None:
    assert opencv_dictionary(dictionary) is not None


def test_model_records_and_validates_selected_dictionary(tmp_path: Path) -> None:
    output = tmp_path / "cube_model.yaml"
    transforms, faces, residuals = _model_parts()
    save_model(
        output,
        transforms,
        faces,
        residuals,
        0.05,
        0.04,
        "test-camera",
        {},
        "DICT_4X4_50",
    )

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["dictionary"] == "DICT_4X4_50"
    _, loaded = load_model(output, "DICT_4X4_50")
    assert set(loaded) == set(range(6))
    with pytest.raises(ValueError, match="does not match"):
        load_model(output, "DICT_APRILTAG_16h5")
