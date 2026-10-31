import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from common.replay_extrinsics import (
    ReplayExtrinsicError,
    ensure_replay_extrinsics,
    load_fixed_extrinsics,
    load_quest_extrinsic,
)


CUBE_MODEL = Path(__file__).resolve().parents[1] / "standalone" / "cube_model.yaml"


def _write_fixed_replay(session: Path, frames: np.ndarray) -> None:
    (session / "rgbd").mkdir()
    with h5py.File(session / "rgbd" / "cam.h5", "w") as stream:
        stream.create_dataset("color", data=frames)
        stream.create_dataset(
            "depth", data=np.full(frames.shape[:3], 1000, dtype=np.uint16)
        )
        stream.attrs["color_format"] = "bgr8"
    height, width = frames.shape[1:3]
    K = [[800.0, 0.0, width / 2], [0.0, 800.0, height / 2], [0.0, 0.0, 1.0]]
    (session / "camera_metadata.json").write_text(
        json.dumps({"cam": {"K": K, "dist": [0.0] * 5}}), encoding="utf-8"
    )


def test_external_y_up_files_are_converted_at_ingress(tmp_path: Path) -> None:
    transform = np.eye(4)
    transform[1, 3] = 2.0
    fixed_path = tmp_path / "fixed.json"
    fixed_path.write_text(
        json.dumps(
            {
                "coordinate_system": "y_up",
                "T_world_cam": {"cam": transform.tolist()},
            }
        ),
        encoding="utf-8",
    )
    quest_path = tmp_path / "quest.json"
    quest_path.write_text(
        json.dumps(
            {
                "coordinate_system": "unity",
                "T_world_tracking": transform.tolist(),
            }
        ),
        encoding="utf-8",
    )

    assert np.allclose(load_fixed_extrinsics(fixed_path, ["cam"])["cam"][1, 3], -2.0)
    assert np.allclose(load_quest_extrinsic(quest_path)[1, 3], -2.0)


def test_normalized_quest_extrinsic_is_preferred_over_legacy_recorder_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "quest").mkdir()
    with h5py.File(tmp_path / "quest" / "quest.h5", "w"):
        pass
    (tmp_path / "tag_cube_pose.json").write_text(
        json.dumps(
            {
                "T_world_tracking": {
                    "pos": [9.0, 9.0, 9.0],
                    "rot_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        ),
        encoding="utf-8",
    )
    normalized = np.eye(4)
    normalized[:3, 3] = [1.0, 2.0, 3.0]
    normalized_path = tmp_path / "quest_world_extrinsic.json"
    normalized_path.write_text(
        json.dumps(
            {
                "coordinate_system": "y_down",
                "T_world_tracking": normalized.tolist(),
            }
        ),
        encoding="utf-8",
    )

    result = ensure_replay_extrinsics(tmp_path, cube_model_path=CUBE_MODEL)

    assert result.quest_path == normalized_path
    assert np.allclose(result.quest_world_tracking, normalized)


def test_missing_fixed_extrinsic_is_generated_from_all_replay_frames(
    tmp_path: Path,
) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 160)
    frame = np.full((400, 400, 3), 255, dtype=np.uint8)
    frame[120:280, 120:280] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    _write_fixed_replay(tmp_path, np.repeat(frame[None], 3, axis=0))

    result = ensure_replay_extrinsics(tmp_path, cube_model_path=CUBE_MODEL)

    assert result.generated == (tmp_path / "tag_cube_extrinsics.json",)
    assert "cam" in result.fixed
    assert np.allclose(result.fixed["cam"][:3, :3], np.eye(3), atol=1e-3)
    payload = json.loads(result.fixed_path.read_text(encoding="utf-8"))
    assert payload["coordinate_system"] == "y_down"
    assert payload["generated_from"] == "all replay RGB frames with cube_model"


def test_missing_extrinsic_reports_full_sequence_tag_failure(tmp_path: Path) -> None:
    frames = np.zeros((4, 200, 240, 3), dtype=np.uint8)
    _write_fixed_replay(tmp_path, frames)

    with pytest.raises(ReplayExtrinsicError) as error:
        ensure_replay_extrinsics(tmp_path, cube_model_path=CUBE_MODEL)

    message = str(error.value)
    assert "cam" in message
    assert "full sequence" in message.lower()
    assert "no AprilTag" in message
    assert not (tmp_path / "tag_cube_extrinsics.json").exists()


def test_missing_quest_extrinsic_is_generated_from_full_sequence(
    tmp_path: Path,
) -> None:
    (tmp_path / "quest").mkdir()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 160)
    frame = np.full((400, 400, 3), 255, dtype=np.uint8)
    frame[120:280, 120:280] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    count = 3
    with h5py.File(tmp_path / "quest" / "quest.h5", "w") as stream:
        stream.create_dataset("color", data=np.repeat(frame[None], count, axis=0))
        for name, value in (
            ("fx", 800.0),
            ("fy", 800.0),
            ("cx", 200.0),
            ("cy", 200.0),
        ):
            stream.create_dataset(name, data=np.full(count, value, np.float32))
        stream.create_dataset("camera_position", data=np.zeros((count, 3), np.float32))
        stream.create_dataset(
            "camera_rotation_xyzw",
            data=np.repeat([[0.0, 0.0, 0.0, 1.0]], count, axis=0),
        )
        stream.attrs["color_format"] = "bgr8"
        stream.attrs["coordinate_system"] = "y_down"

    result = ensure_replay_extrinsics(
        tmp_path, cube_model_path=CUBE_MODEL, min_quest_samples=3
    )

    assert result.generated == (tmp_path / "quest_world_extrinsic.json",)
    assert result.quest_world_tracking is not None
    assert np.allclose(
        result.quest_world_tracking[:3, :3], np.eye(3), atol=1e-3
    )
    payload = json.loads(result.quest_path.read_text(encoding="utf-8"))
    assert payload["coordinate_system"] == "y_down"
    assert payload["quality"]["processed_frames"] == count
    assert payload["quality"]["accepted_tag_frames"] == count
