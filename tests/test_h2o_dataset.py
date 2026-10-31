from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.h2o.dataset import H2OSequence


def _write_pose(path: Path, matrix: np.ndarray, object_id: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = matrix.reshape(-1)
    if object_id is not None:
        values = np.concatenate([[object_id], values])
    np.savetxt(path, values.reshape(1, -1))


def test_h2o_sequence_converts_camera_pose_to_world(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "subject1" / "h1" / "0"
    expected = np.eye(4)
    expected[:3, 3] = [0.4, -0.2, 1.1]

    for index, camera in enumerate(("cam0", "cam2", "cam4")):
        T_world_camera = np.eye(4)
        T_world_camera[0, 3] = index * 0.1
        T_camera_object = np.linalg.inv(T_world_camera) @ expected
        _write_pose(
            sequence_dir / camera / "cam_pose" / "000000.txt",
            T_world_camera,
        )
        folder = "obj_pose_rt" if camera == "cam2" else "obj_pose_RT"
        _write_pose(
            sequence_dir / camera / folder / "000000.txt",
            T_camera_object,
            object_id=6,
        )

    sequence = H2OSequence(sequence_dir, ["cam0", "cam2", "cam4"])
    assert H2OSequence.discover_cameras(sequence_dir) == ["cam0", "cam2", "cam4"]
    assert sequence.common_frame_ids() == [0]
    for camera in sequence.cameras:
        sample = sequence.load(camera, 0)
        assert sample.object_name == "cocoa"
        np.testing.assert_allclose(sample.T_world_object, expected, atol=1e-10)
