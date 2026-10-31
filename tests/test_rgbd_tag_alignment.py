import numpy as np
from scipy.spatial.transform import Rotation

from common.rgbd_tag_alignment import (
    align_common_tag_poses,
    rigid_align_points,
)


def make_transform(rotation_deg: float, translation: list[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", rotation_deg, degrees=True).as_matrix()
    transform[:3, 3] = translation
    return transform


def test_rigid_align_points_recovers_known_transform() -> None:
    source = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.1, 0.2, 0.1]]
    )
    expected = make_transform(35.0, [0.2, -0.1, 0.4])
    target = (expected[:3, :3] @ source.T).T + expected[:3, 3]
    result = rigid_align_points(source, target)
    assert np.allclose(result.transform, expected, atol=1e-10)
    assert result.p95_residual_m < 1e-10


def test_common_tag_pose_alignment_ignores_cube_face_model() -> None:
    target_source = make_transform(-22.0, [0.5, 0.1, -0.2])
    source_tags = {
        2: make_transform(10.0, [0.1, 0.2, 0.8]),
        5: make_transform(80.0, [-0.2, 0.1, 0.9]),
    }
    target_tags = {
        tag_id: target_source @ pose
        for tag_id, pose in source_tags.items()
    }
    result, quality = align_common_tag_poses(source_tags, target_tags)
    assert np.allclose(result, target_source, atol=1e-10)
    assert quality["common_tag_ids"] == [2, 5]
    assert quality["max_rotation_disagreement_deg"] < 1e-9
