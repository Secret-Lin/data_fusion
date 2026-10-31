import numpy as np
import gtsam
import pytest

from core.pose_consensus import select_pose_consensus


def _pose(x: float) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(x, 0.0, 1.0))


def test_quality_override_can_reject_low_quality_majority_cluster() -> None:
    poses = [_pose(0.0), _pose(0.01), _pose(0.5)]
    covariances = [np.eye(6) * 1e-3 for _ in poses]
    baseline = select_pose_consensus(
        poses, covariances, trans_gate_m=0.05, rot_gate_deg=45.0,
        nis_threshold=None, huber_k=1.345,
        qualities=None, quality_override_ratio=1.5,
    )
    assert baseline.selected == [0, 1]
    override = select_pose_consensus(
        poses, covariances, trans_gate_m=0.05, rot_gate_deg=45.0,
        nis_threshold=None, huber_k=1.345,
        qualities=[0.4, 0.4, 0.9], quality_override_ratio=1.5,
    )
    assert override.selected == [2]


@pytest.mark.parametrize("qualities, expected", [
    ([0.4, 0.4, 0.9, 0.0], [2]),
    ([0.0, 0.0, 0.0, 0.0], [0, 1]),
    ([0.4, 0.4, 0.9, np.nan], [0, 1]),
    ([0.4, 0.4, 0.9, -1.0], [0, 1]),
])
def test_zero_quality_is_negative_evidence_not_missing(qualities, expected):
    poses = [_pose(0.0), _pose(0.01), _pose(0.5), _pose(0.9)]
    result = select_pose_consensus(
        poses, [np.eye(6) * 1e-3 for _ in poses], trans_gate_m=0.05,
        rot_gate_deg=45.0, nis_threshold=None, huber_k=1.345,
        qualities=qualities, quality_override_ratio=1.5,
    )
    assert result.selected == expected


def test_equal_quality_override_candidates_have_canonical_tie_break():
    from itertools import permutations

    original = [_pose(0.0), _pose(0.01), _pose(0.5), _pose(0.9)]
    qualities = [0.3, 0.3, 0.9, 0.9]
    for order in permutations(range(4)):
        result = select_pose_consensus(
            [original[i] for i in order], [np.eye(6) * 1e-3 for _ in order],
            trans_gate_m=0.05, rot_gate_deg=45.0, nis_threshold=None,
            huber_k=1.345, qualities=[qualities[i] for i in order],
            quality_override_ratio=1.5,
        )
        assert [order[i] for i in result.selected] == [2]
