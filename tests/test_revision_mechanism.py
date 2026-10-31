"""Constructed unit fixtures, not dataset observations or reported experiments."""
import asyncio
from copy import deepcopy

import gtsam
import numpy as np

from core.pose_consensus import select_pose_consensus
from experiments.h2o.ablation import VARIANTS, run_fusion
from tools.check_revision_mechanism import irreversible_selection, run_control, variant
from tools.run_h2o_foundationpose_ablation import compare_pose_outputs


def test_irreversible_decisions_survive_later_majority():
    poses = [gtsam.Pose3(gtsam.Rot3(), np.array([x, 0., 1.])) for x in (.4, 0., .001)]
    cov = [np.eye(6)*.001]*3
    settings = dict(trans_gate_m=.15, rot_gate_deg=45., nis_threshold=None, huber_k=1.345)
    admitted, decided = set(), set()
    for n in range(1, 4):
        result = irreversible_selection([str(i) for i in range(n)], admitted, decided,
                                         poses[:n], cov[:n], **settings)
        assert result.selected == [0]
    assert set(select_pose_consensus(poses, cov, **settings).selected) == {1, 2}
    assert decided == {"0", "1", "2"}


def test_control_does_not_change_production_or_input():
    rows = []
    for frame in range(3):
        for sensor in range(3):
            pose = np.eye(4)
            pose[:3, 3] = [.4 if sensor == 0 else sensor*.001, .01*frame, 1.]
            rows.append(dict(frame_id=frame, camera_id=f"cam{sensor}",
                             source=f"h2o/cam{sensor}/foundationpose/test", object_name="test",
                             pose_world=pose.tolist(), score=1., t_host=frame/30, arrival_time=frame/30))
    before = deepcopy(rows)
    reference, _ = asyncio.run(run_fusion(rows, VARIANTS["full"], fps=30))
    for mode in ("R0", "R1", "R2"):
        forward, _, failure, trace = asyncio.run(run_control(rows, mode))
        assert failure is None
        assert len(trace) == len(rows)
        if mode == "R0":
            assert not any(r["removed"] for r in trace)
            assert all(r["active_sensors"] == ["h2o/cam0"] for r in forward)
        if mode == "R2":
            assert compare_pose_outputs(reference, forward)["pose_and_covariance_identical"]
    after, _ = asyncio.run(run_fusion(rows, VARIANTS["full"], fps=30))
    assert compare_pose_outputs(reference, after)["pose_and_covariance_identical"]
    assert rows == before


def test_control_pairs_change_only_target_switch():
    assert variant("R0", False, 1.345).overrides == variant("R2", False, 1.345).overrides
    a = variant("R1", False, 1.345).overrides.copy()
    b = variant("R2", False, 1.345).overrides.copy()
    assert a.pop("enable_world_revision_checkpoint") is False
    assert b.pop("enable_world_revision_checkpoint") is True
    assert a == b
