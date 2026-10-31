"""Hand-built solver regressions; these are not H2O experiment observations."""
import asyncio

import gtsam
import numpy as np
import pytest

from config import FusionConfig
from experiments.h2o.ablation import VARIANTS, Variant, build_config, run_fusion
from managers.fusion_manager import HeteroFusionSystem, Observation


def conflicting_sequence() -> list[dict]:
    rows = []
    for frame in range(16):
        for camera in range(3):
            pose = np.eye(4)
            pose[:3, 3] = [0.01 * frame + 0.001 * camera, 0.03, 0.7]
            angle = 0.01 * frame + 0.005 * camera
            if camera == 0:
                pose[0, 3] += 0.4
                angle += 2.8
            pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
            rows.append({"frame_id": frame, "camera_id": f"cam{camera}",
                         "source": f"h2o/cam{camera}/foundationpose/test", "object_name": "test",
                         "t_host": frame / 30, "arrival_time": frame / 30,
                         "score": 0.9, "pose_world": pose.tolist()})
    return rows


def test_provisional_wrong_first_sensor_does_not_change_final_solution() -> None:
    rows = conflicting_sequence()
    forward, _ = asyncio.run(run_fusion(rows, VARIANTS["revision_checkpoint"], fps=30))
    reverse, _ = asyncio.run(run_fusion(rows, VARIANTS["revision_checkpoint_reverse"], fps=30))
    assert len(forward) == len(reverse) == 16
    for a, b in zip(forward, reverse):
        assert a["active_sensors"] == b["active_sensors"] == ["h2o/cam1", "h2o/cam2"]
        np.testing.assert_allclose(a["pose_world"], b["pose_world"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(a["covariance"], b["covariance"], atol=1e-9, rtol=0)


def _queue(system: HeteroFusionSystem, k: int, x: float, *, motion=None) -> None:
    pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(x, 0, 1))
    obs = Observation(k, "camera", "world", pose)
    system.revise_world_state(k, [obs], pose, motion)


def test_offline_default_checkpoint_preserves_live_default_and_no_huber_scope():
    assert not FusionConfig().enable_world_revision_checkpoint
    assert build_config(VARIANTS["full"]).fusion.enable_world_revision_checkpoint
    assert not build_config(VARIANTS["full"], {"enable_world_revision_checkpoint": False}).fusion.enable_world_revision_checkpoint
    assert build_config(VARIANTS["no_huber"]).fusion.robust_huber_k == 0


def test_no_huber_default_is_stable_and_order_independent():
    rows = conflicting_sequence()
    forward, _ = asyncio.run(run_fusion(rows, VARIANTS["no_huber"], fps=30))
    reverse, _ = asyncio.run(run_fusion(rows, Variant(
        "reverse_no_huber", "test", {"robust_huber_k": 0}, reverse_order=True), fps=30))
    assert len(forward) == len(reverse) == 16
    for a, b in zip(forward, reverse):
        np.testing.assert_allclose(a["pose_world"], b["pose_world"], atol=1e-9, rtol=0)
        assert np.linalg.norm(np.asarray(a["pose_world"])[:3, 3]) < 2


def _system() -> HeteroFusionSystem:
    return HeteroFusionSystem(FusionConfig(enable_world_revision_checkpoint=True))


def test_checkpoint_revisions_keep_history_and_motion_without_duplicate_factors() -> None:
    system = _system()
    for k in range(12):
        motion = (k - 1, gtsam.Pose3(), 1 / 30) if k else None
        for offset in (0.2, 0.01, 0.0):
            _queue(system, k, k * 0.01 + offset, motion=motion)
            system.update()
            # Each single input can be queried immediately; no synchronization
            # barrier is introduced by the checkpoint path.
            assert system.current_values().exists(system.tool_key(k))
        assert system.isam.getFactorsUnsafe().size() == 2 * k + 1
        checkpoint_k, checkpoint = system._world_revision_checkpoint
        assert checkpoint_k == k
        assert not checkpoint.calculateEstimate().exists(system.tool_key(k))
        if k:
            assert checkpoint.calculateEstimate().exists(system.tool_key(k - 1))
    assert system.current_values().size() == 12


def test_late_revision_cannot_remove_newer_state() -> None:
    system = _system()
    for k in (0, 1, 0, 0, 1):
        _queue(system, k, 0.01 * k)
        system.update()
    assert system.current_values().size() == 2
    assert system._world_revision_checkpoint is None
    for k in (0, 1):
        assert system.current_values().exists(system.tool_key(k))


def test_batched_states_disable_checkpoint_without_losing_values() -> None:
    system = _system()
    _queue(system, 0, 0.0)
    _queue(system, 1, 0.01, motion=(0, gtsam.Pose3(), 1 / 30))
    system.update()
    assert system.current_values().size() == 2
    assert system.isam.getFactorsUnsafe().size() == 3
    assert system._world_revision_checkpoint is None


def test_unrelated_graph_update_is_never_rewound() -> None:
    system = _system()
    _queue(system, 0, 0.0)
    system.update()
    camera_key = gtsam.symbol("c", 0)
    pose = gtsam.Pose3()
    system.new_values.insert(camera_key, pose)
    system.new_factors.add(gtsam.PriorFactorPose3(
        camera_key, pose, gtsam.noiseModel.Isotropic.Sigma(6, 0.1)))
    _queue(system, 0, 0.01)
    system.update()
    assert system._world_revision_checkpoint is None
    _queue(system, 0, 0.02)
    system.update()
    assert system.current_values().exists(camera_key)
    np.testing.assert_allclose(system.current_values().atPose3(camera_key).matrix(), np.eye(4))


def test_reset_invalidates_checkpoint() -> None:
    system = _system()
    _queue(system, 0, 0.0)
    system.update()
    assert system._world_revision_checkpoint is not None
    system.reset_pending()
    assert system._world_revision_checkpoint is None


def test_failed_rewind_preserves_last_published_solver(monkeypatch) -> None:
    system = _system()
    _queue(system, 0, 0.01)
    system.update()
    before = system.estimate_tool_pose(0).copy()
    indices = list(system._world_factor_indices[0])

    class FailedSolver:
        def update(self, *args):
            raise RuntimeError("test solver failure")

    monkeypatch.setattr(gtsam, "ISAM2", lambda other: FailedSolver())
    _queue(system, 0, 0.02)
    with pytest.raises(RuntimeError, match="test solver failure"):
        system.update()
    system.reset_pending()
    np.testing.assert_array_equal(system.estimate_tool_pose(0), before)
    assert system._world_factor_indices[0] == indices
