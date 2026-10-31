import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.m3t_runner import M3TRunner, depth_fit, validate_pose
from experiments.h2o.ablation import VARIANTS, run_fusion
from tools.run_h2o_m3t import initial_poses


def test_m3t_depth_quality_counts_missing_and_occluded_pixels():
    quality = depth_fit(np.array([[1, 0], [0.5, 2.0]]), np.ones((2, 2)))
    assert quality["observed_overlap_fraction"] == 0.75
    assert quality["depth_inlier_fraction"] == pytest.approx(1 / 3)
    assert depth_fit(np.ones((2, 2)), np.zeros((2, 2)))["depth_inlier_fraction"] is None
    with pytest.raises(ValueError):
        depth_fit(np.zeros((1, 2)), np.ones((2, 2)))


def test_m3t_seed_loader_never_uses_later_frame_or_world_pose(tmp_path):
    path = tmp_path / "observations.jsonl"
    row = {"frame_id": 3, "camera_id": "cam0", "object_name": "book", "pose_cam": np.eye(4).tolist()}
    later = {**row, "frame_id": 4, "pose_cam": (np.eye(4) * 9).tolist()}
    path.write_text(json.dumps(row) + '\n' + json.dumps(later) + '\n')
    assert np.array_equal(initial_poses(path, 3, ["cam0"], "book")["cam0"], np.eye(4))
    with pytest.raises(ValueError, match="Missing"):
        initial_poses(path, 3, ["cam0", "cam1"], "book")
    with pytest.raises(ValueError):
        initial_poses(path, 3, ["cam0"], "lotion")
    with pytest.raises(ValueError):
        validate_pose(np.eye(4) * 2)


def test_m3t_each_camera_independent_and_no_implicit_reinitialization(tmp_path, monkeypatch):
    class Tracker:
        def __init__(self, mesh, cache, K, color, depth, initial, threads):
            self.pose = initial.copy()
            self.shape = depth.shape
        def track(self, color, depth):
            self.pose[0, 3] += 0.001
            return self.pose.copy()
        def rendered_depth(self):
            return np.ones(self.shape)
    monkeypatch.setattr('adapters.m3t_runner.load_native', lambda: SimpleNamespace(SingleCameraTracker=Tracker))
    mesh = tmp_path / 'mesh.obj'
    mesh.write_text('test')
    runner = M3TRunner(mesh, tmp_path / 'cache')
    color, depth, K = np.zeros((4, 5, 3), np.uint8), np.ones((4, 5)), np.eye(3)
    with pytest.raises(ValueError, match='explicit'):
        runner.process('cam0', color, depth, K)
    runner.process('cam0', color, depth, K, np.eye(4))
    runner.process('cam1', color, depth, K, np.eye(4))
    pose, _ = runner.process('cam0', color, depth, K)
    assert pose[0, 3] == pytest.approx(0.001)
    assert runner.trackers['cam1'].pose[0, 3] == 0
    with pytest.raises(ValueError, match='annotations'):
        runner.process('cam0', color, depth, K, np.eye(4))


def test_m3t_routes_honestly_with_same_starting_noise_without_changing_fp():
    rows = []
    for frame in range(3):
        for camera in range(3):
            T = np.eye(4)
            T[:3, 3] = [0.01 * frame + 0.001 * camera, 0, 0.5]
            rows.append({"frame_id": frame, "t_host": frame / 30, "arrival_time": frame / 30,
                         "camera_id": f"cam{camera}", "object_name": "book", "score": 1.0,
                         "source": f"h2o/cam{camera}/foundationpose/book", "pose_world": T.tolist()})
    fp, _ = asyncio.run(run_fusion(rows, VARIANTS['full'], fps=30))
    m3t = [{**r, 'source': r['source'].replace('/foundationpose/', '/m3t/'), 'detector': 'm3t'} for r in rows]
    actual, _ = asyncio.run(run_fusion(m3t, VARIANTS['full'], fps=30))
    assert len(actual) == len(fp) == 3
    for a, b in zip(actual, fp):
        assert np.array_equal(a['pose_world'], b['pose_world'])
        assert np.array_equal(a['covariance'], b['covariance'])
        assert a['active_sensors'] == b['active_sensors']
