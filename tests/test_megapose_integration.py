import asyncio

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from adapters.megapose_runner import MegaPoseRunner
from adapters.megapose_transform import Transform
from experiments.h2o.ablation import VARIANTS, run_fusion


def test_portable_transform_composition_inverse_and_xyzw():
    a, b = np.eye(4), np.eye(4)
    a[:3, :3] = Rotation.from_euler('xyz', [0.3, -0.4, 0.7]).as_matrix()
    a[:3, 3] = [0.1, 0.2, 0.9]
    b[:3, :3] = Rotation.from_euler('z', 1.2).as_matrix()
    b[:3, 3] = [-1, 2, 0.5]
    assert np.allclose((Transform(a) * Transform(b)).matrix, a @ b)
    assert np.allclose(Transform(a).inverse().matrix, np.linalg.inv(a))
    assert np.allclose(Transform(Transform(a).quaternion.coeffs(), a[:3, 3]).matrix, a)


def test_megapose_seed_is_per_camera_and_rejects_future_gt():
    runner = MegaPoseRunner.__new__(MegaPoseRunner)
    T = np.eye(4)
    T[2, 3] = 0.5
    runner.states = {'cam0': T.copy()}
    runner.intrinsics = {'cam0': np.eye(3)}
    color, depth, K = np.zeros((2, 3, 3), np.uint8), np.ones((2, 3)), np.eye(3)
    first, seed = runner._seed('cam0', color, depth, K, None)
    assert not first and np.array_equal(seed, T)
    seed[2, 3] = 999
    assert runner.states['cam0'][2, 3] == 0.5
    with pytest.raises(ValueError, match='subsequent'):
        runner._seed('cam0', color, depth, K, T)
    with pytest.raises(ValueError, match='first-frame'):
        runner._seed('cam1', color, depth, K, None)
    with pytest.raises(ValueError, match='intrinsics'):
        runner._seed('cam0', color, depth, K * 2, None)


def test_megapose_routing_preserves_existing_m3t_fusion_math():
    rows = []
    for frame in range(3):
        for camera in range(3):
            pose = np.eye(4)
            pose[:3, 3] = [frame * 0.01 + camera * 0.001, 0, 0.5]
            rows.append(dict(frame_id=frame, camera_id=f'cam{camera}', object_name='book',
                             source=f'h2o/cam{camera}/m3t/book', detector='m3t',
                             t_host=frame / 30, arrival_time=frame / 30, score=1., pose_world=pose.tolist()))
    original, _ = asyncio.run(run_fusion(rows, VARIANTS['full'], fps=30))
    mega = [{**r, 'source': r['source'].replace('/m3t/', '/megapose/'), 'detector': 'megapose'} for r in rows]
    actual, runtime = asyncio.run(run_fusion(mega, VARIANTS['full'], fps=30))
    assert runtime['n_accepted'] == len(rows)
    for a, b in zip(actual, original):
        assert np.array_equal(a['pose_world'], b['pose_world'])
        assert np.array_equal(a['covariance'], b['covariance'])
        assert a['active_sensors'] == b['active_sensors']
