import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.run_real_frontend import camera_world, load_seeds, seed_camera_pose, sha


def test_seed_uses_first_native_frame_not_later_better_score(tmp_path):
    session = tmp_path / 'capture'
    seed = tmp_path / 'seed'
    seed.mkdir()
    calibration = tmp_path / 'calibration.json'
    calibration.write_text('{}')
    (seed / 'manifest.json').write_text(json.dumps({'status': 'completed', 'session': str(session),
        'calibration_sha256': sha(calibration), 'GT_used_for_initialization_or_fusion': False}))
    rows = [{'camera_id': 'a', 'capture_time': t, 'source_frame_index': i,
             'object_name': 'TomatoSauce', 'source': 'realsense/a/foundationpose/TomatoSauce',
             'pose_world': np.eye(4).tolist(), 'score': score}
            for t, i, score in [(1.1, 11, 100), (1., 10, .1), (.9, 9, 1000)]]
    (seed / 'observations.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    seeds, _ = load_seeds(seed, session, calibration, start=1., duration=.2)
    assert seeds['a']['source_frame_index'] == 10
    calibration.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='calibrations differ'):
        load_seeds(seed, session, calibration)


def test_initial_camera_pose_reconstructs_world_pose_without_mesh_basis_change():
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler('xyz', [20, 30, 40], degrees=True).as_matrix()
    camera[:3, 3] = [.2, -.1, .3]
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler('xyz', [10, -20, 70], degrees=True).as_matrix()
    world[:3, 3] = [.4, .1, .7]
    row = {'source_frame_index': 3, 'capture_time': .125, 'pose_world': world}
    pose = seed_camera_pose(row, 3, .125, camera)
    assert np.allclose(camera @ pose, world, atol=1e-12)
    with pytest.raises(ValueError, match='exact source frame'):
        seed_camera_pose(row, 4, .125, camera)
    with pytest.raises(ValueError, match='exact source frame'):
        seed_camera_pose(row, 3, .126, camera)


def test_fixed_camera_extrinsic_is_not_reinterpreted_as_unity_pose():
    transform = np.eye(4)
    transform[:3, 3] = [.2, -.3, 1.]
    actual = camera_world(None, 0, 'a', {'a': {'T_world_cam': transform.tolist()}})
    assert np.array_equal(actual, transform)


def test_moving_camera_uses_recorded_pose_and_calibrated_tracking_world(monkeypatch):
    import tools.run_real_frontend as module
    tracking = np.eye(4)
    tracking[0, 3] = .5
    calibrated = np.eye(4)
    calibrated[1, 3] = .2
    class Handle(dict):
        attrs = {'coordinate_system': 'unity_cam'}
    handle = Handle(camera_position=[[1, 2, 3]], camera_rotation_xyzw=[[0, 0, 0, 1]])
    seen = []
    def convert(p, q, convention):
        seen.append((p, q, convention))
        return tracking
    monkeypatch.setattr(module, '_tracking_camera', convert)
    result = camera_world(handle, 0, 'quest', {'quest': {'T_world_tracking': calibrated}})
    assert np.allclose(result, calibrated @ tracking)
    assert seen == [([1, 2, 3], [0, 0, 0, 1], 'unity_cam')]


def test_export_retains_native_geometry_and_creates_regular_texture_file(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from PIL import Image
    import trimesh
    import tools.run_real_frontend as module
    source = tmp_path / 'source.obj'
    source.write_text('source fixture')
    mesh = trimesh.creation.box(extents=[.05, .08, .10])
    mesh.apply_translation([.02, .03, .04])
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)), material=trimesh.visual.material.SimpleMaterial(
            image=Image.new('RGB', (4, 4), color='red'), name='textured:material_0'))
    original = mesh.bounds.copy()
    class Repository:
        def __init__(self, *args):
            pass
        def load_mesh(self, name):
            return SimpleNamespace(mesh_path=source, scale_to_m=.01), mesh.copy()
    monkeypatch.setattr(module, 'HopeModelRepository', Repository)
    path, bounds, meta = module.export_mesh(tmp_path / 'cache')
    assert path.is_absolute()
    assert np.array_equal(bounds, original)
    assert (path.parent / 'TomatoSauce.png').stat().st_size > 0
    assert 'textured:' not in path.read_text()
    loaded = trimesh.load(path, force='mesh', process=False)
    assert np.allclose(loaded.bounds, original)
    assert meta['coordinate_policy'] == 'native HOPE mesh basis and origin retained'
