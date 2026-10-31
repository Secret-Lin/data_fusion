import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.run_real_ots_smoke import reference_at, frame_indices


def test_reference_interpolation_and_no_extrapolation():
    poses=np.repeat(np.eye(4)[None],2,axis=0)
    poses[1,0,3]=.1
    poses[1,:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    value=reference_at(np.array([0.,1.]),poses,np.array([.5]))[0]
    assert np.allclose(value[:3,3],[.05,0,0])
    assert np.allclose(value[:3,:3],Rotation.from_euler('z',45,degrees=True).as_matrix())
    with pytest.raises(ValueError,match='extrapolate'):
        reference_at(np.array([0.,1.]),poses,np.array([1.1]))


def test_full_sequence_includes_last_frame_and_rejects_bad_clock():
    times=np.array([10.,10.1,10.2])
    assert frame_indices(times,10,0,None,True).tolist()==[0,1,2]
    assert frame_indices(times,10,0,.2).tolist()==[0,1]
    with pytest.raises(ValueError,match='strictly increasing'):
        frame_indices([1.,1.],0,0,None,True)


def test_cached_analysis_keeps_missing_coverage_and_reference_bounds(tmp_path):
    import json
    from tools.analyze_real_ots_smoke import analyze
    run_dir=tmp_path/'run'
    run_dir.mkdir()
    (run_dir/'manifest.json').write_text(json.dumps({
        'session':'example','origin_host_s':0.,'frames_by_sensor':{'a':4,'b':4}}))
    rows=[{'frame_id':i,'pose_world':np.eye(4).tolist(),'observation_count':1} for i in [-30,0,30]]
    (run_dir/'predictions.json').write_text(json.dumps({
        'fusion':rows,'mean_equal':rows,'sensor_a':rows,'sensor_b':[]}))
    (run_dir/'input_timeline.json').write_text(json.dumps([
        {'camera_id':c,'frame_id':i} for c in ['a','b'] for i in [-30,0,30,60]]))
    np.savez(tmp_path/'example_reference_for_native_mesh.npz',
             candidate_host_time_s=np.array([0.,1.]),
             T_world_ydown_mesh_native_tagcube0_Tomato2=np.repeat(np.eye(4)[None],2,axis=0))
    result=analyze(run_dir,tmp_path)
    assert result['common_targets']==2
    assert result['common_targets_outside_reference']==1
    assert result['missing_sensor_methods']==['sensor_b']
    coverage=json.loads((run_dir/'coverage.json').read_text())
    assert coverage['methods']['sensor_b']['coverage_fraction']==0
    assert coverage['methods']['fusion']['coverage_fraction']==.75


def test_inference_only_never_opens_ots_reference(tmp_path, monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace
    import h5py
    import tools.run_real_ots_smoke as module

    session = tmp_path/'capture'
    session.mkdir()
    calibration = tmp_path/'calibration'
    calibration.mkdir()
    (calibration/'production_checks.json').write_text(json.dumps({
        'capture': {'a': {'T_world_cam': np.eye(4).tolist()}}}))
    (session/'camera_metadata.json').write_text(json.dumps({
        'a': {'K': [[100,0,32],[0,100,32],[0,0,1]], 'dist': [0]*5}}))
    with h5py.File(session/'a.h5', 'w') as handle:
        handle['t_host'] = [10., 10.1]
        handle['color'] = np.zeros((2,64,64,3), dtype=np.uint8)
        handle['depth'] = np.ones((2,64,64), dtype=np.uint16)

    class Manager:
        def __init__(self, bus, cfg, **kwargs):
            self.bus, self.cfg = bus, cfg
            self._T_world_camera = {}
            self._executor = SimpleNamespace(shutdown=lambda **kwargs: None)
        async def _handle_frame(self, packet):
            await self.bus.publish(self.cfg.output_topic, SimpleNamespace(
                source='foundationpose/'+packet.source, pose_world=np.eye(4), score=1.,
                object_name='TomatoSauce', quality_metrics={}))

    async def fusion(observations, *args, **kwargs):
        return observations, {}
    def forbid_reference(*args, **kwargs):
        raise AssertionError('Evaluation reference must not be opened')

    monkeypatch.setattr(module, 'FoundationPoseManager', Manager)
    monkeypatch.setattr(module, 'run_fusion', fusion)
    monkeypatch.setattr(module, 'run_baselines', lambda observations, cameras: {'sensor_a': observations})
    monkeypatch.setattr(module.np, 'load', forbid_reference)
    args = SimpleNamespace(session=session, calibration_dir=calibration, reference_dir=tmp_path/'missing',
                           skip_reference_evaluation=True, output_parent=tmp_path/'out',
                           start=0., duration=1., full_sequence=False)
    output = asyncio.run(module.run(args))
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert manifest['evaluation_status'] == 'reference_pending'
    assert manifest['frames_by_sensor'] == {'a': 2}
    assert not (output/'candidate_reference_errors.json').exists()
