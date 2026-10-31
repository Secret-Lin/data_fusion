"""Deterministic solver tests, not artificial dataset corruption experiments."""
import asyncio

import numpy as np

from config import FusionConfig
from experiments.h2o.ablation import Variant, run_fusion


def row(frame, camera, angle=0., x=0., capture=None, arrival=None, obj='object'):
    pose = np.eye(4)
    c,s = np.cos(angle),np.sin(angle)
    pose[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
    pose[0,3] = x
    return {'frame_id':frame, 'camera_id':camera, 'source':f'h2o/{camera}/foundationpose/{obj}',
            'object_name':obj, 't_host':frame/30 if capture is None else capture,
            'arrival_time':frame/30 if arrival is None else arrival,
            'score':1., 'pose_world':pose.tolist()}


def run(rows, **overrides):
    return asyncio.run(run_fusion(rows, Variant('guard','test',
        {'enable_singleton_history_guard':True, **overrides}), fps=30))[0]


def test_guard_predicts_without_recounting_old_observations_and_covariance_grows():
    rows = [row(0,'a'),row(0,'b'),row(1,'c',np.pi),row(2,'c',np.pi)]
    results = run(rows)
    assert FusionConfig().enable_singleton_history_guard is False
    assert not results[0]['integrity']['prediction_only']
    for value in results[1:]:
        np.testing.assert_allclose(value['pose_world'],np.eye(4),atol=1e-6)
        assert value['observation_count'] == 0
        assert value['active_sensors'] == []
        assert value['integrity']['evidence'] == 'prediction_only_recent_multiview'
        assert not value['integrity']['covariance_calibrated']
    assert np.trace(results[2]['covariance']) > np.trace(results[1]['covariance'])


def test_prediction_does_not_renew_itself_or_block_multiview_reacquisition():
    rows = [row(0,'a'),row(0,'b'),row(1,'c',np.pi),row(2,'c',np.pi),row(2,'d',np.pi)]
    result = run(rows)
    assert result[-1]['observation_count'] == 2
    assert result[-1]['integrity']['reacquired']
    assert not result[-1]['integrity']['prediction_only']
    expired = run([row(0,'a'),row(0,'b'),row(1,'c',np.pi),row(6,'c',np.pi)])
    assert not expired[-1]['integrity']['prediction_only']
    assert expired[-1]['observation_count'] == 1


def test_smooth_single_view_motion_and_cold_start_are_not_suppressed():
    assert not run([row(0,'c',np.pi)])[0]['integrity']['prediction_only']
    rows = [row(0,'a'),row(0,'b'),row(1,'a',.1,.01),row(1,'b',.1,.01),row(2,'c',.2,.02)]
    result = run(rows)
    assert result[-1]['observation_count'] == 1
    assert not result[-1]['integrity']['prediction_only']


def test_future_and_other_object_evidence_cannot_guard_a_state():
    rows = [row(10,'a'),row(10,'b'),row(1,'c',np.pi,arrival=1.)]
    result = run(rows)
    assert not result[-1]['integrity']['prediction_only']
    other = run([row(0,'a'),row(0,'b'),row(1,'c',np.pi,obj='other')])
    assert not other[-1]['integrity']['prediction_only']


def test_same_group_replacement_and_disabled_temporal_are_explicit():
    rows = [row(0,'a'),row(0,'b'),row(1,'c',np.pi),row(1,'c',0.,arrival=.04)]
    result = run(rows)
    assert result[-1]['observation_count'] == 1
    assert not result[-1]['integrity']['prediction_only']
    disabled = run(rows[:3],enable_temporal_factors=False)
    assert not disabled[-1]['integrity']['prediction_only']


def test_no_inputs_do_not_generate_predictions_or_self_certification():
    assert run([]) == []
    result = run([row(0,'a'),row(1,'b',np.pi)])
    assert not result[-1]['integrity']['prediction_only']


def test_guard_same_state_final_solution_is_order_invariant():
    rows = [row(0,'a'),row(0,'b'),row(1,'a',np.pi),row(1,'b'),row(1,'c')]
    forward = run(rows)
    reverse = asyncio.run(run_fusion(rows,Variant('reverse','test',
        {'enable_singleton_history_guard':True},reverse_order=True),fps=30))[0]
    np.testing.assert_allclose(forward[-1]['pose_world'],reverse[-1]['pose_world'],atol=1e-8)
    np.testing.assert_allclose(forward[-1]['covariance'],reverse[-1]['covariance'],atol=1e-8)


def test_prediction_annotation_survives_disabled_integrity_output():
    result = run([row(0,'a'),row(0,'b'),row(1,'c',np.pi)],enable_integrity_monitoring=False)
    assert result[-1]['integrity'] == {}
    assert result[-1]['prediction_only']
    assert result[-1]['observation_count'] == 0
