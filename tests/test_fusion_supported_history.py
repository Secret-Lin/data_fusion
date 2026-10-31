"""Solver regressions, not synthetic dataset experiments."""
import asyncio

import numpy as np

from config import FusionConfig
from experiments.h2o.ablation import Variant, run_fusion


def row(frame, camera, flipped=False):
    pose=np.eye(4)
    if flipped:pose[:3,:3]=np.diag([-1.,-1.,1.])
    return {'frame_id':frame,'camera_id':camera,'source':f'realsense/{camera}/foundationpose/test',
            'object_name':'test','t_host':frame/30,'arrival_time':frame/30,
            'score':1.,'pose_world':pose.tolist()}


def test_supported_history_does_not_follow_singleton_flip():
    rows=[row(0,'a'),row(0,'b'),row(1,'a',True),row(2,'a',True),row(3,'a',True),row(3,'b')]
    old,_=asyncio.run(run_fusion(rows,Variant('old',''),fps=30))
    new,_=asyncio.run(run_fusion(rows,Variant('new','',{'consensus_supported_history_only':True}),fps=30))
    assert old[-1]['active_sensors']==['realsense/a']
    assert new[-1]['active_sensors']==['realsense/b']
    assert new[-1]['integrity']['excluded_history_states']>=1
    assert new[-1]['integrity']['ambiguous']  # Selection is not a GT guarantee.
    assert FusionConfig().consensus_supported_history_only is False


def test_supported_history_respects_existing_gap_and_input_order():
    rows=[row(0,'a'),row(0,'b'),row(1,'a',True),row(60,'a',True),row(60,'b')]
    forward,_=asyncio.run(run_fusion(rows,Variant('forward','',{'consensus_supported_history_only':True}),fps=30))
    reverse,_=asyncio.run(run_fusion(rows,Variant('reverse','',{'consensus_supported_history_only':True},reverse_order=True),fps=30))
    assert not forward[-1]['integrity']['motion_prior_active']
    np.testing.assert_allclose(forward[-1]['pose_world'],reverse[-1]['pose_world'],atol=1e-6)
