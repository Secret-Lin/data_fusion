import numpy as np

from tools.run_real_ots_ablation import equal_camera_baselines, variants


def test_baselines_replace_same_camera_without_changing_raw_observations():
    rows=[]
    for camera,x,time in [('a',100.,0.),('b',2.,.001),('a',0.,.002)]:
        pose=np.eye(4);pose[0,3]=x
        rows.append({'frame_id':0,'camera_id':camera,'arrival_time':time,
                     'pose_world':pose.tolist(),'score':1.,'object_name':'test'})
    means=equal_camera_baselines(rows,['a','b'])
    assert means['mean_equal'][0]['pose_world'][0][3]==1.
    assert len(means['sensor_a'])==1
    assert len(rows)==3


def test_variants_keep_control_and_hold_out_every_camera():
    options=variants(['a','b','c'],True)
    assert options[0].overrides=={}
    assert {v.exclude_camera for v in options if v.exclude_camera}=={'a','b','c'}
    no_history=next(v for v in options if v.name=='guarded_no_history')
    assert no_history.overrides['enable_consensus_history_prediction'] is False


def test_singleton_guard_is_explicit_and_has_current_control():
    assert 'singleton_guard' not in {v.name for v in variants(['a','b'],False)}
    choices = variants(['a','b'],True,True)
    assert choices[0].name == 'original' and choices[0].overrides == {}
    assert next(v for v in choices if v.name=='singleton_guard').overrides == {'enable_singleton_history_guard':True}
    assert {v.exclude_camera for v in choices if v.exclude_camera} == {'a','b'}
