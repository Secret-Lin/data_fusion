import csv

import numpy as np
import pytest

from tools.prepare_new_real_references import fit_clock, static_reference, motive_epoch
from tools.real_ots_reference import interpolate_reference, read_motive, reference_valid


def test_validity_does_not_bridge_missing_samples():
    refs = {'candidate_host_time_s': np.array([0., .01, .02, .03]),
            'reference_valid': np.array([True, False, True, True])}
    queries = np.array([-.01, 0., .005, .01, .015, .02, .025, .04])
    assert reference_valid(refs, queries).tolist() == [False, True, False, False, False, True, True, False]
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[1] = np.nan
    value = interpolate_reference(refs['candidate_host_time_s'], poses, queries[[1, 5, 6]])
    assert np.isfinite(value).all()
    with pytest.raises(ValueError, match='invalid OTS'):
        interpolate_reference(refs['candidate_host_time_s'], poses, np.array([.005]))


def test_static_reference_is_not_a_clock_estimate():
    refs = {'candidate_host_time_s': np.array([100., 200.]),
            'reference_valid': np.array([True, True]), 'static_reference': np.array(True)}
    assert reference_valid(refs, np.array([150.]))[0]
    poses = np.repeat(np.eye(4)[None], 20, axis=0)
    center, spread = static_reference(poses)
    assert np.allclose(center, np.eye(4))
    assert spread['translation_p95_max_mm'][1] == 0


def test_clock_recovers_independent_shift_and_rejects_constant_signal():
    t = np.arange(0., 30., .1)
    ot = np.arange(0., 40., .05)
    def signal(x):
        return np.exp(-((x-7.)/.8)**2) + .7*np.exp(-((x-15.)/1.1)**2) + 1.3*np.exp(-((x-25.)/.6)**2)
    fit = fit_clock(t, signal(t), ot, signal(ot-4.), 4.)
    assert abs(fit['offset_s']-4.) < .03
    assert fit['held_correlation'] > .99
    constant = fit_clock(t, np.ones(len(t)), ot, np.ones(len(ot)), 4.)
    assert constant['train_correlation'] == -1


def test_missing_physical_marker_is_not_missing_body(tmp_path):
    columns = [('Rigid Body', 'TomatoSouce', 'Rotation', a) for a in 'XYZW']
    columns += [('Rigid Body', 'TomatoSouce', 'Position', a) for a in 'XYZ']
    for kind in ['Rigid Body Marker', 'Marker']:
        columns += [(kind, f'TomatoSouce:Marker{n}', 'Position', a) for n in range(4) for a in 'XYZ']
    h = [['Rotation Type', 'Quaternion', 'Length Units', 'Millimeters', 'Coordinate Space', 'Global'], [],
         ['', 'Type']+[c[0] for c in columns], ['', 'Name']+[c[1] for c in columns],
         ['', 'ID']+['id']*len(columns), ['', '']+[c[2] for c in columns],
         ['Frame', 'Time (Seconds)']+[c[3] for c in columns]]
    markers = [0, 0, 0, 20, 0, 0, 0, 20, 0, 0, 0, 20]
    row = [0, 0., 0, 0, 0, 1, 0, 0, 0]+markers+markers
    missing_one = row.copy()
    missing_one[0:2] = [1, .01]
    missing_one[-3:] = ['', '', '']
    missing_two = missing_one.copy()
    missing_two[0:2] = [2, .02]
    missing_two[-6:-3] = ['', '', '']
    path = tmp_path/'motive.csv'
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows(h+[row, missing_one, missing_two])
    value = read_motive(path)
    assert value['finite'].tolist() == [True, True, True]
    assert value['marker_count'].tolist() == [4, 3, 2]
    assert value['valid'].tolist() == [True, True, False]


def test_localized_motive_am_pm():
    assert motive_epoch('2026-09-16 06.01.42.554 下午') - motive_epoch('2026-09-16 06.01.42.554 上午') == 43200


def test_cached_ablation_excludes_bad_reference_without_losing_output_count():
    from tools.run_real_ots_ablation import metrics
    class Reference(dict):
        @property
        def files(self):
            return list(self)
    refs = Reference(candidate_host_time_s=np.arange(4)/30,
                     reference_valid=np.array([True, False, True, True]),
                     T_world_ydown_mesh_native_tagcube0_Tomato2=np.repeat(np.eye(4)[None], 4, axis=0))
    refs['T_world_ydown_mesh_native_tagcube0_Tomato2'][1] = np.nan
    rows = [{'frame_id': i, 'pose_world': np.eye(4).tolist()} for i in range(4)]
    result = metrics(rows, refs, 0.)
    assert result['n_output'] == 4
    assert result['n_evaluated'] == 3
    assert result['n_invalid_reference'] == 1
    assert result['n_outside_reference'] == 0
