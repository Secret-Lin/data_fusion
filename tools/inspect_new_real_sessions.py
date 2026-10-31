"""Bounded tag-only feasibility check; never trusts saved camera extrinsics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from adapters.quest_init import QuestTagInitializer
from experiments.h2o.evaluation import allocate_results
from standalone.cube_model_generator import AprilTagDetector
from standalone.tag_cube_localizer_replay_with_cube_model import CubeModelLocalizer, load_cube_model
from tools.analyze_dual_tag_cubes import _camera_parameters
from tools.run_real_ots_smoke import write_json


def heldout_metrics(estimate, samples):
    if estimate is None or not samples:return {'n':len(samples)}
    poses=np.array(samples)
    t=np.linalg.norm(poses[:,:3,3]-estimate[:3,3],axis=1)*1000
    r=np.rad2deg(Rotation.from_matrix(estimate[:3,:3].T@poses[:,:3,:3]).magnitude())
    return {'n':len(samples),'translation_p50_p95_mm':np.percentile(t,[50,95]).tolist(),
            'rotation_p50_p95_deg':np.percentile(r,[50,95]).tolist(),
            'meaning':'held-out per-frame PnP consistency, not absolute GT accuracy'}


def inspect(session, model_path):
    metadata=json.loads((session/'camera_metadata.json').read_text())
    model=load_cube_model(model_path)
    detector=AprilTagDetector(model.dictionary)
    entry={'session':str(session),'dictionary':model.dictionary,'sources':{},
           'existing_extrinsic_files_ignored':[str(p) for p in session.glob('*extrinsic*.json')],
           'scope':'bounded 48 evenly spaced frames/source; interleaved 2/3 fit, 1/3 validation; not full calibration'}
    for path in sorted(session.rglob('*.h5')):
        serial=path.stem
        localizer=CubeModelLocalizer(model,5.,12.,2.,min_observations=20)
        fit=QuestTagInitializer(model_path,min_samples=20) if serial=='quest' else None
        validation=QuestTagInitializer(model_path,min_samples=20) if fit else None
        heldout=[];sample_frames=[]
        with h5py.File(path,'r') as f:
            times=f['t_host'][:];n=len(times)
            record={'h5':str(path),'frames':n,'duration_s':float(times[-1]-times[0]),
                    'fps_mean':float((n-1)/(times[-1]-times[0])),
                    'timestamps_strict':bool(np.isfinite(times).all() and np.all(np.diff(times)>0)),
                    'color_shape':list(f['color'].shape),'depth_shape':list(f['depth'].shape),
                    'coordinate_system':str(f.attrs.get('coordinate_system',f.attrs.get('coord','y_down')))}
            ids=np.unique(np.linspace(0,n-1,min(48,n),dtype=int))
            for ordinal,i in enumerate(ids):
                is_validation=ordinal%3==2
                im=f['color'][i]
                if str(f.attrs.get('color_format','')).lower().startswith('rgb'):
                    im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
                k,d=_camera_parameters(f,int(i),metadata,serial)
                if fit:
                    target=validation if is_validation else fit
                    before=len(target.samples)
                    target.add_frame(im,{'coordinate_system':record['coordinate_system'],
                        'intrinsics':{'fx':k[0,0],'fy':k[1,1],'cx':k[0,2],'cy':k[1,2]},
                        'cameraPose':{'position':f['camera_position'][i].tolist(),
                                      'rotation':f['camera_rotation_xyzw'][i].tolist()}})
                    valid=len(target.samples)>before
                else:
                    target=CubeModelLocalizer(model,5.,12.,2.,min_observations=3) if is_validation else localizer
                    target.add_processed_frame(serial)
                    hits=detector.detect(im)
                    valid=False
                    if hits:
                        target.stats[serial].detected_any_tag+=1
                        valid=target.add_frame(serial,int(i),hits,k,d)
                    else:target.add_no_tag_frame(serial)
                    if valid and is_validation:heldout.append(np.linalg.inv(target.observations[serial][-1].T_cam_world))
                sample_frames.append({'frame_index':int(i),'relative_time_s':float(times[i]-times[0]),
                                      'validation':is_validation,'accepted':bool(valid)})
        if fit:
            estimate=fit.finish();nfit=len(fit.samples);heldout=validation.samples
            record['T_world_tracking']=estimate.tolist() if estimate is not None else None
        else:
            result=localizer.estimate_all().get(serial)
            estimate=result['T_world_cam'] if result is not None else None
            nfit=len(localizer.observations[serial])
            record['T_world_cam']=estimate.tolist() if estimate is not None else None
            record['fit_quality']=result['quality'] if result is not None else None
        record.update(tested_frames=len(ids),accepted_fit_frames=nfit,
                      provisional_estimate_available=estimate is not None,
                      heldout=heldout_metrics(estimate,heldout),sample_frames=sample_frames)
        if estimate is None:record['reason']='Insufficient valid sampled tag/PnP observations; full scan may still recover calibration.'
        entry['sources'][serial]=record
        print(session.name,serial,'fit',nfit,'validation',len(heldout),'estimate',estimate is not None,flush=True)
    entry['ots_files']=[]
    for path in session.glob('*.csv'):
        with path.open(encoding='utf-8-sig',errors='replace',newline='') as stream:
            rows=[next(csv.reader(stream)) for _ in range(7)]
        entry['ots_files'].append({'path':str(path),'metadata_header':rows[0],
            'has_physical_marker_columns':'Marker' in rows[2],
            'rigid_bodies':sorted(set(name for typ,name in zip(rows[2],rows[3]) if typ=='Rigid Body'))})
    return entry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/laboratory'))
    parser.add_argument('--cube-model',type=Path,default=Path('artifacts/ots_probe_analysis/results_6/cube_model_50_y_down.yaml'))
    parser.add_argument('--output-parent',type=Path,default=Path('artifacts/new_real_sessions'))
    args=parser.parse_args()
    output=allocate_results(args.output_parent)
    report={'cube_model':str(args.cube_model.resolve()),'cube_sha256':hashlib.sha256(args.cube_model.read_bytes()).hexdigest(),
            'OTS_used_for_camera_calibration':False,'sessions':{}}
    for name in ['stable-0','stable-1','stable-2',*[f'Test-{i}' for i in range(6)]]:
        report['sessions'][name]=inspect(args.root/name,args.cube_model)
        write_json(output/'inspection.json',report)
    print(f'OUTPUT: {output.resolve()}',flush=True)


if __name__=='__main__':main()
