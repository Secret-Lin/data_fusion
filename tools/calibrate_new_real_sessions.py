"""Generate new per-session camera extrinsics with tag-only holdout checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from adapters.quest_init import QuestTagInitializer
from experiments.h2o.evaluation import allocate_results
from standalone.cube_model_generator import AprilTagDetector
from standalone.tag_cube_localizer_replay_with_cube_model import CubeModelLocalizer, load_cube_model
from tools.analyze_dual_tag_cubes import _camera_parameters
from tools.run_real_ots_smoke import write_json


SESSIONS=['stable-0','stable-1','stable-2',*[f'Test-{i}' for i in range(6)]]


def project_error(transform, points, pixels, k, dist):
    camera_world=np.linalg.inv(transform)
    xyz=points@camera_world[:3,:3].T+camera_world[:3,3]
    if not np.isfinite(xyz).all() or np.any(xyz[:,2]<=0):return None
    rv,_=cv2.Rodrigues(camera_world[:3,:3])
    projected,_=cv2.projectPoints(points,rv,camera_world[:3,3],k,dist)
    return float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-pixels)**2,axis=1))))


def source_calibration(path, metadata, model_path, full_sequence):
    model=load_cube_model(model_path)
    detector=AprilTagDetector(model.dictionary)
    loc=CubeModelLocalizer(model,5.,12.,2.,min_observations=20)
    quest=QuestTagInitializer(model_path,min_samples=20,strict_geometry=True) if path.stem=='quest' else None
    validation=[];train_frames=0;validation_frames=0;tag_counts={};times_out=[]
    with h5py.File(path,'r') as f:
        times=f['t_host'][:]
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Non-monotonic/invalid input clock')
        indices=np.arange(len(times)) if full_sequence else np.unique(np.linspace(0,len(times)-1,48,dtype=int))
        for ordinal,i in enumerate(indices):
            image=f['color'][i]
            if str(f.attrs.get('color_format','')).lower().startswith('rgb'):
                image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            k,d=_camera_parameters(f,int(i),metadata,path.stem)
            hits=detector.detect(image)
            points,pixels,ids=loc._frame_points(hits)
            for tag_id in ids:tag_counts[str(tag_id)]=tag_counts.get(str(tag_id),0)+1
            tracking=None;meta=None
            if quest:
                meta={'coordinate_system':str(f.attrs.get('coordinate_system',f.attrs.get('coord','unity_cam'))),
                      'intrinsics':{'fx':k[0,0],'fy':k[1,1],'cx':k[0,2],'cy':k[1,2]},
                      'cameraPose':{'position':f['camera_position'][i].tolist(),'rotation':f['camera_rotation_xyzw'][i].tolist()}}
                tracking=quest._tracking_camera(meta)
            if ordinal%4==3:
                validation_frames+=1
                # Use ALL holdout cube detections, not just those whose per-frame
                # PnP looks good. Validation data never participates in fitting.
                if ids:validation.append((int(i),points,pixels,k,d,tracking))
            else:
                train_frames+=1
                if quest:quest.add_frame(image,meta)
                else:
                    loc.add_processed_frame(path.stem)
                    if hits:
                        loc.stats[path.stem].detected_any_tag+=1
                        loc.add_frame(path.stem,int(i),hits,k,d)
                    else:loc.add_no_tag_frame(path.stem)
            times_out.append(float(times[i]-times[0]))
            if full_sequence and ordinal%200==0:print(f'{path.parent.parent.name}/{path.stem}: {ordinal+1}/{len(indices)}',flush=True)
    if quest:
        transform=quest.finish()
        fit_report=quest.fit_report
        accepted=len(quest.samples)
        fit_used=fit_report.get('fit_frames',0)
        transform_key='T_world_tracking'
    else:
        estimate=loc.estimate_all().get(path.stem)
        transform=estimate['T_world_cam'] if estimate else None
        fit_report=estimate['quality'] if estimate else loc.stats[path.stem].asdict()
        accepted=len(loc.observations[path.stem])
        fit_used=estimate['n_observations_used'] if estimate else 0
        transform_key='T_world_cam'
    errors=[];reasons=[]
    for i,pts,pix,k,d,tracking in validation:
        rms=None if transform is None else project_error(transform@tracking if quest else transform,pts,pix,k,d)
        errors.append({'frame_index':i,'rms_px':rms,'positive_depth':rms is not None})
    finite=[e['rms_px'] for e in errors if e['rms_px'] is not None and np.isfinite(e['rms_px'])]
    p95=float(np.percentile(finite,95)) if finite else None
    fraction=sum(e['rms_px'] is not None and e['rms_px']<=5 for e in errors)/len(errors) if errors else 0.
    if transform is None:reasons.append('No converged transform with >=20 retained training frames')
    if len(errors)<10:reasons.append('Fewer than 10 held-out cube-detection frames')
    if len(finite)!=len(errors):reasons.append('Held-out negative-depth/nonfinite projection')
    if p95 is None or p95>5:reasons.append('Held-out RMS reprojection P95 exceeds 5 px or is unavailable')
    if fraction<.8:reasons.append('Fewer than 80% of held-out detections have RMS <=5 px')
    report={'ready':not reasons,'reasons':reasons,'input_frames':len(times),'processed_frames':len(indices),
            'train_frames':train_frames,'accepted_fit_frames':accepted,'retained_fit_frames':fit_used,
            'validation_frames':validation_frames,'validation_tag_frames':len(errors),
            'heldout_rms_p50_p95_px':np.percentile(finite,[50,95]).tolist() if finite else [],
            'heldout_fraction_rms_le5':fraction,'heldout_errors':errors,'fit_report':fit_report,
            'tag_counts':tag_counts,'processed_relative_times_s':times_out,
            transform_key:transform.tolist() if transform is not None else None}
    print(path.parent.parent.name,path.stem,'ready',report['ready'],'heldout P95',p95,flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/laboratory'))
    parser.add_argument('--sessions',nargs='+',default=SESSIONS)
    parser.add_argument('--cube-model',type=Path,default=Path('artifacts/ots_probe_analysis/results_6/cube_model_50_y_down.yaml'))
    parser.add_argument('--output-parent',type=Path,default=Path('artifacts/new_real_calibration'))
    parser.add_argument('--full-sequence',action='store_true',help='Scan every recorded RGB frame; default is 48/source')
    args=parser.parse_args();output=allocate_results(args.output_parent)
    manifest={'full_sequence':args.full_sequence,'status':'running','cube_model':str(args.cube_model.resolve()),
              'cube_sha256':hashlib.sha256(args.cube_model.read_bytes()).hexdigest(),
              'OTS_used_for_camera_calibration':False,'fit_fraction':.75,'validation_fraction':.25,
              'user_confirmed':'Test-5=Quest target mostly invisible; OTS world/cube and Tomato body definition/attachment unchanged',
              'sessions':{}}
    ready={}
    for name in args.sessions:
        root=args.root/name;entry={}
        try:
            md=json.loads((root/'camera_metadata.json').read_text())
            paths=sorted(root.rglob('*.h5'))
            if not paths:raise ValueError('No sensor H5 files')
            for path in paths:
                try:entry[path.stem]=source_calibration(path,md,args.cube_model,args.full_sequence)
                except Exception as exc:entry[path.stem]={'ready':False,'reasons':[repr(exc)]}
        except Exception as exc:entry['session_error']={'ready':False,'reasons':[repr(exc)]}
        manifest['sessions'][name]=entry
        write_json(output/f'{name}_candidates.json',entry)
        if entry and all(v['ready'] for v in entry.values()):ready[name]=entry
        write_json(output/'manifest.json',manifest)
    # Export only whole sessions passing validation. Candidates stay in reports,
    # never silently fill a missing/failed camera from another session or GT.
    write_json(output/'production_checks.json',ready if args.full_sequence else {})
    failed=[name for name in args.sessions if name not in ready]
    manifest.update(status='completed_with_validation_failures' if failed else 'completed',failed_sessions=failed,
                    ready_sessions=list(ready),exported_sessions=list(ready) if args.full_sequence else [])
    write_json(output/'manifest.json',manifest)
    print(f'OUTPUT: {output.resolve()}; validation failed: {failed}',flush=True)
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
