"""Read-only numerical audit of saved real-data predictions (no inference/refusion)."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.h2o.ablation import run_baselines
from experiments.h2o.evaluation import allocate_results
from tools.run_real_ots_smoke import reference_at, write_json


def intervals(ids, fps=30):
    runs=[]
    for i in sorted(ids):
        if runs and i == runs[-1][1]+1:
            runs[-1][1]=i
        else:runs.append([i,i])
    return [{'start_s':a/fps,'end_s':b/fps,'n':b-a+1} for a,b in runs]


def audit(run_dir, calibration_dir):
    manifest=json.loads((run_dir/'manifest.json').read_text())
    predictions=json.loads((run_dir/'predictions.json').read_text())
    observations=[json.loads(x) for x in (run_dir/'observations.jsonl').read_text().splitlines()]
    timeline=json.loads((run_dir/'input_timeline.json').read_text())
    sources=sorted({r['camera_id'] for r in timeline})
    session=Path(manifest['session']).name
    ref=np.load(calibration_dir/f'{session}_reference_for_native_mesh.npz')
    key='T_world_ydown_mesh_native_tagcube0_Tomato2'
    origin=manifest['origin_host_s']
    indexed={m:{r['frame_id']:r for r in rows} for m,rows in predictions.items()}
    grouped=defaultdict(list)
    for row in observations:grouped[row['frame_id']].append(row)
    def errors(rows, exact=False):
        query=origin+np.array([r['capture_time'] if exact else r['frame_id']/30 for r in rows])
        truth=reference_at(ref['candidate_host_time_s'],ref[key],query)
        est=np.array([r['pose_world'] for r in rows])
        return (np.linalg.norm(est[:,:3,3]-truth[:,:3,3],axis=1)*1000,
                np.rad2deg(Rotation.from_matrix(truth[:,:3,:3].transpose(0,2,1)@est[:,:3,:3]).magnitude()))
    err={}
    for method,rows in indexed.items():
        if not rows:continue
        ids=sorted(rows)
        t,r=errors([rows[i] for i in ids])
        err[method]={i:(float(a),float(b)) for i,a,b in zip(ids,t,r)}
    summary={'session':session,'manifest_status':manifest['status'],
             'input_frames':len(timeline),'observations':len(observations),'by_sensor':{},
             'fusion_groups':len(indexed['fusion']), 'duplicate_camera_groups':{},
             'phase_note':'Previously estimated motion intervals; diagnostic, not independent action labels.'}
    for camera in sources:
        input_rows=[r for r in timeline if r['camera_id']==camera]
        obs=[r for r in observations if r['camera_id']==camera]
        counts=Counter(r['frame_id'] for r in obs)
        t,r=errors(obs,exact=True) if obs else (np.array([]),np.array([]))
        summary['by_sensor'][camera]={'input_frames':len(input_rows),'pose_frames':len(obs),
            'native_frame_coverage':len(obs)/len(input_rows),
            'exact_time_t_p50_p95_mm':np.percentile(t,[50,95]).tolist() if len(t) else [],
            'exact_time_r_p50_p95_deg':np.percentile(r,[50,95]).tolist() if len(r) else []}
        summary['duplicate_camera_groups'][camera]={'groups':sum(n>1 for n in counts.values()),'excess_observations':sum(max(n-1,0) for n in counts.values())}
    bad=[i for i,e in err['fusion'].items() if e[1]>90]
    summary['rotation_over_90_deg']={m:{'n':sum(e[1]>90 for e in values.values()),'denominator':len(values)} for m,values in err.items()}
    summary['fusion_bad_intervals']=intervals(bad)
    cases=[]
    for i in bad:
        f=indexed['fusion'][i]
        cases.append({'frame_id':i,'t_s':i/30,'fused_error':err['fusion'][i],
            'active_sensors':f['active_sensors'],'integrity':f['integrity'],
            'sensor_errors':{m:values[i] for m,values in err.items() if m.startswith('sensor_') and i in values}})
    best='sensor_242322077348'
    summary['bad_fusion_rs2423_present']=sum(best in c['sensor_errors'] for c in cases)
    summary['bad_fusion_rs2423_under_10deg']=sum(c['sensor_errors'].get(best,(0,999))[1]<10 for c in cases)
    summary['bad_fusion_active_patterns']=dict(Counter('|'.join(c['active_sensors']) for c in cases))
    summary['integrity_evidence']=dict(Counter(r['integrity'].get('evidence') for r in predictions['fusion']))
    summary['bad_fusion_evidence']=dict(Counter(c['integrity'].get('evidence') for c in cases))
    # Fixed motion boundaries from independent earlier synchronization, not this tracker.
    start,end={'20260915_155343':(34.341,65.941),'20260915_155901':(9.208,46.433)}[session]
    summary['phases']={}
    for phase,predicate in [('before_motion',lambda t:t<start),('motion',lambda t:start<=t<=end),('after_motion',lambda t:t>end)]:
        summary['phases'][phase]={}
        for method in ['fusion','mean_equal',best]:
            values=np.array([e for i,e in err[method].items() if predicate(i/30)])
            summary['phases'][phase][method]={'n':len(values),'t_p50_p95_mm':np.percentile(values[:,0],[50,95]).tolist() if len(values) else [],
                'r_p50_p95_deg':np.percentile(values[:,1],[50,95]).tolist() if len(values) else []}
    # Diagnostic equal-camera baseline: last observation for each source/group,
    # matching the production same-camera replacement policy. No fusion rerun.
    dedup={}
    for row in observations:dedup[(row['frame_id'],row['camera_id'])]=row
    balanced=run_baselines(list(dedup.values()),sources)['mean_equal']
    t,r=errors(balanced)
    summary['equal_camera_mean_diagnostic']={'n':len(balanced),'t_p50_p95_mm':np.percentile(t,[50,95]).tolist(),'r_p50_p95_deg':np.percentile(r,[50,95]).tolist(),
        'note':'New diagnostic only; original reported mean_equal unchanged, no GT selection.'}
    return summary,cases


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,action='append',required=True)
    parser.add_argument('--calibration-dir',type=Path,default=Path('artifacts/ots_probe_analysis/results_6'))
    parser.add_argument('--output-parent',type=Path,default=Path('artifacts/real_ots_audit'))
    args=parser.parse_args()
    output=allocate_results(args.output_parent)
    for directory in args.run_dir:
        summary,cases=audit(directory,args.calibration_dir)
        write_json(output/f'{summary["session"]}_audit.json',summary)
        write_json(output/f'{summary["session"]}_rotation_failures.json',cases)
        print(json.dumps(summary,indent=2))
    print(f'OUTPUT: {output.resolve()}')
