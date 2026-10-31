"""Cached real-data fusion experiments. Never invokes an RGB-D frontend."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.h2o.ablation import VARIANTS, Variant, FusionRunError, build_config, run_baselines, run_fusion
from experiments.h2o.evaluation import allocate_results
from tools.run_real_ots_smoke import reference_at, write_json
from tools.real_ots_reference import reference_valid


def variants(cameras, leave_one_out, singleton_guard=False):
    result=[Variant('original','Original checkpoint fusion'),
            Variant('supported_history','Only multi-view-supported history predicts',
                    {'consensus_supported_history_only':True})]
    if singleton_guard:
        settings = {'enable_singleton_history_guard': True}
        result.append(Variant('singleton_guard', 'Brief explicitly predicted state for conflicting singleton', settings))
        for name in ['no_temporal', 'no_adaptive', 'no_huber']:
            result.append(Variant(f'singleton_{name}', VARIANTS[name].description,
                                  {**settings, **VARIANTS[name].overrides}))
        if leave_one_out:
            result.extend(Variant(f'singleton_without_{c}', 'Leave one physical sensor out',
                                  settings, exclude_camera=c) for c in cameras)
        return result
    for name in ['no_history','no_temporal','no_adaptive','no_huber','no_gate']:
        result.append(Variant(f'guarded_{name}',VARIANTS[name].description,
                              {'consensus_supported_history_only':True,**VARIANTS[name].overrides}))
    if leave_one_out:
        result.extend(Variant(f'guarded_without_{c}','Leave one physical sensor out',
                              {'consensus_supported_history_only':True},exclude_camera=c) for c in cameras)
    return result


def equal_camera_baselines(observations, cameras):
    # The fusion receives every raw observation; only these simple baselines
    # give each camera/state one vote, matching same-camera replacement.
    latest={}
    for row in sorted(observations,key=lambda r:(r['arrival_time'],r['camera_id'])):
        latest[(row['frame_id'],row['camera_id'])]=row
    return run_baselines(list(latest.values()),cameras)


def metrics(rows, refs, origin, ids=None):
    indexed={r['frame_id']:r for r in rows}
    chosen=sorted(set(indexed) if ids is None else set(indexed)&set(ids))
    query_times = origin + np.array(chosen, dtype=float)/30
    keep = reference_valid(refs, query_times)
    valid = [i for i, good in zip(chosen, keep) if good]
    inside = (query_times >= refs['candidate_host_time_s'][0]) & (query_times <= refs['candidate_host_time_s'][-1])
    result={'n_output':len(indexed),'n_evaluated':len(valid),'n_outside_reference':int((~inside).sum()),
            'n_invalid_reference':int((inside & ~keep).sum()),'references':{}}
    result['reference_exclusion_scope'] = 'outside time range OR invalid OTS interpolation endpoints'
    if not valid:return result
    poses=np.array([indexed[i]['pose_world'] for i in valid])
    for key in refs.files:
        if not key.startswith('T_world_ydown_mesh_native_'):continue
        truth=reference_at(refs['candidate_host_time_s'],refs[key],origin+np.array(valid)/30)
        t=np.linalg.norm(poses[:,:3,3]-truth[:,:3,3],axis=1)*1000
        r=np.rad2deg(Rotation.from_matrix(truth[:,:3,:3].transpose(0,2,1)@poses[:,:3,:3]).magnitude())
        result['references'][key]={'translation_p50_p95_mm':np.percentile(t,[50,95]).tolist(),
            'rotation_p50_p95_deg':np.percentile(r,[50,95]).tolist(),
            'rotation_over_90_count':int((r>90).sum()),'translation_over_50mm_count':int((t>50).sum())}
    return result


async def run(args):
    directory=args.run_dir.resolve()
    source=json.loads((directory/'manifest.json').read_text())
    if source['status']!='completed':raise ValueError('Use a completed frontend run')
    raw=(directory/'observations.jsonl').read_bytes()
    observations=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    timeline=json.loads((directory/'input_timeline.json').read_text())
    cameras=sorted({r['camera_id'] for r in timeline})
    # Filter by group once; never split a group between the clip and its suffix.
    if not args.full_sequence:
        keep=lambda r: args.start<=r['frame_id']/30<args.start+args.duration
        observations=[r for r in observations if keep(r)]
        timeline=[r for r in timeline if keep(r)]
    if not observations:raise ValueError('No cached observations in selected interval')
    session=Path(source['session']).name
    output=allocate_results(args.output_parent/session)
    choices=variants(cameras,args.leave_one_out,getattr(args,'singleton_guard',False))
    candidate = 'singleton_guard' if getattr(args,'singleton_guard',False) else 'supported_history'
    manifest={'status':'running','source_run':str(directory),'session':session,
              'observations_sha256':hashlib.sha256(raw).hexdigest(),
              'GT_used_in_fusion':False,'full_sequence':args.full_sequence,
              'start':args.start,'duration':args.duration,'n_raw_observations':len(observations),
              'candidate':candidate,
              'baseline_policy':'last observation per camera/group, equal-camera weights',
              'configurations':{v.name:asdict(build_config(v).fusion) for v in choices},
              'note':'Bounded runs cold-start at clip boundary; not equivalent to full-sequence history.'}
    write_json(output/'manifest.json',manifest)
    predictions=equal_camera_baselines(observations,cameras)
    failures={};stats={}
    try:
        for variant in choices:
            print(f'{session}: {variant.name}',flush=True)
            try:
                rows,stat=await run_fusion(observations,variant,fps=30.)
                predictions[variant.name]=rows;stats[variant.name]=stat
            except FusionRunError as exc:
                predictions[variant.name]=exc.predictions
                failures[variant.name]={'error':str(exc),'failure':exc.failure}
            except Exception as exc:
                predictions[variant.name]=[]
                failures[variant.name]={'error':repr(exc)}
            write_json(output/f'{variant.name}_predictions.json',predictions[variant.name])
        write_json(output/'predictions.json',predictions)
        write_json(output/'fusion_stats.json',stats)
        write_json(output/'failures.json',failures)
        # Open evaluation reference only after all predictions are frozen.
        with np.load(args.calibration_dir/f'{session}_reference_for_native_mesh.npz') as refs:
            scores={k:metrics(v,refs,source['origin_host_s']) for k,v in predictions.items()}
            paired={}
            for name in predictions:
                if name==candidate:continue
                ids={r['frame_id'] for r in predictions[name]}&{r['frame_id'] for r in predictions[candidate]}
                paired[name]={'common_targets':len(ids),'candidate':metrics(predictions[candidate],refs,source['origin_host_s'],ids),
                              'comparator':metrics(predictions[name],refs,source['origin_host_s'],ids)}
        targets={r['frame_id'] for r in timeline}
        coverage={}
        for name,rows in predictions.items():
            method_targets=targets
            if name.startswith('sensor_'):
                method_targets={r['frame_id'] for r in timeline if r['camera_id']==name.removeprefix('sensor_')}
            covered=method_targets & {r['frame_id'] for r in rows}
            coverage[name]={'target_groups':len(method_targets),'output_groups':len(covered),
                            'missing_groups':len(method_targets-covered),
                            'fraction':len(covered)/len(method_targets) if method_targets else None,
                            'prediction_only_groups':sum(bool(r.get('prediction_only',r.get('integrity',{}).get('prediction_only'))) for r in rows)}
        write_json(output/'scores.json',scores);write_json(output/'paired_scores.json',paired)
        write_json(output/'coverage.json',coverage)
        key='T_world_ydown_mesh_native_tagcube0_Tomato2'
        lines=[f'# Cached fusion ablation: {session}', '',
               f'Candidate: {candidate}; prediction-only states are explicitly counted in coverage.json.', '',
               'No frontend rerun. Fixed display reference: tagcube0 + Tomato2; all four references saved.',
               'Rows use own available outputs; use paired_scores.json for matched comparisons.',
               'Missing/failed predictions are counted in coverage, not assigned zero error.', '',
               '| Method | n | T P50/P95 mm | R P50/P95 deg | R >90 count |',
               '|---|---:|---:|---:|---:|']
        for name,value in scores.items():
            v=value['references'].get(key)
            if v:
                t,r=v['translation_p50_p95_mm'],v['rotation_p50_p95_deg']
                lines.append(f'| {name} | {value["n_evaluated"]} | {t[0]:.2f}/{t[1]:.2f} | {r[0]:.2f}/{r[1]:.2f} | {v["rotation_over_90_count"]} |')
            else:lines.append(f'| {name} | 0 | unavailable | unavailable | unavailable |')
        lines.extend(['',f'Failures: {failures}', '',
                      'Candidate is opt-in; singleton output is not ground-truth verified. No live performance claim.',
                      'No MP4 regenerated: existing videos contain frontend boxes, not new fusion projections.'])
        (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        manifest['status']='completed_with_failures' if failures else 'completed'
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:
        write_json(output/'manifest.json',manifest)
        print(f'OUTPUT: {output.resolve()}',flush=True)
    return bool(failures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--full-sequence',action='store_true')
    parser.add_argument('--start',type=float)
    parser.add_argument('--duration',type=float)
    parser.add_argument('--leave-one-out',action='store_true')
    parser.add_argument('--singleton-guard',action='store_true',help='Opt-in recent-multiview singleton protection experiment')
    parser.add_argument('--output-parent',type=Path,default=Path('artifacts/real_ots_ablation'))
    parser.add_argument('--calibration-dir',type=Path,default=Path('artifacts/ots_probe_analysis/results_6'))
    args=parser.parse_args()
    if args.full_sequence:
        if args.start is not None or args.duration is not None:parser.error('Full sequence cannot specify a clip')
    elif args.start is None or args.duration is None or not 0<args.duration<=10:
        parser.error('Specify a <=10 second clip, or explicitly --full-sequence')
    if asyncio.run(run(args)):raise SystemExit(1)


if __name__=='__main__':main()
