"""Compare cached smoke predictions on common timestamps, without GPU reruns."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from tools.run_real_ots_smoke import reference_at,write_json
from tools.real_ots_reference import reference_valid


def analyze(run_dir,calibration_dir):
    manifest=json.loads((run_dir/'manifest.json').read_text())
    methods=json.loads((run_dir/'predictions.json').read_text())
    session=Path(manifest['session']).name
    references=np.load(calibration_dir/f'{session}_reference_for_native_mesh.npz')
    reference_times = references['candidate_host_time_s']
    def in_reference(frame_id):
        t = manifest['origin_host_s'] + frame_id / 30
        return bool(reference_valid(references, np.array([t]))[0])
    def in_time_range(frame_id):
        t = manifest['origin_host_s'] + frame_id / 30
        return reference_times[0] <= t <= reference_times[-1]
    primary=['fusion','mean_equal']+[k for k,v in methods.items() if k.startswith('sensor_') and v]
    indexed={k:{int(r['frame_id']):r for r in methods[k]} for k in primary}
    common=set.intersection(*(set(rows) for rows in indexed.values()))
    common_before_reference = len(common)
    outside_count = sum(not in_time_range(i) for i in common)
    common = {i for i in common if in_reference(i)}
    result={'status':'candidate_reference_comparison_not_benchmark','common_targets':len(common),
            'target_grouping_hz':30,'primary_methods':primary,
            'missing_sensor_methods':[k for k,v in methods.items() if k.startswith('sensor_') and not v],
            'counts':{k:len(v) for k,v in indexed.items()},'methods':{},
            'common_targets_outside_reference':outside_count,
            'common_targets_invalid_reference':common_before_reference-len(common)-outside_count,
            'reference_scope':manifest.get('reference_scope', 'legacy_candidate_reference')}
    result['reference_exclusion_scope'] = 'outside time range OR invalid OTS interpolation endpoints'
    records=[]
    for method,rows in indexed.items():
        ids=sorted(common);out={}
        if not ids:result['methods'][method]=out;continue
        queries=manifest['origin_host_s']+np.array(ids)/30
        estimated=np.array([rows[i]['pose_world'] for i in ids])
        for key in references.files:
            if not key.startswith('T_world_ydown_mesh_native_'):continue
            ref=reference_at(references['candidate_host_time_s'],references[key],queries)
            te=np.linalg.norm(estimated[:,:3,3]-ref[:,:3,3],axis=1)*1000
            re=np.rad2deg(Rotation.from_matrix(ref[:,:3,:3].transpose(0,2,1)@estimated[:,:3,:3]).magnitude())
            out[key]={'n':len(ids),'translation_p50_p95_mm':np.percentile(te,[50,95]).tolist(),
                      'rotation_p50_p95_deg':np.percentile(re,[50,95]).tolist()}
            records.extend({'method':method,'reference':key,'frame_id':i,'translation_mm':float(t),'rotation_deg':float(r)} for i,t,r in zip(ids,te,re))
        result['methods'][method]=out
    write_json(run_dir/'paired_comparison.json',result)
    write_json(run_dir/'paired_frame_errors.json',records)
    # All-method intersection alone can hide partial-sensor failure intervals.
    # Also compare each baseline to fusion on their own common timestamps.
    pairs = {}
    for name, rows in indexed.items():
        if name == 'fusion':continue
        ids = sorted(i for i in set(rows) & set(indexed['fusion']) if in_reference(i))
        entry = {'n':len(ids), 'variants':{}}
        if ids:
            queries = manifest['origin_host_s'] + np.array(ids)/30
            for key in references.files:
                if not key.startswith('T_world_ydown_mesh_native_'):continue
                ref = reference_at(reference_times, references[key], queries)
                stats, errors = {}, {}
                for method in ('fusion', name):
                    pose = np.array([indexed[method][i]['pose_world'] for i in ids])
                    trans = np.linalg.norm(pose[:,:3,3]-ref[:,:3,3],axis=1)*1000
                    rot = np.rad2deg(Rotation.from_matrix(ref[:,:3,:3].transpose(0,2,1)@pose[:,:3,:3]).magnitude())
                    errors[method] = (trans, rot)
                    stats[method] = {'translation_p50_p95_mm':np.percentile(trans,[50,95]).tolist(),
                                     'rotation_p50_p95_deg':np.percentile(rot,[50,95]).tolist()}
                stats['fusion_translation_win_fraction'] = float(np.mean(errors['fusion'][0] < errors[name][0]))
                stats['fusion_rotation_win_fraction'] = float(np.mean(errors['fusion'][1] < errors[name][1]))
                entry['variants'][key] = stats
        pairs[name] = entry
    write_json(run_dir/'pairwise_comparison.json', pairs)
    timeline_path = run_dir/'input_timeline.json'
    coverage = {'input_frames':manifest['frames_by_sensor'],
                'method_output_groups':{k:len(v) for k,v in methods.items()},
                'note':'Includes initialization; no-output is not automatically an occlusion failure.'}
    if timeline_path.exists():
        timeline = json.loads(timeline_path.read_text())
        groups = {r['frame_id'] for r in timeline}
        coverage['input_time_groups'] = len(groups)
        coverage['methods'] = {}
        for method, rows in methods.items():
            if method.startswith('sensor_'):
                camera = method.removeprefix('sensor_')
                targets = {r['frame_id'] for r in timeline if r['camera_id']==camera}
            else:targets=groups
            available = {r['frame_id'] for r in rows}
            covered = len(targets & available)
            coverage['methods'][method] = {'target_groups':len(targets), 'covered_groups':covered,
                                            'no_output_groups':len(targets-available),
                                            'coverage_fraction':covered/len(targets) if targets else None}
    from collections import Counter
    coverage['fusion_observation_count_distribution'] = dict(Counter(r['observation_count'] for r in methods['fusion']))
    write_json(run_dir/'coverage.json',coverage)
    reference_key='T_world_ydown_mesh_native_tagcube0_Tomato2'
    lines=[f'# Candidate OTS comparison: {session}', '',
           f"Reference mode: {result['reference_scope']}; invalid reference endpoints excluded, not bridged.", '',
           'Fixed display reference: tagcube0 + Tomato2; all four calibration combinations retained in JSON.',
           'Reference calibration, optical validity and motion-based synchronization remain preliminary.',
           'No GT was used for initialization, weighting or fusion. This is not a live-latency benchmark.', '',
           f'All-available-method common targets: {len(common)}; missing sensor methods: {result["missing_sensor_methods"]}.', '',
           '| Method | n | Translation P50 / P95 (mm) | Rotation P50 / P95 (deg) |',
           '|---|---:|---:|---:|']
    for method, values in result['methods'].items():
        v=values.get(reference_key)
        if not v:continue
        t, r=v['translation_p50_p95_mm'],v['rotation_p50_p95_deg']
        lines.append(f'| {method} | {v["n"]} | {t[0]:.3f} / {t[1]:.3f} | {r[0]:.3f} / {r[1]:.3f} |')
    lines.extend(['', '## Pairwise overlap (different pairs may have different target sets)', '',
                  '| Comparator | n | Fusion T P50/P95 mm | Comparator T P50/P95 mm | Fusion R P50/P95 deg | Comparator R P50/P95 deg |',
                  '|---|---:|---:|---:|---:|---:|'])
    for name, pair in pairs.items():
        v=pair['variants'].get(reference_key)
        if not v:continue
        cells=[]
        for metric in ['translation_p50_p95_mm','rotation_p50_p95_deg']:
            for method in ['fusion',name]:
                a,b=v[method][metric]
                cells.append(f'{a:.3f} / {b:.3f}')
        lines.append(f'| {name} | {pair["n"]} | '+ ' | '.join(cells)+' |')
    lines.extend(['', 'See coverage.json for missing outputs and pairwise_comparison.json for comparisons',
                  'that are not restricted to the all-sensor overlap. Do not infer full-sequence performance',
                  'from the all-sensor common subset alone. No-output intervals are not assigned zero error.', '',
                  'Videos show per-camera frontend projections; constant-rate encoding uses each source mean FPS.',
                  'Native capture times are retained in observations.jsonl and input_timeline.json (new runs).'])
    (run_dir/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Report: {(run_dir/"summary.md").resolve()}; common targets={len(common)}')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--calibration-dir',type=Path,default=Path('artifacts/ots_probe_analysis/results_6'))
    args=parser.parse_args();analyze(args.run_dir,args.calibration_dir)
