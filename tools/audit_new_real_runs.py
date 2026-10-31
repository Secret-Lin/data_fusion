"""Read-only cached-result audit. Does not run a frontend or refusion."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.h2o.evaluation import allocate_results
from tools.real_ots_reference import interpolate_reference, reference_valid


KEY = 'T_world_ydown_mesh_native_tagcube0_Tomato2'
SESSIONS = ['stable-0', 'stable-1', 'stable-2', *[f'Test-{i}' for i in range(5)]]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stats(values):
    if not len(values):
        return {'n': 0}
    values = np.asarray(values)
    return {'n': len(values), 't_p50_p95_mm': np.percentile(values[:, 0], [50, 95]).tolist(),
            'r_p50_p95_deg': np.percentile(values[:, 1], [50, 95]).tolist(),
            'translation_over_50mm': int((values[:, 0] > 50).sum()),
            'rotation_over_90deg': int((values[:, 1] > 90).sum())}


def errors(rows, refs, origin, key=KEY):
    indexed = {r['frame_id']: r for r in rows}
    ids = np.array(sorted(indexed), dtype=int)
    queries = origin+ids/30
    keep = reference_valid(refs, queries)
    ids, queries = ids[keep], queries[keep]
    if not len(ids):
        return {}
    truth = interpolate_reference(refs['candidate_host_time_s'], refs[key], queries)
    poses = np.array([indexed[i]['pose_world'] for i in ids])
    t = np.linalg.norm(poses[:, :3, 3]-truth[:, :3, 3], axis=1)*1000
    r = Rotation.from_matrix(truth[:, :3, :3].transpose(0, 2, 1)@poses[:, :3, :3]).magnitude()*180/np.pi
    return {int(i): (float(a), float(b)) for i,a,b in zip(ids,t,r)}


def nearest_distance(times, queries):
    times = np.sort(times)
    if not len(times):
        return np.full(len(queries), np.inf)
    right = np.searchsorted(times, queries).clip(0, len(times)-1)
    left = np.maximum(right-1, 0)
    return np.minimum(np.abs(times[right]-queries), np.abs(times[left]-queries))


def main():
    out = allocate_results(Path('artifacts/new_real_audit'))
    report = {'sessions': {}, 'scope': 'Cached predictions only; fixed tagcube0/Tomato2 display reference; no new fusion or GT-based tuning',
              'limits': 'Dynamic timing is motion-checked, not hardware synchronized; invalid OTS excluded. Missing observations are not independently annotated occlusions. Test-5 not run.'}
    pooled = {k: [] for k in ['original', 'supported_history', 'mean_equal', 'mean_markley_weighted', 'best_score']}
    sensitivity = {}
    for name in SESSIONS:
        source = Path('artifacts/new_real_full')/name/'results_0'
        ablation = Path('artifacts/new_real_ablation')/name/'results_0'
        manifest, am = read(source/'manifest.json'), read(ablation/'manifest.json')
        raw = (source/'observations.jsonl').read_bytes()
        obs = [json.loads(line) for line in raw.splitlines()]
        timeline = read(source/'input_timeline.json')
        predictions = read(ablation/'predictions.json')
        timeline_index = {(r['camera_id'], r['source_frame_index']): r for r in timeline}
        mismatches = sum(timeline_index.get((r['camera_id'],r['source_frame_index']),{}).get('frame_id') != r['frame_id'] for r in obs)
        audit = {'frontend_status': manifest['status'], 'ablation_status': am['status'],
                 'failures': read(ablation/'failures.json'), 'n_input': len(timeline),
                 'n_observations': len(obs), 'sha_matches': hashlib.sha256(raw).hexdigest()==am['observations_sha256'],
                 'timeline_group_mismatches': mismatches,
                 'all_frames_processed': len(timeline)==manifest['input_frame_count']==sum(manifest['frames_by_sensor'].values()),
                 'methods': {}, 'paired_with_original': {}, 'native_sensor_coverage': {}}
        targets = {r['frame_id'] for r in timeline}
        with np.load(Path('artifacts/new_real_reference/results_1')/f'{name}_reference_for_native_mesh.npz') as refs:
            err = {k: errors(v, refs, manifest['origin_host_s']) for k,v in predictions.items()}
            for method, value in err.items():
                audit['methods'][method] = stats(list(value.values()))
                groups = {r['frame_id'] for r in predictions[method]}
                audit['methods'][method]['output_groups'] = len(groups)
                audit['methods'][method]['common_grid_availability'] = len(groups & targets)/len(targets)
                if method in pooled:
                    pooled[method].extend(value.values())
                common = sorted(set(value)&set(err['original']))
                audit['paired_with_original'][method] = {
                    'original': stats([err['original'][i] for i in common]),
                    'comparator': stats([value[i] for i in common])}
            audit['reference_valid_input_groups'] = int(reference_valid(refs, manifest['origin_host_s']+np.array(sorted(targets))/30).sum())
            for camera in sorted({r['camera_id'] for r in timeline}):
                frames = {r['source_frame_index'] for r in timeline if r['camera_id']==camera}
                seen = {r['source_frame_index'] for r in obs if r['camera_id']==camera}
                audit['native_sensor_coverage'][camera] = {'n_input':len(frames), 'n_pose_frames':len(seen), 'fraction':len(seen & frames)/len(frames)}
            original = {r['frame_id']: r for r in predictions['original']}
            audit['observation_count_distribution'] = dict(Counter(r['observation_count'] for r in original.values()))
            audit['bad_rotation_cases'] = [
                {'time_s':i/30, 'error':e, 'active_sensors':original[i]['active_sensors'],
                 'integrity':original[i]['integrity'],
                 'sensor_errors':{k:v[i] for k,v in err.items() if k.startswith('sensor_') and i in v}}
                for i,e in err['original'].items() if e[1]>90]
            camera_times = np.array([r['capture_time'] for r in obs if r['camera_id']=='242322077348'])
            for case in audit['bad_rotation_cases']:
                prior = camera_times[camera_times <= case['time_s']]
                case['preceding_rs2423_ms'] = float((case['time_s']-prior.max())*1000) if len(prior) else None
            if name.startswith('stable-'):
                bias = {}
                for camera in ['134722070227', '242322077348']:
                    xyz = np.array([r['pose_world'] for r in predictions['sensor_'+camera]])[:, :3, 3]
                    center = np.median(xyz, axis=0)
                    bias[camera] = {'median_error_vector_mm': ((center-refs[KEY][0,:3,3])*1000).tolist(),
                                    'spread_p95_about_median_mm': float(np.percentile(np.linalg.norm(xyz-center,axis=1)*1000,95))}
                audit['static_translation_bias'] = bias
            ids = np.array(sorted(err['original']), dtype=int)
            distant = nearest_distance(camera_times, ids/30) > .15
            supplement_ids = ids[distant]
            audit['rs2423_missing_over_150ms'] = stats([err['original'][int(i)] for i in supplement_ids])
            audit['rs2423_missing_over_150ms']['times_s'] = (supplement_ids/30).tolist()
            without = err['guarded_without_quest']
            extra = sorted(set(err['supported_history'])-set(without))
            audit['quest_added_evaluable_groups'] = stats([err['supported_history'][i] for i in extra])
            for method in pooled:
                if set(err[method]) != set(err['original']):
                    raise ValueError(f'Pooled comparison has unmatched targets: {name}/{method}')
            audit['raw_vs_cached_original_max_pose_difference'] = max(
                np.max(np.abs(np.array(r['pose_world'])-np.array(original[r['frame_id']]['pose_world'])))
                for r in read(source/'predictions.json')['fusion'])
            if not name.startswith('stable-'):
                timing = {}
                for shift in [-.1,0.,.1]:
                    shifted = {k:refs[k] for k in refs.files}
                    shifted['candidate_host_time_s'] = refs['candidate_host_time_s']+shift
                    a = errors(predictions['original'],shifted,manifest['origin_host_s'])
                    b = errors(predictions['mean_equal'],shifted,manifest['origin_host_s'])
                    common = set(a)&set(b)
                    timing[str(shift)] = {'original':stats([a[i] for i in common]),'mean_equal':stats([b[i] for i in common])}
                sensitivity[name] = timing
        report['sessions'][name] = audit
        print(name, audit['methods']['original'], flush=True)
    report['pooled_same_output_methods'] = {k:stats(v) for k,v in pooled.items()}
    report['clock_shift_sensitivity_not_optimization'] = sensitivity
    variant_keys = read(Path('artifacts/new_real_ablation/stable-0/results_0/scores.json'))['original']['references']
    report['reference_combination_sensitivity'] = {}
    for key in variant_keys:
        wins = {'translation_p95_sessions':0, 'rotation_p95_sessions':0}
        for name in SESSIONS:
            scores = read(Path('artifacts/new_real_ablation')/name/'results_0/scores.json')
            a,b = scores['original']['references'][key],scores['mean_equal']['references'][key]
            wins['translation_p95_sessions'] += int(a['translation_p50_p95_mm'][1]<b['translation_p50_p95_mm'][1])
            wins['rotation_p95_sessions'] += int(a['rotation_p50_p95_deg'][1]<b['rotation_p50_p95_deg'][1])
        report['reference_combination_sensitivity'][key] = wins
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('OUTPUT:',out.resolve())


if __name__ == '__main__':
    main()
