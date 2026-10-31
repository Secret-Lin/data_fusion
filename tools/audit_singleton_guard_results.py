"""Audit saved singleton-guard experiments without frontend or fusion execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.h2o.evaluation import allocate_results
from tools.audit_new_real_runs import KEY, SESSIONS, errors, read, stats


def main() -> None:
    output = allocate_results(Path('artifacts/singleton_guard_audit'))
    report = {'scope': 'Cached predictions only; no parameter tuning or refusion',
              'reference': KEY, 'sessions': {}, 'pooled': {}, 'all_references': {}}
    pooled = {}
    reference_pool = {}
    for session in SESSIONS:
        run = Path('artifacts/new_real_singleton_ablation') / session / 'results_0'
        manifest = read(run / 'manifest.json')
        source = Path(manifest['source_run'])
        source_manifest = read(source / 'manifest.json')
        predictions = read(run / 'predictions.json')
        previous = read(Path('artifacts/new_real_ablation') / session / 'results_0' / 'predictions.json')
        indexed = {k: {r['frame_id']: r for r in rows} for k, rows in predictions.items()}
        original, candidate = indexed['original'], indexed['singleton_guard']
        integrity = {
            'status': manifest['status'], 'failures': read(run / 'failures.json'),
            'configurations': len(manifest['configurations']),
            'sha_matches': hashlib.sha256((source / 'observations.jsonl').read_bytes()).hexdigest() == manifest['observations_sha256'],
            'same_output_ids': set(original) == set(candidate),
            'previous_original_ids_match': set(original) == {r['frame_id'] for r in previous['original']},
            'previous_original_max_matrix_difference': max(float(np.max(np.abs(
                np.array(r['pose_world']) - original[r['frame_id']]['pose_world']))) for r in previous['original']),
        }
        if (integrity['status'] != 'completed' or integrity['failures']
                or not integrity['sha_matches'] or not integrity['same_output_ids']
                or not integrity['previous_original_ids_match']
                or integrity['previous_original_max_matrix_difference'] > 1e-10):
            raise ValueError(f'Integrity failure: {session}: {integrity}')
        prediction_ids = sorted(i for i, row in candidate.items() if row.get('prediction_only', False))
        changed = sorted(i for i in candidate if not np.allclose(candidate[i]['pose_world'], original[i]['pose_world'], atol=1e-10, rtol=0))
        with np.load(Path('artifacts/new_real_reference/results_1') / f'{session}_reference_for_native_mesh.npz') as refs:
            err = {k: errors(rows, refs, source_manifest['origin_host_s']) for k, rows in predictions.items()}
            methods = {k: stats(list(e.values())) for k, e in err.items()}
            for method in ['original', 'singleton_guard', 'supported_history', 'mean_equal',
                           'mean_markley_weighted', 'best_score', 'singleton_no_temporal',
                           'singleton_no_adaptive', 'singleton_no_huber']:
                if set(err[method]) != set(err['original']):
                    raise ValueError(f'Unmatched pooled targets: {session}/{method}')
                pooled.setdefault(method, []).extend(err[method].values())
            paired = {}
            for method, value in err.items():
                common = sorted(set(value) & set(err['singleton_guard']))
                paired[method] = {'candidate': stats([err['singleton_guard'][i] for i in common]),
                                  'comparator': stats([value[i] for i in common])}
            for key in sorted(k for k in refs.files if k.startswith('T_world_ydown_mesh_native_')):
                for method in ['original', 'singleton_guard', 'mean_equal', 'mean_markley_weighted']:
                    values = errors(predictions[method], refs, source_manifest['origin_host_s'], key)
                    reference_pool.setdefault(key, {}).setdefault(method, []).extend(values.values())
            cases = []
            case_ids = set(prediction_ids) | {i for i, e in err['original'].items() if e[1] > 90} | {i for i, e in err['singleton_guard'].items() if e[1] > 90}
            for i in sorted(case_ids):
                cases.append({'frame_id': i, 'time_s': i / 30,
                              'original_error_mm_deg': err['original'].get(i),
                              'candidate_error_mm_deg': err['singleton_guard'].get(i),
                              'candidate': candidate[i], 'original_integrity': original[i].get('integrity')})
            predicted_valid = [i for i in prediction_ids if i in err['singleton_guard']]
            changed_valid = [i for i in changed if i in err['singleton_guard']]
            delta = np.array([np.array(err['singleton_guard'][i]) - err['original'][i] for i in changed_valid])
            audit = {'integrity': integrity, 'coverage': read(run / 'coverage.json'), 'methods': methods,
                     'paired': paired, 'prediction_only_count': len(prediction_ids),
                     'prediction_only_evaluable': len(predicted_valid), 'changed_output_count': len(changed),
                     'changed_evaluable_count': len(changed_valid),
                     'changed_error_delta_min_max_mm_deg': [delta.min(axis=0).tolist(), delta.max(axis=0).tolist()] if len(delta) else None,
                     'prediction_only_original': stats([err['original'][i] for i in predicted_valid]),
                     'prediction_only_candidate': stats([err['singleton_guard'][i] for i in predicted_valid]),
                     'prediction_annotation_valid': all(candidate[i]['observation_count'] == 0 and not candidate[i]['active_sensors'] for i in prediction_ids),
                     'cases': cases}
            report['sessions'][session] = audit
            print(session, json.dumps({'original': methods['original'], 'candidate': methods['singleton_guard'],
                                      'predicted': len(prediction_ids), 'changed': len(changed)}), flush=True)
    report['pooled'] = {k: stats(v) for k, v in pooled.items()}
    report['all_references'] = {key: {k: stats(v) for k, v in values.items()} for key, values in reference_pool.items()}
    (output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('POOLED', json.dumps(report['pooled']))
    print('OUTPUT', output.resolve())


if __name__ == '__main__':
    main()
