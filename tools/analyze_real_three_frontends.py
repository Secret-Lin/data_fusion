"""Read-only scoring/audit of three frozen real-data frontend experiments."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.h2o.evaluation import allocate_results
from tools.audit_new_real_runs import KEY, SESSIONS, errors, read, stats
from tools.real_ots_reference import reference_valid


ROOT = Path(__file__).resolve().parents[1]
PARENTS = {'FoundationPose': 'new_real_singleton_ablation',
           'M3T': 'new_real_m3t_ablation', 'MegaPose': 'new_real_megapose_ablation'}
CANDIDATE = 'singleton_guard'
MAIN = ['mean_equal', 'mean_markley_weighted', 'best_score', 'original', CANDIDATE,
        'supported_history', 'singleton_no_temporal', 'singleton_no_adaptive', 'singleton_no_huber']


def summarize(values, targets=None):
    values = np.asarray(values, dtype=float).reshape(-1, 2)
    result = stats(values)
    if len(values):
        result['mean_translation_mm_rotation_deg'] = values.mean(axis=0).tolist()
        bad = (values[:, 0] > 50) | (values[:, 1] > 10)
        result['joint_bad_observed_count'] = int(bad.sum())
        result['joint_bad_observed_fraction'] = float(bad.mean())
    if targets is not None:
        missing = targets - len(values)
        if missing < 0:
            raise ValueError('More predictions than evaluation targets')
        result.update(targets=targets, missing=missing, output_fraction=len(values) / targets,
                      joint_bad_or_missing_fraction=(missing + result.get('joint_bad_observed_count', 0)) / targets)
    return result


def observation_diagnostics(report):
    """Assess recorded depth-fit evidence, not calibrated probabilities or GT gates."""
    aggregate = defaultdict(Counter)
    initialization = {}
    for session, frontends in report['sessions'].items():
        initialization[session] = {}
        reference = ROOT / 'artifacts/new_real_reference/results_1' / f'{session}_reference_for_native_mesh.npz'
        with np.load(reference) as refs:
            for frontend, entry in frontends.items():
                source = Path(entry['source'])
                origin = read(source / 'manifest.json')['origin_host_s']
                observations = [json.loads(line) for line in (source / 'observations.jsonl').read_text().splitlines()]
                # One last native observation per camera/group, as in the simple baselines.
                latest = {}
                first = {}
                for row in sorted(observations, key=lambda r: (r['arrival_time'], r['camera_id'])):
                    latest[row['camera_id'], row['frame_id']] = row
                    first.setdefault(row['camera_id'], row)
                initial_errors = {}
                for camera in sorted(first):
                    rows = [r for (c, _), r in latest.items() if c == camera]
                    evaluated = errors(rows, refs, origin)
                    counts = aggregate[frontend, camera]
                    for row in rows:
                        if row['frame_id'] not in evaluated:
                            continue
                        t, r = evaluated[row['frame_id']]
                        counts['evaluated_camera_groups'] += 1
                        quality = row.get('quality_metrics') or {}
                        fit = quality.get('depth_inlier_fraction')
                        if fit is not None:
                            counts['depth_fit_available'] += 1
                            if fit >= .9:
                                counts['depth_fit_ge_0_9'] += 1
                                counts['high_depth_fit_rotation_gt_30deg'] += int(r > 30)
                                counts['high_depth_fit_rotation_gt_90deg'] += int(r > 90)
                                counts['high_depth_fit_translation_gt_50mm'] += int(t > 50)
                    e = errors([first[camera]], refs, origin)
                    initial_errors[camera] = {'time_s': first[camera]['capture_time'],
                                              'error_mm_deg_if_valid_reference': e.get(first[camera]['frame_id'])}
                initialization[session][frontend] = initial_errors
    return {'scope': 'Last observation per camera per 30-Hz group; fixed reference; diagnostic only. '
                     'Depth-fit >=0.9 is not a probability or a production threshold.',
            'by_frontend_camera': {frontend: {camera: dict(counts) for (f, camera), counts in aggregate.items() if f == frontend}
                                   for frontend in PARENTS},
            'first_observation_errors': initialization}


def main():
    report = {'reference': KEY, 'scope': 'Frozen predictions; GT only evaluation; no tuning or refusion',
              'sessions': {}, 'pooled': {}, 'cross_frontend_common': {}, 'protocol_groups': {},
              'all_reference_combinations': {}, 'paired_sensors': {}}
    pools, common_pools, group_pools, pair_pools, reference_pools = (defaultdict(list) for _ in range(5))
    target_total = 0
    config_reference = None
    for session in SESSIONS:
        reference_file = ROOT / 'artifacts/new_real_reference/results_1' / f'{session}_reference_for_native_mesh.npz'
        with np.load(reference_file) as refs:
            all_errors, all_predictions, session_report = {}, {}, {}
            timeline_reference = None
            calibration_hash = None
            for frontend, parent in PARENTS.items():
                run = ROOT / 'artifacts' / parent / session / 'results_0'
                am = read(run / 'manifest.json')
                source = Path(am['source_run'])
                sm = read(source / 'manifest.json')
                raw = (source / 'observations.jsonl').read_bytes()
                obs = [json.loads(line) for line in raw.splitlines()]
                predictions = read(run / 'predictions.json')
                timeline = read(source / 'input_timeline.json')
                assert sm['status'] == am['status'] == 'completed'
                assert not read(run / 'failures.json')
                assert hashlib.sha256(raw).hexdigest() == am['observations_sha256']
                assert sm['input_frame_count'] == len(timeline) == sum(sm['frames_by_sensor'].values())
                if timeline_reference is None:
                    timeline_reference, calibration_hash = timeline, sm['calibration_sha256']
                else:
                    assert timeline == timeline_reference, (frontend, session, 'timeline')
                    assert sm['calibration_sha256'] == calibration_hash
                if config_reference is None:
                    config_reference = am['configurations']
                assert am['configurations'] == config_reference, (frontend, session, 'fusion configuration')
                origin = sm['origin_host_s']
                targets = np.array(sorted({r['frame_id'] for r in timeline}), dtype=int)
                valid_targets = set(targets[reference_valid(refs, origin + targets / 30)].tolist())
                if frontend == 'FoundationPose':
                    target_total += len(valid_targets)
                err = {m: errors(rows, refs, origin) for m, rows in predictions.items()}
                all_errors[frontend], all_predictions[frontend] = err, predictions
                candidate = {r['frame_id']: r for r in predictions[CANDIDATE]}
                for method in MAIN:
                    assert set(err[method]) == set(err[CANDIDATE])
                    assert set(err[method]) <= valid_targets
                    pools[frontend, method].extend(err[method].values())
                    group = 'static' if session.startswith('stable') else ('unobstructed_protocol' if session in ['Test-0', 'Test-1'] else 'realsense_occlusion_protocol')
                    group_pools[frontend, group, method].extend(err[method].values())
                pairs = {}
                for method in [m for m in err if m.startswith('sensor_') or m.startswith('singleton_without_')]:
                    shared = sorted(set(err[CANDIDATE]) & set(err[method]))
                    pairs[method] = {}
                    for role, name in [('fusion', CANDIDATE), ('comparator', method)]:
                        vals = [err[name][i] for i in shared]
                        pairs[method][role] = summarize(vals)
                        pair_pools[frontend, method, role].extend(vals)
                support = Counter(r.get('integrity', {}).get('support_count', -1) for r in candidate.values())
                predicted = [r for r in candidate.values() if r.get('prediction_only')]
                opportunities = {'fusion_bad_despite_one_joint_good_same_group': 0, 'no_joint_good_same_group': 0, 'evaluated_groups': len(err[CANDIDATE])}
                for i, value in err[CANDIDATE].items():
                    views = [v[i] for m, v in err.items() if m.startswith('sensor_') and i in v]
                    good = any(t <= 50 and r <= 10 for t, r in views)
                    if not good:
                        opportunities['no_joint_good_same_group'] += 1
                    elif value[0] > 50 or value[1] > 10:
                        opportunities['fusion_bad_despite_one_joint_good_same_group'] += 1
                seed_deltas = []
                if frontend != 'FoundationPose':
                    assert sm['mesh']['export_policy'] == 'native_metre_mesh_windows_safe_material_v2'
                    seeds = read(source / 'initialization.json')
                    seed_manifest = read(Path(sm['seed_run']) / 'manifest.json')
                    assert seed_manifest['calibration_sha256'] == calibration_hash
                    assert hashlib.sha256((Path(sm['seed_run']) / 'observations.jsonl').read_bytes()).hexdigest() == sm['seed_observations_sha256']
                    first = {r['camera_id']: r for r in obs if r['mode'] == 'initialize'}
                    assert len(first) == len(seeds) == 3
                    seed_deltas = [float(np.max(np.abs(np.array(first[c]['pose_world']) - seeds[c]['pose_world']))) for c in seeds]
                    assert max(seed_deltas) < 1e-5
                    assert max(float(np.max(np.abs(np.array(r['T_world_camera']) @ r['pose_cam'] - r['pose_world']))) for r in obs) < 1e-8
                for key in [k for k in refs.files if k.startswith('T_world_ydown_mesh_native_')]:
                    for method in ['mean_equal', CANDIDATE, 'supported_history']:
                        values = errors(predictions[method], refs, origin, key)
                        reference_pools[frontend, key, method].extend(values.values())
                session_report[frontend] = {
                    'source': str(source), 'ablation': str(run), 'audit_passed': True,
                    'prediction_sha256': hashlib.sha256((run / 'predictions.json').read_bytes()).hexdigest(),
                    'reference_sha256': hashlib.sha256(reference_file.read_bytes()).hexdigest(),
                    'input_frames': len(timeline), 'raw_observations': len(obs),
                    'output_groups': len(candidate), 'input_groups': len(targets), 'valid_reference_targets': len(valid_targets),
                    'prediction_only_count': len(predicted), 'support_distribution': dict(support),
                    'seed_max_matrix_deltas': seed_deltas, 'opportunities': opportunities,
                    'methods': {m: summarize(list(e.values()), len(valid_targets)) for m, e in err.items()}, 'paired': pairs}
                print(session, frontend, json.dumps({m: session_report[frontend]['methods'][m] for m in ['mean_equal', CANDIDATE]}), flush=True)
            common = set.intersection(*(set(e[CANDIDATE]) for e in all_errors.values()))
            for frontend, err in all_errors.items():
                for method in MAIN:
                    common_pools[frontend, method].extend(err[method][i] for i in sorted(common))
                session_report[frontend]['cross_frontend_common'] = {m: summarize([err[m][i] for i in sorted(common)]) for m in METHODS_FOR_PRINT}
            report['sessions'][session] = session_report
    for frontend in PARENTS:
        report['pooled'][frontend] = {m: summarize(pools[frontend, m], target_total) for m in MAIN}
        report['cross_frontend_common'][frontend] = {m: summarize(common_pools[frontend, m]) for m in MAIN}
        report['protocol_groups'][frontend] = {g: {m: summarize(group_pools[frontend, g, m]) for m in MAIN} for g in ['static', 'unobstructed_protocol', 'realsense_occlusion_protocol']}
        report['paired_sensors'][frontend] = {m: {role: summarize(vals) for (f, method, role), vals in pair_pools.items() if f == frontend and method == m} for m in sorted({method for f, method, role in pair_pools if f == frontend})}
        report['all_reference_combinations'][frontend] = {key: {m: summarize(vals) for (f, k, m), vals in reference_pools.items() if f == frontend and k == key} for key in sorted({k for f, k, m in reference_pools if f == frontend})}
    report['valid_reference_target_count'] = target_total
    output = allocate_results(ROOT / 'artifacts/real_three_frontend_analysis')
    (output / 'analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    diagnostics = observation_diagnostics(report)
    (output / 'observation_diagnostics.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    print('OUTPUT', output)
    for frontend in PARENTS:
        print('COMMON', frontend, json.dumps({m: report['cross_frontend_common'][frontend][m] for m in METHODS_FOR_PRINT}))


METHODS_FOR_PRINT = ['mean_equal', 'original', CANDIDATE, 'supported_history']

if __name__ == '__main__':
    main()
