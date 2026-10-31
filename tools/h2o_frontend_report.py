"""Read-only, audited three-frontend H2O report and publication figures."""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



ROOT = Path(__file__).resolve().parents[1]
AUDITS = {
    'FoundationPose': ROOT / 'artifacts/h2o_checkpoint_audit/foundationpose/results_0/summary.json',
    'M3T': ROOT / 'artifacts/h2o_checkpoint_audit/m3t/results_0/summary.json',
    'MegaPose': ROOT / 'artifacts/h2o_checkpoint_audit/megapose/results_0/summary.json',
}
BASELINES = [f'sensor_cam{i}' for i in range(5)] + ['mean_equal', 'full']
CURVES = ['sensor_cam1', 'mean_equal', 'stored_fusion', 'full']
SELECTED = BASELINES + ['stored_fusion']
ABLATIONS = ['full', 'no_adaptive', 'no_huber', 'no_gate', 'no_robust', 'no_grouping',
             'no_temporal', 'no_history', 'no_integrity']
CONTROLS = ['arrival_grouping', 'reverse_order', 'weak_initial_prior', 'quality_consensus',
            'revision_checkpoint', 'revision_checkpoint_reverse', 'revision_checkpoint_quality']
METRICS = ['translation_mean_mm', 'translation_p95_mm', 'rotation_mean_deg',
           'rotation_p95_deg', 'failure_rate_including_missing', 'coverage']
HEADINGS = ['T mean\n(mm)', 'T P95\n(mm)', 'R mean\n(deg)', 'R P95\n(deg)', 'Failure\n(%)', 'Coverage\n(%)']
LABELS = {'full': 'Ours (checkpoint)', 'mean_equal': 'Equal fusion',
          'revision_checkpoint': 'Checkpoint (= current default)',
          'revision_checkpoint_quality': 'Checkpoint + quality (= quality override)',
          'quality_consensus': 'Quality override (candidate)',
          'best_score': 'Best score (= cam0 for constant scores)',
          'mean_markley_weighted': 'Score-weighted (= equal for constant scores)',
          'stored_fusion': 'Previous fusion',
          'no_adaptive': 'No adaptive covariance', 'no_huber': 'No Huber',
          'no_gate': 'No innovation gates', 'no_robust': 'No Huber or gates',
          'no_grouping': 'No capture grouping', 'no_temporal': 'No temporal factors',
          'no_history': 'No temporal factors/history', 'no_integrity': 'No integrity monitoring',
          'arrival_grouping': 'Arrival grouping', 'reverse_order': 'Reverse camera order',
          'weak_initial_prior': 'Weak initial prior',
          'revision_checkpoint_reverse': 'Checkpoint, reversed order',
          'no_online_calibration': 'No online calibration (not applicable)'}
LABELS.update({f'sensor_cam{i}': f'Camera {i}' for i in range(5)})
LABELS.update({f'without_cam{i}': f'Without camera {i}' for i in range(5)})
COLORS = {'sensor_cam0': '#9467bd', 'sensor_cam1': '#1f77b4', 'sensor_cam2': '#8c564b',
          'sensor_cam3': '#7f7f7f', 'sensor_cam4': '#17becf', 'mean_equal': '#ff7f0e',
          'full': '#d62728', 'revision_checkpoint': '#2ca02c', 'stored_fusion': '#2ca02c'}
FRONTEND_COLORS = {'FoundationPose': '#1f77b4', 'M3T': '#d62728', 'MegaPose': '#ff7f0e'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def scaled(metric, value):
    return value * 100 if metric in ('failure_rate_including_missing', 'coverage') else value


def prepare(output_parent=ROOT / 'artifacts/h2o_frontend_comparison', audits=None):
    """Pool saved errors exactly; do not rerun tracking/fusion or select good takes."""
    from experiments.h2o.evaluation import MetricAccumulator, allocate_results, read_jsonl, write_json
    from tools.audit_h2o_fusion import validate_targets, verify_metrics

    output = allocate_results(Path(output_parent).resolve())
    audits = audits or AUDITS
    records, arrays, summary, fingerprints, target_index = [], {}, {}, {}, {}
    stability = []
    object_records = []
    for frontend, audit_path in audits.items():
        audit = read_json(audit_path)
        summary[frontend] = audit['groups']
        fingerprints[frontend] = {'audit': str(audit_path),
                                  'sha256': hashlib.sha256(Path(audit_path).read_bytes()).hexdigest()}
        keys = set()
        pools = defaultdict(MetricAccumulator)
        objects = defaultdict(lambda: defaultdict(MetricAccumulator))
        for subject, checked in audit['audits'].items():
            order = checked['output_diagnostics']['full_vs_reverse_order']
            stability.append(dict(frontend=frontend, subject=subject, takes=checked['n_takes'],
                                  frames=checked['n_frames'], identical_order_takes=order['identical_pose_and_covariance_takes'],
                                  max_order_pose_delta=order['max_pose_matrix_delta']))
            root = Path(checked['input'])
            if hashlib.sha256((root / 'summary.json').read_bytes()).hexdigest() != checked['source_summary_sha256']:
                raise ValueError(f'Source changed since audit: {root}; audit again')
            source = read_json(root / 'summary.json')
            if source.get('failed_takes') or source.get('failed_methods') or source.get('code_changed_during_run'):
                raise ValueError(f'Incomplete or mixed-code source: {root}')
            for entry in source['per_take']:
                take = entry['take']
                directory = root / 'takes' / take
                report = read_json(directory / 'summary.json')
                truth = read_jsonl(Path(report['input']) / 'ground_truth.jsonl')
                identity = (subject, take)
                keys.add(identity)
                target = [(r['frame_id'], r['object_name'], r['pose_world']) for r in truth]
                if identity in target_index:
                    reference = target_index[identity]
                    if ([(r[0], r[1]) for r in target] != [(r[0], r[1]) for r in reference]
                            or not np.allclose([r[2] for r in target], [r[2] for r in reference], atol=1e-10, rtol=0)):
                        raise ValueError(f'Cross-frontend GT mismatch: {identity}')
                else:
                    target_index[identity] = target
                for method, item in report['methods'].items():
                    if item['status'] == 'completed':
                        records.append(dict(frontend=frontend, subject=subject, take=take,
                                            object=report['object_name'], method=method, **item['metrics']))
                for method in SELECTED:
                    with gzip.open(directory / method / 'scores.jsonl.gz', 'rt', encoding='utf-8') as stream:
                        rows = [json.loads(line) for line in stream if line.strip()]
                    validate_targets(rows, truth, f'{frontend}/{subject}/{take}/{method}')
                    one = MetricAccumulator()
                    one.add(rows)
                    verify_metrics(one.result(), report['methods'][method]['metrics'], method)
                    pools[method].add(rows)
                    objects[report['object_name']][method].add(rows)
            print(f'{frontend} {subject}: {len(source["per_take"])} takes read', flush=True)
        if len(keys) != audit['groups']['all_subjects']['n_takes']:
            raise ValueError('Duplicated/missing take identities')
        if frontend != next(iter(audits)) and keys != first_keys:
            raise ValueError('Frontends do not cover the same take set')
        first_keys = keys
        for method, pool in pools.items():
            verify_metrics(pool.result(), audit['groups']['all_subjects']['methods'][method], f'{frontend}/{method}')
            for metric in ('translation_error_m', 'rotation_error_deg'):
                arrays[f'{frontend}|{method}|{metric}'] = np.concatenate(pool.chunks.get(metric, [np.array([])]))
        for obj, methods in objects.items():
            for method, pool in methods.items():
                object_records.append(dict(frontend=frontend, object=obj, method=method, **pool.result()))
    df, obj = pd.DataFrame(records), pd.DataFrame(object_records)
    df.to_csv(output / 'per_take_metrics.csv', index=False)
    obj.to_csv(output / 'per_object_metrics.csv', index=False)
    np.savez_compressed(output / 'frame_errors.npz', **arrays)
    data = {'groups': summary, 'sources': fingerprints, 'stability': stability,
            'matched_takes': len(first_keys), 'matched_frames': sum(len(v) for v in target_index.values()),
            'note': 'Checkpoint v2; exact frame pooling; every take retained. GT-assisted seeds; M3T/MegaPose inherit FP first poses. '
                    'Failure=T>50mm OR R>10deg OR missing. Rotation/ADD not symmetry-adjusted. '
                    'Checkpoint is the offline default, not a separate candidate. No live-latency claim.'}
    write_json(output / 'report.json', data)
    return load(output)


def load(output):
    output = Path(output)
    return {'output': output, **read_json(output / 'report.json'),
            'takes': pd.read_csv(output / 'per_take_metrics.csv'),
            'objects': pd.read_csv(output / 'per_object_metrics.csv'),
            'arrays': np.load(output / 'frame_errors.npz')}


def save(fig, data, name):
    fig.tight_layout()
    folder = data['output'] / 'figures'
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f'{name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(folder / f'{name}.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)


def metrics_table(data, methods, stem):
    """One figure per frontend, one table category per notebook cell."""
    for frontend, groups in data['groups'].items():
        metrics = groups['all_subjects']['methods']
        rows = []
        for method in methods:
            if method in metrics:
                rows.append([LABELS.get(method, method)] + [scaled(k, metrics[method][k]) for k in METRICS])
            elif method == 'no_online_calibration':
                rows.append([LABELS[method]] + [None] * len(METRICS))
        table = pd.DataFrame(rows, columns=['Method'] + [s.replace('\n', ' ') for s in HEADINGS])
        table.to_csv(data['output'] / f'{stem}_{frontend}.csv', index=False)
        table.to_latex(data['output'] / f'{stem}_{frontend}.tex', index=False, float_format='%.3f', na_rep='N/A')
        fig, ax = plt.subplots(figsize=(14, 1.8 + 0.35 * len(rows)))
        ax.axis('off')
        text = [[row[0]] + [('N/A' if v is None else f'{v:.3f}') for v in row[1:]] for row in rows]
        artist = ax.table(cellText=text, colLabels=['Method'] + HEADINGS, loc='center',
                          cellLoc='right', colWidths=[0.38] + [0.103] * 6)
        artist.auto_set_font_size(False)
        artist.set_fontsize(9)
        artist.scale(1, 1.75)
        for (r, c), cell in artist.get_celld().items():
            cell.set_edgecolor('#dddddd')
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_facecolor('#eeeeee')
                cell.set_text_props(weight='bold')
            elif rows[r - 1][0] == LABELS['full']:
                cell.set_facecolor('#fce8e6')
            if c == 0:
                cell.set_text_props(ha='left')
        ax.set_title(f'{frontend} | {stem.replace("_", " ")} | {data["matched_takes"]} takes / {data["matched_frames"]:,} frames', pad=18)
        fig.text(0.02, 0.01, 'Frame-pooled errors; lower is better except coverage. Failure: >50 mm or >10 deg or missing. N/A is not zero.', fontsize=8)
        save(fig, data, f'{stem}_{frontend}')


def overview(data):
    selected = ['translation_mean_mm', 'translation_p95_mm', 'rotation_p95_deg', 'failure_rate_including_missing']
    labels = ['Translation mean (mm)', 'Translation P95 (mm)', 'Rotation P95 (deg)', 'Failure (%)']
    fig, axes = plt.subplots(len(data['groups']), 4, figsize=(16, 4 * len(data['groups'])), sharex='col', squeeze=False)
    for i, (frontend, groups) in enumerate(data['groups'].items()):
        for j, (metric, label) in enumerate(zip(selected, labels)):
            ax = axes[i, j]
            values = [scaled(metric, groups['all_subjects']['methods'][m][metric]) for m in BASELINES]
            ax.barh(np.arange(len(BASELINES)), values, color=[COLORS[m] for m in BASELINES], alpha=0.75)
            ax.set_yticks(np.arange(len(BASELINES)), [LABELS[m] for m in BASELINES], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(label)
            ax.set_title(frontend)
            ax.grid(axis='x', alpha=0.3)
    fig.suptitle('Same fusion design with three frontend observation streams | all frames; lower is better', y=1.02)
    save(fig, data, '01_pooled_comparison')


def ecdf(data):
    fig, axes = plt.subplots(len(data['groups']), 2, figsize=(12, 4 * len(data['groups'])), sharex='col', sharey=True, squeeze=False)
    for i, frontend in enumerate(data['groups']):
        for j, metric in enumerate(('translation_error_m', 'rotation_error_deg')):
            ax = axes[i, j]
            for method in CURVES:
                values = np.sort(data['arrays'][f'{frontend}|{method}|{metric}']) * (1000 if j == 0 else 1)
                # Display decimation only; exact metrics/quantiles use every frame.
                index = np.unique(np.linspace(0, len(values)-1, min(2500, len(values))).astype(int))
                n = data['groups'][frontend]['all_subjects']['methods'][method]['n_expected']
                ax.plot(values[index], (index+1)/n, color=COLORS[method], linewidth=1.2,
                        label=LABELS[method], linestyle='--' if method == 'stored_fusion' else '-')
            if j == 0:
                ax.set_xscale('symlog', linthresh=1)
            else:
                ax.set_xlim(0, 180)
            ax.set_ylim(0, 1.01)
            ax.set_title(frontend)
            ax.set_xlabel('Translation error (mm; symlog)' if j == 0 else 'Rotation error (deg)')
            ax.set_ylabel('Fraction of all target frames')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc='lower right')
    fig.suptitle('Error CDF | full tails retained; missing estimates prevent reaching 1', y=1.02)
    save(fig, data, '02_error_cdf')


def per_object(data):
    fig, axes = plt.subplots(2, len(data['groups']), figsize=(17, 8), sharex=True, sharey='row', squeeze=False)
    objects = sorted(data['objects']['object'].unique())
    for j, frontend in enumerate(data['groups']):
        for i, metric in enumerate(('translation_mean_mm', 'failure_rate_including_missing')):
            ax = axes[i, j]
            for method in ('sensor_cam1', 'mean_equal', 'full'):
                df = data['objects'].query('frontend == @frontend and method == @method').set_index('object').loc[objects]
                values = df[metric].to_numpy() * (100 if i else 1)
                ax.plot(objects, values, '-o', color=COLORS[method], markersize=3, linewidth=1.2, alpha=0.8, label=LABELS[method])
            ax.set_title(frontend)
            ax.set_ylabel('Translation mean (mm)' if i == 0 else 'Failure (%)')
            # Set shared bounds from ALL panels, not the first frontend; setting
            # only bottom=0 disables subsequent autoscaling and clips MegaPose.
            upper = data['objects'].query("method in ['sensor_cam1', 'mean_equal', 'full']")[metric].max()
            ax.set_ylim(0, 100 if i == 1 else 1.08 * upper)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle('Every H2O object | frame-pooled within object; cam1 is fixed, not selected per object', y=1.02)
    save(fig, data, '03_per_object')


def paired_takes(data):
    fig, axes = plt.subplots(len(data['groups']), 2, figsize=(13, 4 * len(data['groups'])), squeeze=False)
    for i, frontend in enumerate(data['groups']):
        part = data['takes'].query('frontend == @frontend')
        for j, metric in enumerate(('translation_mean_mm', 'rotation_mean_deg')):
            wide = part.pivot(index=['subject', 'take'], columns='method', values=metric)
            ax = axes[i, j]
            for method in ('sensor_cam1', 'mean_equal'):
                delta = (wide['full'] - wide[method]).dropna().sort_values().to_numpy()
                better, worse = (delta < -1e-6).sum(), (delta > 1e-6).sum()
                ax.plot(np.arange(1, len(delta)+1), delta, '-o', markersize=2, linewidth=1.2,
                        color=COLORS[method], alpha=0.75,
                        label=f'vs {LABELS[method]}: {better} improved / {len(delta)-better-worse} tied / {worse} worse')
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_yscale('symlog', linthresh=1)
            ax.set_ylabel('Fusion minus baseline (mm)' if j == 0 else 'Fusion minus baseline (deg)')
            ax.set_xlabel('Take rank (each comparator sorted independently)')
            ax.set_title(frontend + ' | negative favors fusion')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    fig.suptitle('Paired take means | all improvements and regressions retained; symlog scale', y=1.02)
    save(fig, data, '04_paired_take_gains')


def ablation_deltas(data):
    metrics = ['translation_mean_mm', 'translation_p95_mm', 'rotation_p95_deg', 'failure_rate_including_missing']
    fig, axes = plt.subplots(len(data['groups']), 4, figsize=(17, 4.5 * len(data['groups'])), sharex='col', squeeze=False)
    for i, (frontend, groups) in enumerate(data['groups'].items()):
        base = groups['all_subjects']['methods']['full']
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            values = [scaled(metric, groups['all_subjects']['methods'][m][metric] - base[metric]) for m in ABLATIONS[1:]]
            ax.barh(np.arange(len(values)), values, color=['#d62728' if v >= 0 else '#1f77b4' for v in values], alpha=0.75)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_yticks(np.arange(len(values)), [LABELS[m] for m in ABLATIONS[1:]], fontsize=8)
            ax.invert_yaxis()
            ax.set_title(frontend)
            ax.set_xlabel(['Delta T mean (mm)', 'Delta T P95 (mm)', 'Delta R P95 (deg)', 'Delta failure (pp)'][j])
            ax.grid(axis='x', alpha=0.3)
    fig.suptitle('Ablation minus default | positive: removing component hurts; negative: ablation improves', y=1.02)
    save(fig, data, '05_ablation_deltas')


def leave_one_out(data):
    methods = ['full'] + [f'without_cam{i}' for i in range(5)]
    metrics = ['translation_mean_mm', 'translation_p95_mm', 'rotation_p95_deg', 'failure_rate_including_missing']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, metric, label in zip(axes, metrics, ['T mean (mm)', 'T P95 (mm)', 'R P95 (deg)', 'Failure (%)']):
        for frontend, color in FRONTEND_COLORS.items():
            values = [scaled(metric, data['groups'][frontend]['all_subjects']['methods'][m][metric]) for m in methods]
            ax.plot(np.arange(6), values, '-o', color=color, markersize=3, linewidth=1.2, alpha=0.8, label=frontend)
        ax.set_xticks(np.arange(6), ['All 5'] + [f'No {i}' for i in range(5)], rotation=30)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Leave-one-camera-out | whole-sequence removal, not a temporary occlusion test', y=1.03)
    save(fig, data, '06_leave_one_out')


def benefit_summary(data):
    """Headline means plus joint failure; tails are in the adjacent table/CDF."""
    names = list(data['groups'])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, metric, title in zip(axes, ['translation_mean_mm', 'rotation_mean_deg', 'failure_rate_including_missing'],
                                ['Translation mean (mm)', 'Rotation mean (deg)', 'Failure (%)']):
        for j, method in enumerate(['mean_equal', 'full']):
            values = [scaled(metric, data['groups'][f]['all_subjects']['methods'][method][metric]) for f in names]
            bars = ax.bar(np.arange(len(names)) + (j - 0.5) * 0.35, values, width=0.35,
                          color=COLORS[method], alpha=0.75, label=LABELS[method])
            ax.bar_label(bars, labels=[f'{v:.1f}' for v in values], fontsize=8, padding=3)
        ax.set_xticks(np.arange(len(names)), names, fontsize=9)
        ax.set_ylabel(title)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.16)
        if metric == 'failure_rate_including_missing':
            ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)
    save(fig, data, '00_cross_frontend_benefits')


def take_intervals(data):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, metric, label in zip(axes, ['translation_mean_mm', 'rotation_mean_deg', 'failure_rate_including_missing'],
                                ['Delta mean translation (mm)', 'Delta mean rotation (deg)', 'Delta failure (pp)']):
        for i, (frontend, groups) in enumerate(data['groups'].items()):
            row = groups['all_subjects']['comparisons']['full_minus']['mean_equal'][metric]
            scale = 100 if metric.startswith('failure') else 1
            value = row['macro_mean_delta'] * scale
            lo, hi = np.array(row['macro_mean_bootstrap_ci95']) * scale
            ax.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt='o', capsize=4,
                        color=FRONTEND_COLORS[frontend], linewidth=1.2, markersize=4)
        ax.set_yticks(range(len(data['groups'])), list(data['groups']))
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    save(fig, data, '07_take_bootstrap')


def complementarity(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, metric, label in zip(axes, ['translation_mean_mm', 'failure_rate_including_missing'],
                                ['Increase in translation mean (mm)', 'Increase in failure (pp)']):
        for frontend, groups in data['groups'].items():
            m = groups['all_subjects']['methods']
            values = [scaled(metric, m[f'without_cam{i}'][metric] - m['full'][metric]) for i in range(5)]
            ax.plot(range(5), values, '-o', markersize=3, linewidth=1.2,
                    color=FRONTEND_COLORS[frontend], label=frontend, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(range(5), [f'No cam{i}' for i in range(5)])
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save(fig, data, '08_camera_complementarity')


def stability_table(data):
    table = pd.DataFrame(data['stability']).groupby('frontend', sort=False).agg(
        takes=('takes', 'sum'), frames=('frames', 'sum'),
        identical_order_takes=('identical_order_takes', 'sum'), max_pose_delta=('max_order_pose_delta', 'max'))
    table.to_csv(data['output'] / 'table_04_stability.csv')
    return table


def publication_tables(data):
    """One audited source for both notebook and manuscript numbers."""
    folder = data['output'] / 'paper_tables'
    folder.mkdir(exist_ok=True)
    specifications = {
        'main': (['sensor_cam1', 'sensor_cam3', 'mean_equal', 'full'], METRICS[:-1]),
        'ablation': (['full', 'no_adaptive', 'no_huber', 'no_robust', 'no_temporal', 'no_history'],
                     ['translation_mean_mm', 'rotation_mean_deg', 'failure_rate_including_missing']),
        'revision': (['stored_fusion', 'full'], ['translation_mean_mm', 'translation_p95_mm', 'rotation_p95_deg', 'failure_rate_including_missing']),
    }
    for stem, (methods, metrics) in specifications.items():
        lines = []
        for frontend, groups in data['groups'].items():
            for method in methods:
                m = groups['all_subjects']['methods'][method]
                label = LABELS[method].replace('Ours (checkpoint)', 'Ours')
                values = [f'{scaled(k, m[k]):.2f}' for k in metrics]
                lines.append(' & '.join([frontend, label] + values) + r' \\')
            lines.append(r'\midrule')
        headings = {'translation_mean_mm': 'T mean (mm)', 'translation_p95_mm': 'T P95 (mm)',
                    'rotation_mean_deg': 'R mean (deg)', 'rotation_p95_deg': 'R P95 (deg)',
                    'failure_rate_including_missing': r'Failure (\%)'}
        table = [r'\begin{tabular}{ll' + 'r' * len(metrics) + r'}\toprule',
                 ' & '.join(['Frontend', 'Method'] + [headings[m] for m in metrics]) + r' \\\midrule',
                 *lines[:-1], r'\bottomrule\end{tabular}']
        (folder / f'{stem}_table.tex').write_text('\n'.join(table) + '\n', encoding='utf-8')
