"""Publication views of frozen real-data results; no tracking or fusion imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FRONTENDS = ['FoundationPose', 'M3T', 'MegaPose']
METHODS = ['mean_equal', 'original', 'singleton_guard', 'supported_history',
           'singleton_no_temporal', 'singleton_no_adaptive', 'singleton_no_huber']
LABELS = dict(zip(METHODS, ['Equal fusion', 'Original fusion', 'Ours + guard',
                          'Supported history', 'No temporal factors', 'No adaptive covariance', 'No Huber']))
COLORS = {'mean_equal': '#ff7f0e', 'singleton_guard': '#d62728'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare():
    from experiments.h2o.evaluation import allocate_results

    source = ROOT / 'artifacts/real_three_frontend_analysis/results_0'
    report = read(source / 'analysis.json')
    for session, frontends in report['sessions'].items():
        for frontend, entry in frontends.items():
            paths = [(Path(entry['ablation']) / 'predictions.json', entry['prediction_sha256']),
                     (ROOT / 'artifacts/new_real_reference/results_1' / f'{session}_reference_for_native_mesh.npz', entry['reference_sha256'])]
            for path, expected in paths:
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Source changed: {session}/{frontend}/{path}')
    output = allocate_results(ROOT / 'artifacts/real_three_frontend_report')
    for name in ['analysis.json', 'observation_diagnostics.json', 'summary.md']:
        shutil.copy2(source / name, output / name)
    return load(output)


def load(output):
    output = Path(output)
    return {'output': output, 'analysis': read(output / 'analysis.json'),
            'diagnostics': read(output / 'observation_diagnostics.json')}


def metric(record, name):
    if name == 'tmean':
        return record['mean_translation_mm_rotation_deg'][0]
    if name == 'rmean':
        return record['mean_translation_mm_rotation_deg'][1]
    if name == 't95':
        return record['t_p50_p95_mm'][1]
    if name == 'r95':
        return record['r_p50_p95_deg'][1]
    raise ValueError(name)


def save(fig, data, name):
    folder = data['output'] / 'figures'
    folder.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ['png', 'pdf']:
        fig.savefig(folder / f'{name}.{extension}', dpi=170, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def headline(data):
    common = data['analysis']['cross_frontend_common']
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    for ax, name, title in zip(axes.flat, ['tmean', 't95', 'rmean', 'r95'],
                               ['Translation mean (mm)', 'Translation P95 (mm)', 'Rotation mean (deg)', 'Rotation P95 (deg)']):
        x = np.arange(3)
        for i, method in enumerate(COLORS):
            values = [metric(common[f][method], name) for f in FRONTENDS]
            bars = ax.bar(x + (i - .5) * .34, values, .34, color=COLORS[method], label=LABELS[method])
            ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
        ax.set_xticks(x, ['FP', 'M3T', 'MegaPose'])
        ax.set_ylabel(title)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.grid(axis='y', alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, .965))
    fig.suptitle('Physical recordings: matched 12,226 groups; lower is better', y=1.015)
    save(fig, data, 'real3_01_common')


def heatmap(data, values, row_labels, column_labels, titles, stem, note):
    limit = max(1., max(float(np.max(np.abs(v))) for v in values))
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.7, .43 * len(row_labels) + 1.5)))
    norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
    for ax, matrix, title in zip(axes, values, titles):
        ax.imshow(matrix, cmap='RdBu_r', norm=norm, aspect='auto')
        ax.set_xticks(np.arange(len(column_labels)), column_labels)
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        ax.set_title(title)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f'{value:+.1f}%', ha='center', va='center', fontsize=9,
                    color='white' if abs(value) > .65 * limit else '#111111')
    fig.suptitle(note + '\nBlue: lower error; red: higher error. Values are relative P95 changes.', fontsize=10, y=1.015)
    save(fig, data, stem)


def protocols(data):
    groups = ['static', 'unobstructed_protocol', 'realsense_occlusion_protocol']
    matrices = []
    for name in ['t95', 'r95']:
        matrices.append(np.array([[100 * (metric(data['analysis']['protocol_groups'][f][g]['singleton_guard'], name) /
                                          metric(data['analysis']['protocol_groups'][f][g]['mean_equal'], name) - 1)
                                   for f in FRONTENDS] for g in groups]))
    heatmap(data, matrices, ['Static', 'Unobstructed motion', 'RS occlusion protocol'],
            ['FP', 'M3T', 'MegaPose'], ['Translation P95 change', 'Rotation P95 change'], 'real3_02_protocols',
            'Ours + guard vs equal fusion; matched within frontend/protocol, not across frontends')


def ablations(data):
    methods = [m for m in METHODS if m not in ['mean_equal', 'singleton_guard']]
    common = data['analysis']['cross_frontend_common']
    matrices = [np.array([[100 * (metric(common[f][m], name) / metric(common[f]['singleton_guard'], name) - 1)
                           for f in FRONTENDS] for m in methods]) for name in ['t95', 'r95']]
    heatmap(data, matrices, [LABELS[m] for m in methods], ['FP', 'M3T', 'MegaPose'],
            ['Translation P95 change', 'Rotation P95 change'], 'real3_03_ablation',
            'Variant vs ours + guard; same 12,226 groups; supported history is a separate candidate')


def coverage(data):
    fig, ax = plt.subplots(figsize=(8, 3.7))
    source = data['analysis']['pooled']
    records = [source[f]['singleton_guard'] for f in FRONTENDS]
    good = np.array([(s['n'] - s['joint_bad_observed_count']) / s['targets'] * 100 for s in records])
    bad = np.array([s['joint_bad_observed_count'] / s['targets'] * 100 for s in records])
    missing = np.array([s['missing'] / s['targets'] * 100 for s in records])
    bottom = np.zeros(3)
    for vals, label, color in [(good, 'Within both thresholds', '#1f77b4'), (bad, 'Output above either threshold', '#d62728'), (missing, 'Missing output', '#bdbdbd')]:
        ax.barh(FRONTENDS, vals, left=bottom, color=color, label=label)
        for i, v in enumerate(vals):
            if v > 3:
                ax.text(bottom[i] + v/2, i, f'{v:.1f}%', ha='center', va='center', color='white', fontsize=9)
        bottom += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel('Share of 13,223 reference-valid input groups (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.24), ncol=1, frameon=False, fontsize=9)
    fig.suptitle('Ours + guard: availability is not accuracy\nJoint thresholds: translation 50 mm and rotation 10 deg; not clinical criteria', fontsize=10)
    save(fig, data, 'real3_04_coverage')


def quality(data):
    rows = data['diagnostics']['by_frontend_camera']
    totals = [sum(c['depth_fit_ge_0_9'] for c in rows[f].values()) for f in FRONTENDS]
    bad = [sum(c['high_depth_fit_rotation_gt_30deg'] for c in rows[f].values()) for f in FRONTENDS]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    values = np.array(bad) / totals * 100
    bars = ax.bar(FRONTENDS, values, color=['#1f77b4', '#d62728', '#ff7f0e'], width=.55)
    ax.bar_label(bars, labels=[f'{v:.2f}%\n{b:,}/{n:,}' for v, b, n in zip(values, bad, totals)], padding=4, fontsize=7)
    ax.set_ylim(0, 108)
    ax.set_ylabel('Rotation error >30 deg (%)', fontsize=9)
    ax.set_xticks(range(3), ['FP', 'M3T', 'MegaPose'], fontsize=9)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', alpha=.2)
    fig.suptitle('High depth fit: inlier fraction >=0.9', fontsize=9)
    save(fig, data, 'real3_05_quality')


def table(data, kind):
    a = data['analysis']
    rows = []
    if kind in ['main', 'ablation']:
        methods = ['mean_equal', 'singleton_guard'] if kind == 'main' else METHODS[1:]
        for f in FRONTENDS:
            for m in methods:
                s = a['cross_frontend_common'][f][m]
                rows.append(dict(frontend=f, method=LABELS[m], n=s['n'],
                                 **{k: metric(s, k) for k in ['tmean', 't95', 'rmean', 'r95']},
                                 joint_bad_percent=s['joint_bad_observed_fraction'] * 100))
    elif kind in ['sensors', 'leave_one_out']:
        prefix = 'sensor_' if kind == 'sensors' else 'singleton_without_'
        for f in FRONTENDS:
            for m, pairs in a['paired_sensors'][f].items():
                if m.startswith(prefix):
                    for role, s in pairs.items():
                        rows.append(dict(frontend=f, comparator=m, role=role, n=s['n'],
                                         **{k: metric(s, k) for k in ['tmean', 't95', 'rmean', 'r95']}))
    else:
        raise ValueError(kind)
    return pd.DataFrame(rows).round(3)


def publication_tables(data):
    folder = data['output'] / 'paper_tables'
    folder.mkdir(exist_ok=True)
    for kind in ['main', 'ablation']:
        methods = ['mean_equal', 'singleton_guard'] if kind == 'main' else METHODS[1:]
        keys = ['tmean', 't95', 'rmean', 'r95'] if kind == 'main' else ['t95', 'r95']
        headings = {'tmean': 'T mean', 't95': 'T P95', 'rmean': 'R mean', 'r95': 'R P95'}
        lines = [r'\begin{tabular}{ll' + 'r'*len(keys) + '}', r'\toprule',
                 ' & '.join(['Frontend', 'Method'] + [headings[k] for k in keys]) + r' \\\midrule']
        for f in FRONTENDS:
            for m in methods:
                s = data['analysis']['cross_frontend_common'][f][m]
                lines.append(' & '.join([f, LABELS[m]] + [f'{metric(s, k):.2f}' for k in keys]) + r' \\')
            lines.append(r'\midrule')
        lines[-1:] = [r'\bottomrule\end{tabular}']
        (folder / f'real3_{kind}.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


FIGURES = [headline, protocols, ablations, coverage, quality]


if __name__ == '__main__':
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.axisbelow': True})
    data = prepare()
    for function in [*FIGURES, publication_tables]:
        function(data)
    print('OUTPUT', data['output'])
