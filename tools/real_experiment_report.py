"""Publication figures from frozen real-scene caches; never invokes fusion or tracking."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SESSIONS = ['stable-0', 'stable-1', 'stable-2', *[f'Test-{i}' for i in range(5)]]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/real_experiment_report'
METHODS = ['mean_equal', 'mean_markley_weighted', 'best_score', 'original', 'singleton_guard']
ABLATIONS = ['singleton_guard', 'singleton_no_temporal', 'singleton_no_adaptive', 'singleton_no_huber']
CAMERAS = ['sensor_134722070227', 'sensor_242322077348', 'sensor_quest']
LABELS = dict(zip(METHODS, ['Equal fusion', 'Weighted fusion', 'Best score', 'Original fusion', 'Ours + history guard']))
LABELS.update(dict(zip(ABLATIONS[1:], ['No temporal', 'No adaptive covariance', 'No Huber'])))
LABELS.update(dict(zip(CAMERAS, ['RealSense 1347', 'RealSense 2423', 'Quest'])))
COLORS = {'mean_equal': '#ff7f0e', 'mean_markley_weighted': '#9467bd', 'best_score': '#8c564b',
          'original': '#2ca02c', 'singleton_guard': '#d62728',
          CAMERAS[0]: '#7f7f7f', CAMERAS[1]: '#1f77b4', CAMERAS[2]: '#17becf'}


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(method: str, values: np.ndarray, **extra) -> dict:
    from tools.audit_new_real_runs import stats

    result = stats(values)
    return dict(method=method, n=result['n'],
                t50=result.get('t_p50_p95_mm', [None, None])[0],
                t95=result.get('t_p50_p95_mm', [None, None])[1],
                r50=result.get('r_p50_p95_deg', [None, None])[0],
                r95=result.get('r_p50_p95_deg', [None, None])[1],
                t_bad=result.get('translation_over_50mm'), r_bad=result.get('rotation_over_90deg'), **extra)


def prepare() -> dict:
    """Create a new numbered snapshot. All baselines share frozen observations."""
    from experiments.h2o.evaluation import allocate_results
    from tools.audit_new_real_runs import KEY, errors, nearest_distance

    output = allocate_results(PARENT)
    arrays, sources, coverage, timing, rows, missing, paired = {}, {}, [], [], [], [], []
    pooled = {m: [] for m in METHODS + ABLATIONS[1:]}
    for session in SESSIONS:
        run = ROOT / 'artifacts/new_real_singleton_ablation' / session / 'results_0'
        manifest = read(run / 'manifest.json')
        source = Path(manifest['source_run'])
        sm = read(source / 'manifest.json')
        if manifest['status'] != 'completed' or read(run / 'failures.json'):
            raise ValueError(f'Incomplete run: {run}')
        if fingerprint(source / 'observations.jsonl') != manifest['observations_sha256']:
            raise ValueError(f'Observations changed: {source}')
        preds = read(run / 'predictions.json')
        ref = ROOT / 'artifacts/new_real_reference/results_1' / f'{session}_reference_for_native_mesh.npz'
        sources[session] = {str(p.relative_to(ROOT)): fingerprint(p) for p in [
            run / 'predictions.json', run / 'manifest.json', run / 'coverage.json',
            run / 'fusion_stats.json', source / 'observations.jsonl', ref]}
        with np.load(ref) as refs:
            err = {m: errors(p, refs, sm['origin_host_s']) for m, p in preds.items()}
        expected = set(err['singleton_guard'])
        for method, values in err.items():
            ids = sorted(values)
            a = np.array([values[i] for i in ids], dtype=float).reshape(-1, 2)
            arrays[f'{session}|{method}|ids'] = np.array(ids, dtype=int)
            arrays[f'{session}|{method}|errors'] = a
            rows.append(record(method, a, session=session))
            if method in pooled:
                if set(ids) != expected:
                    raise ValueError(f'Unmatched target sets: {session}/{method}')
                pooled[method].append(a)
            shared = sorted(set(values) & expected)
            for role, name in [('candidate', 'singleton_guard'), ('comparator', method)]:
                paired.append(record(name, np.array([err[name][i] for i in shared]).reshape(-1, 2),
                                     session=session, comparator=method, role=role))
        observations = [json.loads(s) for s in (source / 'observations.jsonl').read_text().splitlines() if s]
        rs_times = np.array([r['capture_time'] for r in observations if r['camera_id'] == '242322077348'])
        ids = np.array(sorted(expected))
        unavailable = nearest_distance(rs_times, ids / 30) > .15
        for method in ['original', 'singleton_guard']:
            missing.append(record(method, np.array([err[method][int(i)] for i in ids[unavailable]]).reshape(-1, 2), session=session))
        cov = read(run / 'coverage.json')
        for method, value in cov.items():
            coverage.append(dict(session=session, method=method, **value))
        for method, value in read(run / 'fusion_stats.json').items():
            timing.append(dict(session=session, method=method, **value))
        arrays[f'{session}|prediction_ids'] = np.array([r['frame_id'] for r in preds['singleton_guard'] if r.get('prediction_only')], dtype=int)
        print(f'{session}: checked and scored cached outputs', flush=True)
    tables = {'main': pd.DataFrame([record(m, np.concatenate(a)) for m, a in pooled.items()]),
              'sessions': pd.DataFrame(rows), 'coverage': pd.DataFrame(coverage),
              'timing': pd.DataFrame(timing), 'missing_rs2423': pd.DataFrame(missing),
              'paired': pd.DataFrame(paired)}
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    np.savez_compressed(output / 'errors.npz', **arrays)
    report = {'sources': sources, 'reference': KEY,
              'note': 'Eight development sessions; fixed tagcube0/Tomato2 reference; invalid OTS excluded. '
                      'No observed RS2423 pose within +/-150 ms is a missing-observation proxy, not annotated occlusion. '
                      'Test-5 excluded. Timing is sequential handler wall time, not acquisition-to-display latency.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return load(output)


def load(output: Path, verify: bool = True) -> dict:
    output = Path(output)
    meta = read(output / 'report.json')
    if verify:
        for paths in meta['sources'].values():
            for path, sha in paths.items():
                if fingerprint(ROOT / path) != sha:
                    raise ValueError(f'Snapshot source changed: {path}')
    with np.load(output / 'errors.npz') as saved:
        arrays = {key: saved[key] for key in saved.files}
    return dict(output=output, arrays=arrays, **meta,
                **{key: pd.read_csv(output / f'{key}.csv') for key in
                   ['main', 'sessions', 'coverage', 'timing', 'missing_rs2423', 'paired']})


def save(fig, data: dict, name: str) -> None:
    folder = data['output'] / 'figures'
    folder.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f'{name}.png', dpi=170, bbox_inches='tight')
    fig.savefig(folder / f'{name}.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)


def table(data: dict, name: str) -> pd.DataFrame:
    """One table category per notebook cell; never merge unmatched sensor targets."""
    if name in ['main', 'ablation']:
        chosen = METHODS if name == 'main' else ABLATIONS
        frame = data['main'].set_index('method').loc[chosen].reset_index()
        frame['method'] = frame['method'].map(LABELS)
    elif name == 'best_sensor':
        frame = data['paired'].query('comparator == "sensor_242322077348"').copy()
        frame = frame[['session', 'role', 'n', 't95', 'r95']]
    elif name == 'coverage':
        frame = data['coverage'].query('method == "singleton_guard"')[[
            'session', 'target_groups', 'output_groups', 'fraction', 'prediction_only_groups']].copy()
    elif name == 'timing':
        frame = data['timing'].query('method == "singleton_guard"')[[
            'session', 'n_input', 'compute_ms_mean_per_observation', 'compute_ms_p95_per_observation']].copy()
    else:
        raise ValueError(name)
    frame.to_csv(data['output'] / f'table_{name}.csv', index=False)
    return frame.round(3)


def headline(data: dict) -> None:
    df = data['main'].set_index('method').loc[METHODS]
    fig, panels = plt.subplots(2, 2, figsize=(9, 5.4))
    axes = panels.ravel()
    for index, (ax, metric, title) in enumerate(zip(axes, ['t95', 'r95', 't_bad', 'r_bad'],
                                 ['Translation P95 (mm)', 'Rotation P95 (deg)', 'Translation >50 mm (count)', 'Rotation >90 deg (count)'])):
        y = np.arange(len(df))
        ax.hlines(y, 0, df[metric], color='#dddddd', linewidth=1)
        ax.scatter(df[metric], y, c=[COLORS[m] for m in df.index], s=25, zorder=3)
        for yi, value in zip(y, df[metric]):
            ax.annotate(f'{value:.2f}' if '95' in metric else str(int(value)), (value, yi), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8)
        ax.set_yticks(y, [LABELS[m] for m in df.index] if index % 2 == 0 else [])
        ax.invert_yaxis()
        ax.set_xlim(0, df[metric].max() * 1.3)
        ax.set_xlabel(title, fontsize=9)
        ax.grid(axis='x', alpha=.25)
    fig.suptitle('Real scenes | same 12,226 reference-valid time groups; lower is better', fontsize=11)
    save(fig, data, 'real_01_headline')


def ecdf(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for j, ax in enumerate(axes):
        for method in METHODS:
            values = np.sort(np.concatenate([data['arrays'][f'{s}|{method}|errors'][:, j] for s in SESSIONS]))
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), lw=1.1,
                    label=LABELS[method], color=COLORS[method], linestyle='--' if method == 'original' else '-')
        ax.set_xscale('symlog', linthresh=1)
        ax.set_xlabel(['Translation error (mm; symlog)', 'Rotation error (deg; symlog)'][j])
        ax.set_ylabel('Fraction of evaluated outputs')
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle('Full error tails retained | conditional on available output and valid OTS reference', fontsize=10)
    save(fig, data, 'real_02_cdf')


def per_session(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, unit in zip(axes, ['t95', 'r95'], ['mm', 'deg']):
        for method in ['mean_equal', 'mean_markley_weighted', 'original', 'singleton_guard']:
            frame = data['sessions'].query('method == @method').set_index('session').loc[SESSIONS]
            ax.plot(SESSIONS, frame[metric], 'o-', ms=3, lw=1, color=COLORS[method], label=LABELS[method])
        ax.set_ylabel(f'P95 ({unit})')
        ax.tick_params(axis='x', rotation=35)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle('All eight sessions | static, unobstructed-motion, and RealSense-occlusion protocols', fontsize=10)
    save(fig, data, 'real_03_sessions')


def ablation(data: dict) -> None:
    frame = data['main'].set_index('method')
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    for ax, metric, title in zip(axes, ['t95', 'r95', 't_bad', 'r_bad'], ['T P95 (mm)', 'R P95 (deg)', 'T >50 mm (count)', 'R >90 deg (count)']):
        delta = frame.loc[ABLATIONS[1:], metric] - frame.loc['singleton_guard', metric]
        ax.axvline(0, color='black', lw=.8)
        ax.hlines(range(3), 0, delta, color='#999999', lw=1)
        ax.scatter(delta, range(3), c=['#1f77b4', '#ff7f0e', '#9467bd'], s=25)
        ax.set_yticks(range(3), [LABELS[m] for m in ABLATIONS[1:]] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.grid(axis='x', alpha=.25)
    fig.suptitle('Ablation minus full candidate | positive = deterioration; negative values retained', fontsize=10)
    save(fig, data, 'real_04_ablation')


def missing_views(data: dict) -> None:
    frame = data['missing_rs2423'].query('method == "singleton_guard"').set_index('session').loc[SESSIONS[3:]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, metric, title in zip(axes, ['n', 't95', 'r95'], ['Evaluable supplemental groups', 'Supplemental T P95 (mm)', 'Supplemental R P95 (deg)']):
        ax.plot(frame.index, frame[metric], 'o-', ms=4, lw=1.1, color=COLORS['singleton_guard'])
        ax.set_ylabel(title)
        ax.grid(alpha=.25)
    fig.suptitle('No saved RS2423 observation within +/-150 ms | missing-observation proxy, not occlusion labels', fontsize=9)
    save(fig, data, 'real_05_missing_views')


def failure_traces(data: dict) -> None:
    windows = [('stable-0', 22, 31.3), ('Test-2', 47.5, 49.5), ('Test-4', 37.5, 39.5)]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2), sharex='col')
    for col, (session, lo, hi) in enumerate(windows):
        for method in [CAMERAS[1], 'original', 'singleton_guard']:
            ids = data['arrays'][f'{session}|{method}|ids']
            values = data['arrays'][f'{session}|{method}|errors']
            # NaNs break both absent-observation and invalid-reference gaps.
            grid = np.arange(int(lo * 30), int(hi * 30) + 1)
            indexed = dict(zip(ids, values))
            plotted = np.array([indexed.get(i, [np.nan, np.nan]) for i in grid])
            for j in range(2):
                axes[j, col].plot(grid / 30, plotted[:, j], '.-', ms=2, lw=.8,
                                  color=COLORS[method], label=LABELS[method])
        pred = data['arrays'][f'{session}|prediction_ids'] / 30
        for t in pred[(pred >= lo) & (pred <= hi)]:
            for j in range(2):
                axes[j, col].axvline(t, color='#d62728', lw=.7, alpha=.25)
        axes[0, col].set_title(f'{session}: ' + ['jumps suppressed', 'translation regression', 'expired support'][col], fontsize=10)
        axes[1, col].set_xlabel('Session time (s)')
        for j in range(2):
            axes[j, col].grid(alpha=.25)
            axes[j, col].set_ylabel(['T error (mm)', 'R error (deg)'][j])
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Diagnostic examples, not representative sampling | red vertical lines: prediction-only outputs', fontsize=10)
    save(fig, data, 'real_06_cases')


def timing(data: dict) -> None:
    frame = table(data, 'timing')
    fig, ax = plt.subplots(figsize=(8, 3.3))
    for metric, label, color in [('compute_ms_mean_per_observation', 'Mean', '#1f77b4'),
                                  ('compute_ms_p95_per_observation', 'Per-session P95', '#d62728')]:
        ax.plot(frame['session'], frame[metric], 'o-', lw=1.1, ms=3, label=label, color=color)
    ax.set_ylabel('Handler + optimizer wall time\n(ms / observation)')
    ax.legend()
    ax.grid(alpha=.25)
    ax.set_title('Recorded sequential replay timings | not live or capture-to-display latency', fontsize=10)
    save(fig, data, 'real_07_compute')


def publication_tables(data: dict) -> None:
    folder = data['output'] / 'paper_tables'
    folder.mkdir(exist_ok=True)
    for name, methods in [('real_main', METHODS), ('real_ablation', ABLATIONS)]:
        frame = data['main'].set_index('method').loc[methods]
        lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
                 r'Method & T P95 & R P95 & T $>50$ mm & R $>90^\circ$ \\', r'\midrule']
        for method, row in frame.iterrows():
            lines.append(f"{LABELS[method]} & {row.t95:.2f} & {row.r95:.2f} & {int(row.t_bad)} & {int(row.r_bad)} " + r'\\')
        lines += [r'\bottomrule', r'\end{tabular}']
        (folder / f'{name}.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.axisbelow': True})
    data = prepare()
    for function in [headline, ecdf, per_session, ablation, missing_views, failure_traces, timing, publication_tables]:
        function(data)
    for name in ['main', 'ablation', 'best_sensor', 'coverage', 'timing']:
        table(data, name)
    print('OUTPUT', data['output'].resolve())
