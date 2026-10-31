"""Recompute bounded timing summaries from traces, without loading RGB-D/GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(run_dir: Path) -> dict:
    """Raise on inconsistent traces/counts/statistics; return a compact report."""
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    traces = {name: [json.loads(line) for line in (run_dir / f'{name}.jsonl').read_text().splitlines()]
              for name in ('frames', 'detections', 'fused', 'receiver', 'backlog')}
    stage_file = run_dir / 'stages.jsonl'
    traces['stages'] = ([json.loads(line) for line in stage_file.read_text().splitlines()]
                        if stage_file.exists() else [])

    def check(actual, expected, name):
        if not np.isclose(actual, expected, rtol=1e-9, atol=1e-7):
            raise ValueError(f'{run_dir.name}: {name}: {actual} != {expected}')

    def stats(values, expected, name):
        check(len(values), expected['n'], name + '.n')
        if len(values):
            for key, value in zip(('mean', 'p50', 'p95', 'max'),
                                  (np.mean(values), *np.percentile(values, [50, 95]), np.max(values))):
                check(value, expected[key], name + '.' + key)

    if manifest['status'] != 'completed' or not traces['fused']:
        raise ValueError('Not a completed timing run with fused outputs')
    check(len(traces['frames']), manifest['input_frames'], 'lossless count')
    check(len({(r['source'], r['frame_index']) for r in traces['frames']}),
          manifest['input_frames'], 'unique frames')
    check(len(traces['frames']), summary['processed_frames'], 'processed frames')
    for source, expected in manifest.get('input_frames_by_source', {}).items():
        check(sum(r['source'] == source for r in traces['frames']), expected, source)
    check(len(traces['fused']), summary['fresh_fused_outputs'], 'published updates')
    check(len(traces['receiver']), summary['udp_received'], 'received packets')
    if len({r['packet_id'] for r in traces['receiver']}) != len(traces['receiver']):
        raise ValueError('Duplicate packet identifiers')
    published = {r['update_id']: r for r in traces['fused']}
    for row in traces['receiver']:
        update = published[row['update_id']]
        if row['fused'] != update['fused'] or row['capture_due'] != update['capture_due']:
            raise ValueError('Receiver payload differs from published update')
        if not row['received'] >= row['sent'] >= row['published']:
            raise ValueError('Non-monotonic receipt timestamps')
    first = traces['fused'][0]
    epoch = (first['capture_due'] - first['fused']['t_capture']
             + manifest['start_s'] + manifest['warmup_capture_s'])
    for window in ('release_window', 'including_drain'):
        metrics = summary[window]
        rows = [r for r in traces['receiver'] if epoch <= r['received'] < epoch + metrics['window_s']]
        updates = len({r['update_id'] for r in rows})
        groups = len({(r['fused']['object_name'], r['fused']['sync_group']) for r in rows})
        check(len(rows), metrics['packets'], window + '.packets')
        for key, count in [('packet_hz', len(rows)), ('distinct_update_hz', updates),
                           ('distinct_capture_group_hz', groups)]:
            check(count / metrics['window_s'], metrics[key], window + '.' + key)
        check(1 - updates / len(rows), metrics['repeat_fraction'], window + '.repeats')
        stats([(r['received'] - r['capture_due']) * 1000 for r in rows], metrics['result_age_ms'], window + '.age')
        stats([(r['received'] - r['sent']) * 1000 for r in rows], metrics['udp_loopback_ms'], window + '.udp')
    for key in ('frontend_ms', 'release_lag_ms'):
        stats([r[key] for r in traces['frames']], summary[key], key)
    stats([r['queue_ms'] for r in traces['frames']], summary['frontend_queue_ms'], 'queue')
    stats([r['fusion_ms'] for r in traces['detections']], summary['fusion_ms'], 'fusion')
    stats([(r['published'] - r['capture_due']) * 1000 for r in traces['fused']],
          summary['release_to_fused_ms'], 'release_to_fused')
    check(max(r['pending'] for r in traces['backlog']), summary['max_pending_frames'], 'peak queue')
    check(sum(r['done'] <= epoch + manifest['measured_capture_s'] for r in traces['frames']),
          summary['processed_within_release_window'], 'processed within release')
    for stage, expected in summary.get('stages_ms', {}).items():
        stats([r['ms'] for r in traces['stages'] if r['stage'] == stage], expected, stage)
    for mode, expected in summary.get('foundationpose_modes_ms', {}).items():
        stats([r['ms'] for r in traces['stages'] if r['stage'] == 'foundationpose' and r['mode'] == mode],
              expected, mode)
    frozen = run_dir / 'benchmark_source.py'
    if frozen.exists():
        expected = next(v for k, v in manifest['code_sha256'].items() if k.endswith('benchmark_recorded_realtime.py'))
        if hashlib.sha256(frozen.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen benchmark code hash mismatch')
    return {'run': str(run_dir), 'status': 'passed', 'frames': len(traces['frames']),
            'packets': len(traces['receiver']), 'stages': len(traces['stages']),
            'source_count_manifest_present': 'input_frames_by_source' in manifest,
            'frozen_runner_present': frozen.exists()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps([audit(path) for path in args.runs], indent=2))


if __name__ == '__main__':
    main()
