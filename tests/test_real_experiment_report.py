import json

import numpy as np
import pytest

from tools import real_experiment_report as report


def test_record_reports_separate_tail_counts_and_exact_quantiles():
    values = np.array([[1., 2.], [51., 3.], [20., 91.]])
    result = report.record('example', values)
    assert result['n'] == 3
    assert result['t_bad'] == result['r_bad'] == 1
    assert result['t95'] == pytest.approx(np.percentile(values[:, 0], 95))
    assert result['r95'] == pytest.approx(np.percentile(values[:, 1], 95))


def test_empty_reference_subset_is_not_zero_error():
    result = report.record('example', np.empty((0, 2)))
    assert result['n'] == 0
    assert result['t95'] is None
    assert result['r_bad'] is None


def test_changed_snapshot_source_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    source = tmp_path / 'source.json'
    source.write_text('{}', encoding='utf-8')
    (tmp_path / 'report.json').write_text(json.dumps({
        'sources': {'session': {'source.json': report.fingerprint(source)}}}), encoding='utf-8')
    source.write_text('{"changed": true}', encoding='utf-8')
    with pytest.raises(ValueError, match='Snapshot source changed'):
        report.load(tmp_path)


def test_frozen_real_snapshot_matches_audited_counts():
    directory = report.PARENT / 'results_0'
    if not directory.exists():
        pytest.skip('Optional real-scene report artifacts are not installed')
    data = report.load(directory)
    main = data['main'].set_index('method')
    assert set(main['n']) == {12226}
    assert main.loc['singleton_guard', 't95'] == pytest.approx(18.852633400546388)
    assert main.loc['original', 'r_bad'] == 8
    assert main.loc['singleton_guard', 'r_bad'] == 2
    cov = data['coverage'].query('method == "singleton_guard"')
    assert cov['output_groups'].sum() == 14269
    assert cov['prediction_only_groups'].sum() == 19
    paired = data['paired'].query('comparator == "sensor_242322077348"')
    counts = paired.pivot(index='session', columns='role', values='n')
    assert (counts['candidate'] == counts['comparator']).all()
