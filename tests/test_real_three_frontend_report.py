"""Ensure publication tables retain every frontend and unfavourable results."""
from tools import real_three_frontend_report as report


def test_main_table_uses_common_targets_and_keeps_regressions(tmp_path):
    common = {}
    for frontend in report.FRONTENDS:
        common[frontend] = {}
        for method in report.METHODS:
            value = 1. if method == 'mean_equal' else 2.
            common[frontend][method] = {
                'n': 10, 'mean_translation_mm_rotation_deg': [value, value * 3],
                't_p50_p95_mm': [value, value * 2], 'r_p50_p95_deg': [value, value * 4],
                'joint_bad_observed_fraction': .1}
    data = {'output': tmp_path, 'analysis': {'cross_frontend_common': common}}
    table = report.table(data, 'main')
    assert len(table) == 6
    assert set(table.n) == {10}
    assert list(table.query('method == "Ours + guard"').t95) == [4., 4., 4.]
    report.publication_tables(data)
    main = (tmp_path / 'paper_tables/real3_main.tex').read_text()
    ablation = (tmp_path / 'paper_tables/real3_ablation.tex').read_text()
    for frontend in report.FRONTENDS:
        assert main.count(frontend) == 2
        assert ablation.count(frontend) == 6
    assert main.startswith(r'\begin{tabular}')
    assert main.rstrip().endswith(r'\end{tabular}')
