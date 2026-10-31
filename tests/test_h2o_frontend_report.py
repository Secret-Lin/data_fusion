"""Reporting regressions: retain every frontend and visible error tail."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from tools import h2o_frontend_report as report


def test_per_object_shared_limits_retain_largest_frontend(monkeypatch):
    records = []
    for frontend, error in zip(report.FRONTEND_COLORS, [7.0, 17.0, 300.0]):
        for method in ("sensor_cam1", "mean_equal", "full"):
            records.append(dict(frontend=frontend, method=method, object="lotion",
                                translation_mean_mm=error,
                                failure_rate_including_missing=0.9))
    captured = []
    monkeypatch.setattr(report, "save", lambda fig, data, name: captured.append(fig))
    report.per_object({"groups": dict.fromkeys(report.FRONTEND_COLORS),
                       "objects": pd.DataFrame(records)})
    fig = captured[0]
    try:
        assert len(fig.axes) == 6
        for ax in fig.axes[:3]:
            assert ax.get_ylim()[1] >= 300
            for line in ax.lines:
                assert max(line.get_ydata()) <= ax.get_ylim()[1]
        for ax in fig.axes[3:]:
            assert ax.get_ylim() == (0, 100)
    finally:
        plt.close(fig)


def test_publication_tables_include_all_frontends_and_scale_failure(tmp_path):
    methods = set(report.BASELINES + report.ABLATIONS + ["stored_fusion"])
    metrics = dict.fromkeys(report.METRICS, 1.0)
    metrics["failure_rate_including_missing"] = 0.125
    groups = {frontend: {"all_subjects": {"methods": dict.fromkeys(methods, metrics)}}
              for frontend in report.FRONTEND_COLORS}
    report.publication_tables({"output": tmp_path, "groups": groups})
    for name in ("main", "ablation", "revision"):
        table = (tmp_path / "paper_tables" / f"{name}_table.tex").read_text()
        assert table.startswith(r"\begin{tabular}")
        assert table.rstrip().endswith(r"\bottomrule\end{tabular}")
        for frontend in report.FRONTEND_COLORS:
            assert frontend in table
        assert "12.50" in table
