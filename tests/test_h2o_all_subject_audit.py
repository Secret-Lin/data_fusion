import pytest

from tools.audit_h2o_all_subjects import matched_comparison
from tools.audit_h2o_fusion import COMPARISON_METRICS


def test_all_subject_audit_pairs_each_take_without_collapsing_names():
    records = [{"subject": subject, "take": "h1_0", "methods": {
        "full": dict.fromkeys(COMPARISON_METRICS, full),
        "candidate": dict.fromkeys(COMPARISON_METRICS, candidate),
    }} for subject, full, candidate in [("subject1", 2, 1), ("subject2", 2, 4)]]
    compared = matched_comparison(records, "candidate", "full")
    translation = compared["translation_mean_mm"]
    assert translation["n_takes"] == 2
    assert translation["first_better"] == translation["second_better"] == 1
    assert translation["macro_mean_delta"] == pytest.approx(0.5)


def test_all_subject_audit_reports_absent_metrics_not_zero_error():
    records = [{"methods": {
        "full": dict.fromkeys(COMPARISON_METRICS, None),
        "candidate": dict.fromkeys(COMPARISON_METRICS, 2),
    }}]
    assert all(v is None for v in matched_comparison(records, "full", "candidate").values())
