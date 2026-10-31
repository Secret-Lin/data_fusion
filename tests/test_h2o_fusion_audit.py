import pytest

from tools.audit_h2o_fusion import paired_take_summary, validate_targets, verify_metrics


def test_paired_take_delta_sign_and_ties() -> None:
    result = paired_take_summary([1, 2, 5], [2, 2, 3])
    assert (result["first_better"], result["ties"], result["second_better"]) == (1, 1, 1)
    assert result["macro_mean_delta"] == pytest.approx(1 / 3)
    assert result["bootstrap_unit"] == "take"
    assert result == paired_take_summary([1, 2, 5], [2, 2, 3])


def test_paired_bootstrap_does_not_invent_within_take_independence() -> None:
    assert paired_take_summary([1], [3])["macro_mean_bootstrap_ci95"] is None
    assert paired_take_summary([1, 2, 3], [2, 3, 4])["macro_mean_bootstrap_ci95"] == [-1.0, -1.0]
    with pytest.raises(ValueError):
        paired_take_summary([1], [1, 2])
    with pytest.raises(ValueError):
        paired_take_summary([float("nan")], [1])


def test_target_validation_rejects_omissions_and_wrong_object() -> None:
    rows = [{"frame_id": 0, "object_name": "book"}, {"frame_id": 1, "object_name": "book"}]
    validate_targets(list(reversed(rows)), rows, "ok")
    for invalid in (rows[:1], [rows[0], rows[0]], [{"frame_id": 0, "object_name": "can"}, rows[1]]):
        with pytest.raises(ValueError, match="targets"):
            validate_targets(invalid, rows, "test")


def test_metric_verification_accepts_roundoff_not_missing_frames() -> None:
    verify_metrics({"mean": 0.1 + 0.2, "count": 3, "optional": None},
                   {"mean": 0.3, "count": 3, "optional": None}, "test")
    with pytest.raises(ValueError, match="mismatch"):
        verify_metrics({"count": 3}, {"count": 2}, "test")
