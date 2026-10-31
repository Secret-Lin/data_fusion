import json

import pytest

from tools.analyze_h2o_subjects import checked_take_index, paired_comparisons
from tools.audit_h2o_fusion import COMPARISON_METRICS


def test_subject_pairing_keeps_repeated_take_names_and_missing_metrics():
    takes = []
    for subject, fusion, sensor in [("subject1", 1.0, 3.0), ("subject2", 5.0, 4.0)]:
        takes.append({"subject": subject, "take": "h1_0", "methods": {
            "stored_fusion": dict.fromkeys(COMPARISON_METRICS, fusion),
            "sensor_cam1": dict.fromkeys(COMPARISON_METRICS, sensor),
        }})
    takes[1]["methods"]["sensor_cam1"]["rotation_mean_deg"] = None
    result = paired_comparisons(takes)["sensor_cam1"]
    translation = result["translation_mean_mm"]
    assert translation["n_takes"] == 2
    assert translation["first_better"] == translation["second_better"] == 1
    assert translation["macro_mean_delta"] == pytest.approx(-0.5)
    assert result["rotation_mean_deg"]["takes_without_metric"] == 1
    assert result["rotation_mean_deg"]["macro_mean_bootstrap_ci95"] is None


def test_completed_index_rejects_silent_take_omission(tmp_path):
    take = tmp_path / "takes" / "h1_0"
    take.mkdir(parents=True)
    (take / "ground_truth.jsonl").write_text(json.dumps({"frame_id": 0}) + "\n")
    complete = {"n_takes_failed": 0, "n_takes_requested": 1, "n_takes_completed": 1, "failed_takes": [],
                "per_take": [{"take": "h1/0"}]}
    assert checked_take_index(tmp_path, complete) == [("h1_0", take)]
    for changed in (
        {**complete, "n_takes_requested": 2, "per_take": [{"take": "h1/0"}, {"take": "h1/1"}]},
        {**complete, "n_takes_failed": 1},
        {**complete, "n_takes_completed": 0},
        {**complete, "per_take": [{"take": "h1/0"}, {"take": "h1/0"}]},
    ):
        with pytest.raises(ValueError, match="Incomplete"):
            checked_take_index(tmp_path, changed)
