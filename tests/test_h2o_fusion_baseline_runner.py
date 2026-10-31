"""Runner tests with generated unit fixtures, not experimental observations."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.fusion_baselines import BaselineConfig, BaselineRunError
from experiments.h2o.ablation import read_gzip
from experiments.h2o.evaluation import score_poses, write_json
from tools import run_h2o_fusion_baselines as runner


def fixture_cache(directory: Path) -> None:
    directory.mkdir()
    gt, observations = [], []
    for k in range(3):
        pose = np.eye(4); pose[2, 3] = .7
        gt.append(dict(frame_id=k, object_name="test", pose_world=pose.tolist()))
        observations.append(dict(frame_id=k, object_name="test", camera_id="cam0", score=1.,
                                 source="h2o/cam0/foundationpose/test", t_host=k/30., arrival_time=k/30.,
                                 pose_world=pose.tolist()))
    for name, rows in (("ground_truth.jsonl", gt), ("tracking_before_fusion.jsonl", observations)):
        (directory / name).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    write_json(directory / "summary.json", {"cameras": ["cam0"], "init_mode": "unit_fixture"})


def test_numerical_failure_preserves_denominator_and_other_methods(tmp_path, monkeypatch):
    fixture_cache(tmp_path / "input")
    code = runner.source_hashes()
    original = runner.run_baseline
    def failing(observations, name, *args):
        if name == "eskf_fixed":
            prefix = [{**observations[0]}]
            raise BaselineRunError("unit injected failure", prefix, {"frame_id": 1})
        return original(observations, name, *args)
    monkeypatch.setattr(runner, "run_baseline", failing)
    job = {"frontend": "foundationpose", "subject": "subject1", "take": "fixture",
           "input": str(tmp_path / "input"), "relative_output": "foundationpose/subject1/takes/fixture"}
    output = tmp_path / "output"
    report = runner.run_take(job, str(output), asdict(BaselineConfig()), 0, code)
    bad = report["methods"]["eskf_fixed"]
    assert bad["status"] == "failed"
    assert bad["metrics"]["n_expected"] == 3
    assert bad["metrics"]["n_missing"] == 2
    assert bad["metrics"]["failure_rate_including_missing"] == pytest.approx(2/3)
    assert report["methods"]["ours"]["status"] == "completed"
    assert report["common_output_metrics"]["ours"]["n"] == 1
    summary = runner.summarize_reports(output, [report], [], {"code_sha256": code})
    assert summary["status"] == "incomplete_or_failed"
    assert len(summary["failed_methods"]) == 1
    assert summary["methods_by_frontend"]["foundationpose"]["eskf_fixed"]["n_expected"] == 3


def test_common_outputs_are_not_method_specific_survivors():
    truth = [{"frame_id": k, "object_name": "x", "pose_world": np.eye(4).tolist()} for k in range(3)]
    first = score_poses(truth, truth[:2])
    second = score_poses(truth, truth[1:])
    common = runner.common_scores({"a": first, "b": second})
    assert [r["frame_id"] for r in common["a"]] == [1]
    assert [r["frame_id"] for r in common["b"]] == [1]


def test_empty_observation_take_is_retained(tmp_path):
    fixture_cache(tmp_path / "input")
    (tmp_path / "input/tracking_before_fusion.jsonl").write_text("", encoding="utf-8")
    job = {"frontend": "foundationpose", "subject": "subject1", "take": "empty",
           "input": str(tmp_path / "input"), "relative_output": "empty"}
    report = runner.run_take(job, str(tmp_path / "output"), asdict(BaselineConfig()), 0, runner.source_hashes())
    for name, item in report["methods"].items():
        assert item["metrics"]["n_missing"] == 3
        assert item["metrics"]["failure_rate_including_missing"] == 1.
        assert len(read_gzip(tmp_path / "output/empty" / name / "scores.jsonl.gz")) == 3


def test_zero_score_comparator_does_not_drop_take(tmp_path):
    fixture_cache(tmp_path / "input")
    source = tmp_path / "input/tracking_before_fusion.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row in rows:
        row["score"] = 0.
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    job = {"frontend": "foundationpose", "subject": "subject1", "take": "zero",
           "input": str(tmp_path / "input"), "relative_output": "zero"}
    report = runner.run_take(job, str(tmp_path / "output"), asdict(BaselineConfig()), 0, runner.source_hashes())
    assert report["status"] == "completed"
    assert report["zero_score_groups_uniform_simple_baseline"] == 3
