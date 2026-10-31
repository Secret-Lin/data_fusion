from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.h2o.ablation import (
    VARIANTS, generate_observations, load_take, main, observations_from_legacy,
    run_baselines, run_case, run_fusion,
)
from experiments.h2o.benchmark import _run
from experiments.h2o.dataset import H2OSequence
from experiments.h2o.evaluation import (
    MetricAccumulator, allocate_results, bootstrap_mean_ci, read_jsonl, score_poses, summarize,
)
from experiments.h2o.smoke_fixture import create_smoke_sequence


@pytest.fixture
def inputs(tmp_path: Path) -> tuple:
    take = create_smoke_sequence(tmp_path / "dataset", frames=12)
    gt, clean, cameras = load_take(take, 0, 30.0)
    observations = generate_observations(clean)
    return take, gt, clean, cameras, observations


def test_annotations_are_unchanged_without_noise_dropout_delay_or_score_ranking(inputs: tuple) -> None:
    _, _, clean, _, observations = inputs
    subset = generate_observations([r for r in clean if r["camera_id"] == "cam2"],
                                   take_key="subject/h1/0", seed=7)
    assert subset == [r for r in observations if r["camera_id"] == "cam2"]
    assert observations == generate_observations(clean, take_key="different_take", seed=17)
    original = {(r["frame_id"], r["camera_id"]): r for r in clean}
    assert len(observations) == len(clean)
    for row in observations:
        assert row["pose_world"] == original[row["frame_id"], row["camera_id"]]["pose_world"]
        assert row["arrival_time"] == row["t_host"]
        assert row["score"] == 1.0
        assert row["synthetic_corruption"] is False
    assert generate_observations([]) == []
    for scenario in ("nominal", "outliers", "occlusion", "delay60"):
        with pytest.raises(ValueError, match="removed"):
            generate_observations(clean, scenario=scenario)


def test_missing_frames_count_as_failure_and_single_never_falls_back(inputs: tuple) -> None:
    _, gt, _, cameras, observations = inputs
    observations = [r for r in observations if r["frame_id"] != 4 and r["camera_id"] != "cam0"]
    methods = run_baselines(observations, cameras)
    assert methods["sensor_cam0"] == []
    metric = summarize(score_poses(gt, methods["sensor_cam0"]))
    assert metric["failure_rate_including_missing"] == 1
    assert metric["coverage"] == 0
    assert metric["translation_p95_mm"] is None
    assert 4 not in {r["frame_id"] for r in methods["mean_equal"]}
    assert summarize(score_poses(gt, methods["mean_equal"]))["n_missing"] >= 1


def test_markley_mean_handles_rotation_wraparound() -> None:
    from experiments.h2o.benchmark import _rotvec_to_matrix
    rows = []
    for i, angle in enumerate([179, -179]):
        T = np.eye(4)
        T[:3, :3] = _rotvec_to_matrix(np.array([0, 0, np.deg2rad(angle)]))
        rows.append({"frame_id": 0, "object_name": "can", "camera_id": f"cam{i}",
                     "pose_world": T.tolist(), "score": 1})
    result = run_baselines(rows, ["cam0", "cam1"])["mean_equal"][0]
    assert np.asarray(result["pose_world"])[0, 0] == pytest.approx(-1)


def test_compact_aggregation_preserves_pooled_quantiles(inputs: tuple) -> None:
    _, gt, _, cameras, observations = inputs
    predictions = run_baselines(observations, cameras)["sensor_cam4"]
    rows = score_poses(gt, predictions)
    accumulator = MetricAccumulator()
    accumulator.add(rows[:5])
    accumulator.add(rows[5:])
    assert accumulator.result() == summarize(rows)
    assert bootstrap_mean_ci([1.0]) is None


def test_offline_runner_matches_legacy_fusion_exactly(inputs: tuple, tmp_path: Path) -> None:
    take, _, _, cameras, _ = inputs
    legacy = tmp_path / "legacy"
    asyncio.run(_run(sequence=H2OSequence(take, cameras), output_dir=legacy,
                     max_frames=12, fps=30.0, seed=7, synthetic_noise=False))
    observations = observations_from_legacy(legacy, set(range(12)))
    # Explicit legacy control: H2O's current default uses revision checkpoints,
    # while the old annotation runner intentionally retains its original config.
    predictions, _ = asyncio.run(run_fusion(observations, VARIANTS["full"], fps=30,
                                            base_overrides={"enable_world_revision_checkpoint": False}))
    old = {r["frame_id"]: r for r in read_jsonl(legacy / "fusion_full.jsonl")}
    assert len(predictions) == len(old)
    for row in predictions:
        np.testing.assert_allclose(row["pose_world"], old[row["frame_id"]]["pose_world"], atol=1e-10)


def test_integrity_is_pose_neutral_and_no_grouping_removes_fusion(inputs: tuple) -> None:
    _, _, _, _, observations = inputs
    full, _ = asyncio.run(run_fusion(observations, VARIANTS["full"], fps=30))
    disabled, _ = asyncio.run(run_fusion(observations, VARIANTS["no_integrity"], fps=30))
    for a, b in zip(full, disabled):
        np.testing.assert_allclose(a["pose_world"], b["pose_world"], atol=1e-12)
        assert b["integrity"] == {}
    separate, _ = asyncio.run(run_fusion(observations, VARIANTS["no_grouping"], fps=30))
    last = {r["frame_id"]: r for r in observations}
    assert all(r["observation_count"] == 1 for r in separate)
    for row in separate:
        np.testing.assert_allclose(row["pose_world"], last[row["frame_id"]]["pose_world"], atol=1e-8)


def test_no_history_is_distinct_from_no_temporal(inputs: tuple) -> None:
    _, _, _, _, observations = inputs
    temporal_off, _ = asyncio.run(run_fusion(observations, VARIANTS["no_temporal"], fps=30))
    history_off, _ = asyncio.run(run_fusion(observations, VARIANTS["no_history"], fps=30))
    # The two switches have different semantics even when a benign fixture
    # happens to produce similar poses: no_temporal keeps consensus history,
    # while no_history removes it as well.
    assert VARIANTS["no_temporal"].overrides["enable_temporal_factors"] is False
    assert "enable_consensus_history_prediction" not in VARIANTS["no_temporal"].overrides
    assert VARIANTS["no_history"].overrides["enable_consensus_history_prediction"] is False
    assert len(temporal_off) == len(history_off) == len({r["frame_id"] for r in observations})


def test_revisable_consensus_excludes_same_frame_outlier_without_dropping_state(inputs: tuple) -> None:
    _, _, _, _, original = inputs
    # A hand-built unit-test fault, never injected into dataset experiments.
    observations = json.loads(json.dumps(original))
    for row in observations:
        if row["camera_id"] == "cam4" and row["frame_id"] == 4:
            row["pose_world"][0][3] += 0.3
    full_rows, full = asyncio.run(run_fusion(observations, VARIANTS["full"], fps=30))
    ungated_rows, ungated = asyncio.run(run_fusion(observations, VARIANTS["no_gate"], fps=30))
    # Every raw input remains available to the revisable consensus stage. The
    # outlier is excluded from the selected support set and is reported in the
    # state evidence, instead of causing the state to disappear.
    assert full["n_rejected"] == 0
    assert ungated["n_rejected"] == 0
    full_frame = next(row for row in full_rows if row["frame_id"] == 4)
    assert "h2o/cam4" in full_frame["integrity"]["excluded_sensors"]
    assert full_frame["observation_count"] == 4


def test_all_case_variants_resume_and_reject_changed_inputs(inputs: tuple, tmp_path: Path) -> None:
    _, gt, _, cameras, observations = inputs
    directory = tmp_path / "case"
    variants = list(VARIANTS.values())
    report = asyncio.run(run_case(observations, gt, cameras, variants, directory,
                                 fps=30, manifest={"test": 1}))
    assert report["methods"]["no_temporal"]["status"] == "completed"
    assert report["methods"]["no_online_calibration"]["status"] == "not_applicable"
    for name, method in report["methods"].items():
        if method["status"] == "completed":
            assert method["metrics"]["n_expected"] == len(gt)
            assert (directory / name / "poses.jsonl.gz").is_file()
    assert asyncio.run(run_case(observations, gt, cameras, variants, directory,
                               fps=30, manifest={"test": 1})) == report
    with pytest.raises(ValueError, match="mismatch"):
        asyncio.run(run_case(observations[:-1], gt, cameras, variants, directory,
                             fps=30, manifest={"test": 1}))


def test_subject_cli_summary_and_sequential_result_allocation(inputs: tuple, tmp_path: Path) -> None:
    take, _, _, _, _ = inputs
    first, second = allocate_results(tmp_path), allocate_results(tmp_path)
    assert first.name == "results_0"
    assert second.name == "results_1"
    output = tmp_path / "results_2"
    argv = ["--subject-root", str(take.parents[1]), "--output", str(output), "--seeds", "7",
            "--scenarios", "annotations", "--variants", "full,no_integrity,no_temporal",
            "--max-frames", "8"]
    main(argv)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_take_cases"] == 1
    assert summary["artificial_noise"] is False
    assert summary["scenarios"]["annotations"]["full"]["metrics"]["coverage"] == 1
    assert (output / "summary.md").is_file()
    assert (output / "per_take.csv").is_file()
    main([*argv, "--resume"])
    with pytest.raises(ValueError, match="not empty"):
        main(argv)
    with pytest.raises(ValueError, match="Cannot resume"):
        main([*argv, "--resume", "--seeds", "17"])
    with pytest.raises(ValueError, match="Artificial scenarios removed"):
        main([*argv, "--scenarios", "outliers"])
    with pytest.raises(ValueError, match="not repeated seeds"):
        main([*argv, "--seeds", "7,17"])
    with pytest.raises(ValueError, match="Legacy synthetic inputs"):
        main([*argv, "--legacy-results", str(output)])
