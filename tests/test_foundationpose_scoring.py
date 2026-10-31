from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adapters.foundationpose_runner import FoundationPoseRunner


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_vendor_rank_offset_and_tensor_conversion(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # The previous conversion silently returned constant fallback on CUDA.
    for raw in (-3.0, 0.0, 2.0):
        value = torch.tensor(100 + raw, device=device)
        assert FoundationPoseRunner._raw_score_logit(value) == raw
    with pytest.raises(ValueError, match="finite"):
        FoundationPoseRunner._raw_score_logit(torch.tensor(float("nan"), device=device))


def make_runner(tmp_path):
    repository = SimpleNamespace(resolve=lambda name: SimpleNamespace(name=name))
    runner = FoundationPoseRunner(tmp_path, model_repository=repository, score_tracking=True)
    runner._loaded = True
    estimator = SimpleNamespace(
        model_center=np.array([0.1, -0.2, 0.3]), mesh=None, mesh_tensors={}, diameter=0.4,
        pose_last=torch.eye(4), scores=torch.tensor([102.0]),
    )
    runner._estimators[("cam", "can")] = estimator
    runner._glctx = None
    calls = []

    def predict(**kwargs):
        calls.append(kwargs)
        return torch.tensor([97.0]), None

    runner._scorer = SimpleNamespace(predict=predict)
    return runner, estimator, calls


def test_cached_scoring_centers_mesh_once_without_advancing_tracker(tmp_path) -> None:
    runner, estimator, calls = make_runner(tmp_path)
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [0.4, 0.2, 1.0]
    original_pose = pose.copy()
    last = estimator.pose_last.clone()
    score = runner.score_pose(source="cam", object_name="can",
                              color_bgr=np.zeros((4, 6, 3), np.uint8),
                              depth_m=np.ones((4, 6)), camera_matrix=np.eye(3), pose_cam=pose)
    expected = pose.copy()
    expected[:3, 3] += pose[:3, :3] @ estimator.model_center
    np.testing.assert_allclose(calls[0]["ob_in_cams"][0], expected)
    np.testing.assert_array_equal(pose, original_pose)
    assert torch.equal(estimator.pose_last, last)
    assert score.raw_logit == -3.0 and score.status == "single_pose_rank"


def test_live_track_diagnostic_is_not_a_fusion_probability(tmp_path) -> None:
    runner, estimator, calls = make_runner(tmp_path)
    estimator.track_one = lambda **kwargs: np.eye(4)
    result = runner.process(source="cam", object_name="can", color_bgr=np.zeros((4, 6, 3), np.uint8),
                            depth_m=np.ones((4, 6)), camera_matrix=np.eye(3))
    assert result.score == 0.9
    assert result.scoring.raw_logit == -3.0
    assert len(calls) == 1
    runner.score_tracking = False
    result = runner.process(source="cam", object_name="can", color_bgr=np.zeros((4, 6, 3), np.uint8),
                            depth_m=np.ones((4, 6)), camera_matrix=np.eye(3))
    assert result.scoring.status == "disabled"
    assert len(calls) == 1


def test_score_failures_remain_visible_and_do_not_invent_confidence(tmp_path) -> None:
    runner, estimator, _ = make_runner(tmp_path)

    def failed_predict(**kwargs):
        raise RuntimeError("test scorer fault")

    runner._scorer.predict = failed_predict
    score = runner._track_score(estimator, np.zeros((4, 6, 3), np.uint8), np.ones((4, 6)), np.eye(3))
    assert score.raw_logit is None
    assert score.status == "error" and "test scorer fault" in score.error
    assert runner._registration_score(estimator).raw_logit == 2.0


def test_same_image_candidate_ranks_ignore_inter_camera_score_offsets() -> None:
    from tools.rerank_h2o_cached_poses import aggregate_view_ranks

    scores = np.array([[100.0, 101.0, 99.0], [-10.0, -9.0, -11.0]])
    ranks = aggregate_view_ranks(scores)
    assert np.argmin(ranks) == 1
    np.testing.assert_allclose(ranks, aggregate_view_ranks(scores + [[-1000], [2000]]))
    np.testing.assert_allclose(aggregate_view_ranks(np.ones((2, 3))), [0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="No scored views"):
        aggregate_view_ranks(np.full((2, 3), np.nan))


def test_depth_fit_diagnostic_is_render_only_and_does_not_change_tracker(tmp_path) -> None:
    runner, estimator, _ = make_runner(tmp_path)
    runner._render_fn = lambda **kwargs: (
        torch.zeros((1, 4, 6, 3)),
        torch.full((1, 4, 6), 1.0),
        None,
    )
    before = estimator.pose_last.clone()
    result = runner.measure_depth_fit(
        source="cam", object_name="can", depth_m=np.ones((4, 6), np.float32),
        camera_matrix=np.eye(3), pose_cam=np.eye(4),
    )
    assert result["status"] == "ok"
    assert result["observed_overlap_fraction"] == pytest.approx(1.0)
    assert result["depth_inlier_fraction"] == pytest.approx(1.0)
    assert result["depth_abs_error_median_m"] == pytest.approx(0.0)
    assert torch.equal(estimator.pose_last, before)
