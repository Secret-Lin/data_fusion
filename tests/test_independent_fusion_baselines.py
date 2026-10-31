"""Constructed math/unit fixtures only; never reported as H2O experiments."""
import asyncio
from copy import deepcopy

import gtsam
import numpy as np
import pytest

from config import FusionConfig
from experiments.fusion_baselines import (
    METHODS, BaselineConfig, BaselineRunError, Measurement, PoseVelocityFilter,
    batch_pose, canonical_groups, innovation, measurement, motion_jacobian,
    pose_factor, process_covariance, run_baseline,
)
from experiments.h2o.ablation import VARIANTS, run_fusion
from tools.run_h2o_foundationpose_ablation import compare_pose_outputs


def rows(n=5):
    output = []
    for k in range(n):
        for camera in range(3):
            pose = gtsam.Pose3.Expmap(np.array([0., 0., .02*k, .004*k+.001*camera, 0., .7]))
            output.append(dict(frame_id=k, object_name="test", camera_id=f"cam{camera}",
                               source=f"h2o/cam{camera}/foundationpose/test", score=.9,
                               t_host=k/30., arrival_time=k/30., pose_world=pose.matrix().tolist()))
    return output


def numerical_jacobian(function, n):
    eps = 1e-6
    return np.column_stack([(function(np.eye(n)[i]*eps)-function(-np.eye(n)[i]*eps))/(2*eps)
                            for i in range(n)])


def test_prediction_and_reset_jacobians():
    velocity = np.array([.3, -.5, .6, .1, .3, -.2])
    dt = .17
    nominal = gtsam.Pose3.Expmap(velocity*dt)
    def error(e):
        pose = gtsam.Pose3.Expmap(e[:6]).compose(gtsam.Pose3.Expmap((velocity+e[6:])*dt))
        return np.r_[gtsam.Pose3.Logmap(nominal.between(pose)), e[6:]]
    np.testing.assert_allclose(motion_jacobian(velocity, dt), numerical_jacobian(error, 12), atol=2e-9)
    reset = numerical_jacobian(lambda e: gtsam.Pose3.Logmap(
        gtsam.Pose3.Expmap(-velocity).compose(gtsam.Pose3.Expmap(velocity+e))), 6)
    np.testing.assert_allclose(reset, gtsam.Pose3.ExpmapDerivative(velocity), atol=2e-9)


def test_innovation_and_noise_jacobians():
    x = gtsam.Pose3.Expmap(np.array([.1, -.2, .3, .2, .1, 1.]))
    z = x.compose(gtsam.Pose3.Expmap(np.array([.2, .1, -.3, .01, -.05, .02])))
    obs = Measurement("a", z, np.eye(6)*.01)
    _, h, noise = innovation(x, obs)
    dh = numerical_jacobian(lambda e: gtsam.Pose3.Logmap(x.compose(gtsam.Pose3.Expmap(e)).between(z)), 6)
    dn = numerical_jacobian(lambda e: gtsam.Pose3.Logmap(x.between(z.compose(gtsam.Pose3.Expmap(e)))), 6)
    np.testing.assert_allclose(h[:, :6], -dh, atol=2e-9)
    np.testing.assert_allclose(noise, dn @ obs.covariance @ dn.T, atol=2e-9)


def test_process_noise_zero_velocity_closed_form_and_nonzero_reference():
    from scipy.linalg import expm
    cfg, dt = BaselineConfig(), .1
    density = np.diag([cfg.angular_accel_density**2]*3+[cfg.linear_accel_density**2]*3)
    expected = np.block([[dt**3/3*density, dt**2/2*density], [dt**2/2*density, dt*density]])
    np.testing.assert_allclose(process_covariance(np.zeros(6), dt, cfg), expected, atol=1e-14)
    velocity = np.array([1., -2., .5, .5, -.2, .3])
    # Independent Van Loan discretization of A=[-ad(v), I; 0, 0].
    skew = lambda x: np.array([[0., -x[2], x[1]], [x[2], 0., -x[0]], [-x[1], x[0], 0.]])
    ad = np.block([[skew(velocity[:3]), np.zeros((3, 3))], [skew(velocity[3:]), skew(velocity[:3])]])
    a = np.zeros((12, 12)); a[:6, :6] = -ad; a[:6, 6:] = np.eye(6)
    q = np.zeros((12, 12)); q[6:, 6:] = density
    van_loan = expm(np.block([[a, q], [np.zeros((12, 12)), -a.T]])*dt)
    exact = van_loan[:12, 12:] @ van_loan[:12, :12].T
    np.testing.assert_allclose(process_covariance(velocity, dt, cfg), exact, atol=1e-12)


def test_custom_factor_analytic_jacobian():
    obs = measurement(rows(1)[0], FusionConfig(), True)
    x = obs.pose.compose(gtsam.Pose3.Expmap(np.array([.1, .2, -.1, .05, .1, -.05])))
    key = gtsam.symbol("x", 0)
    values = gtsam.Values(); values.insert(key, x)
    a, _ = pose_factor(key, obs, 0.).linearize(values).jacobian()
    jac = numerical_jacobian(lambda e: gtsam.Pose3.Logmap(obs.pose.between(x.retract(e))), 6)
    expected = np.linalg.cholesky(np.linalg.inv(obs.covariance)).T @ jac
    np.testing.assert_allclose(a, expected, atol=1e-6)


def test_single_camera_batch_and_initialization_not_double_counted():
    cfg = BaselineConfig()
    observations = [measurement(rows(1)[0], FusionConfig(), True)]
    pose, cov, _ = batch_pose(observations, cfg, cfg.huber_k)
    np.testing.assert_allclose(pose.matrix(), observations[0].pose.matrix(), atol=1e-12)
    np.testing.assert_allclose(cov, observations[0].covariance, atol=1e-12)
    filt = PoseVelocityFilter(cfg, True)
    filt.step(observations, 0.)
    np.testing.assert_allclose(filt.covariance[:6, :6], cov, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_order_invariance_covariance_and_no_mutation(method):
    data = rows()
    before = deepcopy(data)
    a, _ = run_baseline(data, method)
    b, _ = run_baseline(data[::-1], method)
    assert compare_pose_outputs(a, b)["pose_and_covariance_identical"]
    assert data == before
    for row in a:
        assert np.linalg.eigvalsh(row["covariance"]).min() > 0
        np.testing.assert_allclose(np.asarray(row["pose_world"])[:3, :3].T @ np.asarray(row["pose_world"])[:3, :3], np.eye(3), atol=1e-10)


def test_duplicate_replacement_and_invalid_group():
    data = rows(1)
    new = deepcopy(data[0]); new["arrival_time"] += .001; new["score"] = .2
    group = canonical_groups(data+[new], .04)[0]
    assert len(group) == 3 and group[0]["score"] == .2
    new["arrival_time"] = 0.
    with pytest.raises(ValueError, match="Ambiguous"):
        canonical_groups(data+[new], .04)
    data[0]["capture_time"] = .1
    with pytest.raises(ValueError, match="span"):
        canonical_groups(data, .04)


def test_missing_prediction_expiration_and_reinitialization():
    filt = PoseVelocityFilter(BaselineConfig(), True)
    assert filt.step([], 0.)[0] is None
    observations = [measurement(rows(1)[0], FusionConfig(), True)]
    filt.step(observations, .01)
    assert filt.step([], .1)[1]["prediction_only"]
    assert filt.step([], .6)[0] is None
    assert filt.step(observations, .7)[1]["status"] == "reinitialized"
    with pytest.raises(ValueError, match="increasing"):
        filt.step(observations, .7)


def test_gating_uses_prediction_and_does_not_renew_observation_age():
    obs = measurement(rows(1)[0], FusionConfig(), True)
    bad = Measurement("bad", obs.pose.compose(gtsam.Pose3.Expmap(np.array([0., 0., 0., 3., 0., 0.]))), obs.covariance)
    filt = PoseVelocityFilter(BaselineConfig(), True)
    filt.step([obs], 0.)
    _, detail = filt.step([bad], .03)
    assert detail["rejected"] == ["bad"] and detail["prediction_only"]
    assert filt.last_observed == 0.


def test_constant_velocity_and_object_isolation():
    data = rows(50)
    poses, _ = run_baseline(data, "eskf_quality")
    assert np.linalg.norm(np.asarray(poses[-1]["pose_world"])[:3, 3] - np.asarray(data[-2]["pose_world"])[:3, 3]) < .003
    other = [{**r, "object_name": "other"} for r in data]
    combined, _ = run_baseline(data+other, "eskf_quality")
    assert compare_pose_outputs(poses, [r for r in combined if r["object_name"] == "test"])["pose_and_covariance_identical"]


def test_quality_covariance_matches_production():
    from core.event_bus import EventBus
    from core.types import DetectionResult
    from experiments.h2o.ablation import build_config
    from managers.fusion_manager import FusionManager
    async def check():
        cfg = build_config(VARIANTS["full"])
        manager = FusionManager(EventBus(), asyncio.get_running_loop(), cfg)
        manager.configure_runtime(offline_replay=True)
        row = rows(1)[0]
        row["quality_metrics"] = {"q_area": .8, "depth_inlier_fraction": .4, "q_time": .7}
        await manager._on_pose_message(DetectionResult(
            t_host=row["t_host"], source=row["source"], pose_world=np.array(row["pose_world"]),
            object_name="test", score=row["score"], sync_group=0, quality_metrics=row["quality_metrics"]))
        obs = next(iter(manager._world_candidates.values()))
        obs = next(iter(obs.values()))
        expected = manager._get_system("test").quality.pose_covariance(obs)
        np.testing.assert_allclose(measurement(row, cfg.fusion, True).covariance, expected, atol=1e-14)
        np.testing.assert_allclose(measurement(row, cfg.fusion, False).covariance,
                                   np.diag([.035**2]*3+[.01**2]*3), atol=1e-14)
    asyncio.run(check())


def test_failure_preserves_only_completed_prefix_and_empty_input():
    data = rows(3)
    data[4]["pose_world"][0][0] = 99.
    with pytest.raises(BaselineRunError) as error:
        run_baseline(data, "eskf_fixed")
    assert [r["frame_id"] for r in error.value.predictions] == [0]
    assert error.value.failure["frame_id"] == 1
    assert run_baseline([], "se3_huber_fixed")[0] == []


def test_left_world_transform_equivariance():
    data = rows(8)
    transform = gtsam.Pose3.Expmap(np.array([.1, .3, -.2, .5, -.3, 1.]))
    changed = [{**r, "pose_world": transform.compose(gtsam.Pose3(np.asarray(r["pose_world"]))).matrix().tolist()} for r in data]
    for method in ("se3_huber_quality", "eskf_quality", "eskf_gated_quality"):
        first, _ = run_baseline(data, method)
        second, _ = run_baseline(changed, method)
        for a, b in zip(first, second):
            expected = transform.compose(gtsam.Pose3(np.asarray(a["pose_world"]))).matrix()
            np.testing.assert_allclose(b["pose_world"], expected, atol=1e-8)


def test_batch_huber_limits_outlier_and_handles_rotation_wrap():
    cfg = BaselineConfig()
    cov = np.diag([.035**2]*3+[.01**2]*3)
    measures = [Measurement(str(i), gtsam.Pose3(gtsam.Rot3(), np.array([x, 0., 1.])), cov)
                for i, x in enumerate([0., .001, 1.])]
    robust, _, _ = batch_pose(measures, cfg, cfg.huber_k)
    gaussian, _, _ = batch_pose(measures, cfg, 0.)
    assert robust.translation()[0] < .02
    assert gaussian.translation()[0] > .3
    rotations = [Measurement(str(i), gtsam.Pose3(gtsam.Rot3.Rz(np.deg2rad(a)), np.array([0., 0., 1.])), cov)
                 for i, a in enumerate([179., -179., 178.])]
    pose, _, _ = batch_pose(rotations, cfg, cfg.huber_k)
    assert np.rad2deg(np.linalg.norm(gtsam.Rot3.Logmap(pose.rotation()))) > 175.
