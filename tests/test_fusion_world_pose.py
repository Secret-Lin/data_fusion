import asyncio

import numpy as np

from config import BackendConfig
from core.event_bus import EventBus
from core.types import DetectionResult
from managers.fusion_manager import FusionManager, Observation, QualityScorer


def _pose(x: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[0, 3] = x
    return result


def test_world_pose_is_not_transformed_twice_and_observations_share_step() -> None:
    async def scenario() -> None:
        cfg = BackendConfig()
        cfg.fusion.sync_window_s = 0.04
        manager = FusionManager(EventBus(), asyncio.get_running_loop(), cfg)

        first = await manager._on_pose_message(
            DetectionResult(
                t_host=10.0,
                source="realsense/134722070227/stream/dope/TomatoSauce",
                pose_world=_pose(1.0),
                score=1.0,
            )
        )
        assert first == ("TomatoSauce", 0)
        system = manager._gtsam_instances["TomatoSauce"]
        system.update()
        assert np.isclose(system.estimate_tool_pose(0)[0, 3], 1.0, atol=1e-6)

        second = await manager._on_pose_message(
            DetectionResult(
                t_host=10.02,
                source="realsense/242322077348/stream/dope/TomatoSauce",
                pose_world=_pose(1.02),
                score=1.0,
            )
        )
        assert second == ("TomatoSauce", 0)
        system.update()
        fused_x = system.estimate_tool_pose(0)[0, 3]
        assert 1.0 < fused_x < 1.02

    asyncio.run(scenario())


def test_foundationpose_is_accepted_and_same_state_outlier_is_rejected() -> None:
    async def scenario() -> None:
        cfg = BackendConfig()
        cfg.fusion.enable_revisable_consensus = False
        manager = FusionManager(EventBus(), asyncio.get_running_loop(), cfg)
        accepted = await manager._on_pose_message(
            DetectionResult(
                t_host=20.0,
                source="realsense/134722070227/stream/foundationpose/TomatoSauce",
                pose_world=_pose(0.5),
                score=1.0,
            )
        )
        assert accepted == ("TomatoSauce", 0)
        manager._gtsam_instances["TomatoSauce"].update()

        rejected = await manager._on_pose_message(
            DetectionResult(
                t_host=20.01,
                source="realsense/242322077348/stream/foundationpose/TomatoSauce",
                pose_world=_pose(1.0),
                score=1.0,
            )
        )
        assert rejected is None

    asyncio.run(scenario())


def test_seg3d_detection_is_not_filtered_out() -> None:
    async def scenario() -> None:
        cfg = BackendConfig()
        manager = FusionManager(EventBus(), asyncio.get_running_loop(), cfg)
        result = await manager._on_pose_message(
            DetectionResult(
                t_host=30.0,
                source="134722070227/seg3d",
                pose_world=_pose(0.25),
                score=0.8,
            )
        )
        assert result == ("__default__", 0)
        system = manager._gtsam_instances["__default__"]
        system.update()
        assert np.isclose(system.estimate_tool_pose(0)[0, 3], 0.25, atol=1e-6)

    asyncio.run(scenario())


def test_higher_quality_foundationpose_is_not_blocked_by_dope_gate() -> None:
    async def scenario() -> None:
        cfg = BackendConfig()
        manager = FusionManager(EventBus(), asyncio.get_running_loop(), cfg)
        first = await manager._on_pose_message(
            DetectionResult(
                t_host=40.0,
                source="realsense/cam1/dope/TomatoSauce",
                pose_world=_pose(0.0),
                score=0.5,
            )
        )
        assert first == ("TomatoSauce", 0)
        manager._gtsam_instances["TomatoSauce"].update()
        better = await manager._on_pose_message(
            DetectionResult(
                t_host=40.01,
                source="realsense/cam2/foundationpose/TomatoSauce",
                pose_world=_pose(0.3),
                score=1.0,
            )
        )
        assert better == ("TomatoSauce", 0)

    asyncio.run(scenario())


def test_robust_noise_keeps_gaussian_covariance_available_for_nis() -> None:
    cfg = BackendConfig()
    observation = Observation(0, "sensor", "world", None)
    sigmas = QualityScorer(cfg.fusion).pose_sigmas(observation)
    assert sigmas.shape == (6,)
    assert np.all(sigmas > 0.0)


def test_explicit_sync_group_is_authoritative_for_out_of_order_results() -> None:
    async def scenario() -> None:
        cfg = BackendConfig()
        bus = EventBus()
        output = bus.subscribe(cfg.fusion.topic_fused)
        manager = FusionManager(bus, asyncio.get_running_loop(), cfg)

        newer = await manager._on_pose_message(
            DetectionResult(
                t_host=50.10,
                source="realsense/cam1/foundationpose/TomatoSauce",
                pose_world=_pose(0.10),
                score=1.0,
                sync_group=8,
            )
        )
        older = await manager._on_pose_message(
            DetectionResult(
                t_host=50.00,
                source="realsense/cam2/foundationpose/TomatoSauce",
                pose_world=_pose(0.00),
                score=1.0,
                sync_group=7,
            )
        )
        assert newer == ("TomatoSauce", 8)
        assert older == ("TomatoSauce", 7)

        await manager._update_and_publish("TomatoSauce", 7)
        fused = output.get_nowait()
        assert fused["step"] == 7
        assert fused["sync_group"] == 7
        assert fused["t_capture"] == 50.00
        assert fused["t_publish"] >= fused["t_capture"]

    asyncio.run(scenario())
