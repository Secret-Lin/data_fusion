import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from adapters.foundationpose_runner import (
    FoundationPoseEstimate,
    HopeModelRepository,
)
from adapters.cnos_runner import CnosMask
from config import FoundationPoseConfig
from core.event_bus import EventBus
from core.types import DetectionResult, TimeStampedPacket
from managers.foundationpose_manager import FoundationPoseManager


REPO_ROOT = Path(__file__).resolve().parents[1]
HOPE_ROOT = REPO_ROOT / "assets" / "models" / "hope_original" / "cans"


def test_hope_repository_discovers_and_scales_cans() -> None:
    if not HOPE_ROOT.is_dir():
        pytest.skip("Optional HOPE can mesh assets are not installed")
    repository = HopeModelRepository(HOPE_ROOT, scale_to_m=0.01)
    assert repository.names() == [
        "AlphabetSoup",
        "Cherries",
        "Corn",
        "GreenBeans",
        "Mushrooms",
        "Peaches",
        "PeasAndCarrots",
        "Pineapple",
        "TomatoSauce",
        "Tuna",
    ]
    model, mesh = repository.load_mesh("tomatosauce")
    assert model.name == "TomatoSauce"
    assert np.all(mesh.extents > 0.03)
    assert np.all(mesh.extents < 0.10)


class FakeRunner:
    def __init__(self) -> None:
        self.calls = []
        self.reset_calls = []

    def available_objects(self):
        return ["Corn", "TomatoSauce", "Tuna"]

    def model_bounds(self, object_name):
        return np.array([[-0.03, -0.03, -0.04], [0.03, 0.03, 0.04]])

    def reset(self, source=None, object_name=None):
        self.reset_calls.append((source, object_name))
        return None

    def process(self, **kwargs):
        self.calls.append(kwargs)
        return FoundationPoseEstimate(
            object_name=kwargs["object_name"],
            pose_cam=np.eye(4),
            score=1.0,
            mode="register" if kwargs["mask"] is not None else "track",
        )


class FakeRecoveryRunner:
    def __init__(self, masks: list[list[CnosMask]]) -> None:
        self.responses = list(masks)
        self.calls: list[tuple[tuple[int, ...], list[str]]] = []

    def prepare(self, object_names):
        return None

    def detect(self, color_bgr, object_names):
        self.calls.append((tuple(color_bgr.shape), list(object_names)))
        return self.responses.pop(0) if self.responses else []


def test_partial_initial_masks_are_deferred_to_cnos_recovery(
    tmp_path: Path,
) -> None:
    mask = np.ones((8, 12), dtype=np.uint8) * 255
    mask_path = tmp_path / "realsense_cam__TomatoSauce.png"
    assert cv2.imwrite(str(mask_path), mask)
    manager = FoundationPoseManager(
        EventBus(),
        FoundationPoseConfig(
            objects=["Corn", "TomatoSauce", "Tuna"],
            initial_mask_path=str(tmp_path),
            recovery_enable=True,
        ),
        realsense_topics=["realsense/cam/stream"],
        runner=FakeRunner(),
        recovery_runner=FakeRecoveryRunner([]),
    )

    assert ("realsense/cam", "TomatoSauce") in manager._pending_masks
    assert ("realsense/cam", "Corn") not in manager._pending_masks
    assert ("realsense/cam", "Tuna") not in manager._pending_masks
    manager._executor.shutdown(wait=True)


def test_manager_converts_rgbd_and_publishes_detection() -> None:
    async def scenario() -> None:
        bus = EventBus()
        output = bus.subscribe("tracking/detections")
        status = bus.subscribe("tracking/foundationpose/status")
        runner = FakeRunner()
        cfg = FoundationPoseConfig(objects=["TomatoSauce"])
        manager = FoundationPoseManager(bus, cfg, runner=runner)
        manager._T_world_camera["realsense/123456/stream"] = np.eye(4)
        mask = np.ones((4, 6), dtype=np.uint8) * 255
        manager._handle_command(
            {"source": "realsense/123456/stream", "object_name": "TomatoSauce", "mask": mask}
        )
        packet = TimeStampedPacket(
            t_host=12.5,
            source="realsense/123456/stream",
            payload={
                "camera": {
                    "K": [[100.0, 0.0, 3.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
                    "depth_scale_m": 0.001,
                },
                "frames": {
                    "color": np.zeros((4, 6, 3), dtype=np.uint8),
                    "depth": np.ones((4, 6), dtype=np.uint16) * 750,
                },
            },
        )
        await manager._handle_frame(packet)
        detection = output.get_nowait()
        state = status.get_nowait()
        assert isinstance(detection, DetectionResult)
        assert detection.source.endswith("/foundationpose/TomatoSauce")
        assert np.allclose(detection.pose_world, np.eye(4))
        assert state["mode"] == "register"
        assert np.allclose(runner.calls[0]["depth_m"], 0.75)
        assert runner.calls[0]["mask"] is mask
        manager._handle_command(
            {
                "source": packet.source,
                "class_name": "can",
                "mask": mask,
            },
            automatic=True,
        )
        assert not manager._pending_masks
        manager._executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_manager_can_attach_rendered_depth_quality_without_changing_pose() -> None:
    async def scenario() -> None:
        bus = EventBus()
        output = bus.subscribe("tracking/detections")
        status = bus.subscribe("tracking/foundationpose/status")
        runner = FakeRunner()
        runner.measure_depth_fit = lambda **kwargs: {
            "status": "ok",
            "observed_overlap_fraction": 0.9,
            "depth_inlier_fraction": 0.25,
            "depth_abs_error_median_m": 0.1,
        }
        cfg = FoundationPoseConfig(objects=["TomatoSauce"], depth_fit_quality_enable=True)
        manager = FoundationPoseManager(bus, cfg, runner=runner)
        manager._T_world_camera["realsense/123456/stream"] = np.eye(4)
        mask = np.ones((4, 6), dtype=np.uint8) * 255
        manager._handle_command({
            "source": "realsense/123456/stream",
            "object_name": "TomatoSauce",
            "mask": mask,
        })
        packet = TimeStampedPacket(
            t_host=12.5,
            source="realsense/123456/stream",
            payload={
                "camera": {
                    "K": [[100.0, 0.0, 3.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
                    "depth_scale_m": 0.001,
                },
                "frames": {
                    "color": np.zeros((4, 6, 3), dtype=np.uint8),
                    "depth": np.ones((4, 6), dtype=np.uint16) * 750,
                },
            },
        )
        await manager._handle_frame(packet)
        detection = output.get_nowait()
        state = status.get_nowait()
        assert detection.quality_metrics["depth_inlier_fraction"] == 0.25
        assert state["quality_metrics"]["observed_overlap_fraction"] == 0.9
        np.testing.assert_allclose(detection.pose_world, np.eye(4))
        manager._executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_automatic_mask_class_mapping() -> None:
    bus = EventBus()
    runner = FakeRunner()
    cfg = FoundationPoseConfig(
        objects=["TomatoSauce"], class_to_object={"can": "TomatoSauce"}
    )
    manager = FoundationPoseManager(bus, cfg, runner=runner)
    mask = np.ones((3, 3), dtype=np.uint8)
    manager._handle_command(
        {"source": "realsense/cam/stream", "class_name": "can", "mask": mask},
        automatic=True,
    )
    assert manager._pending_masks[("realsense/cam/stream", "TomatoSauce")] is mask
    manager._executor.shutdown(wait=True)


def test_manager_uses_mobile_rgbd_pose_and_world_tracking(tmp_path: Path) -> None:
    async def scenario() -> None:
        extrinsic_path = tmp_path / "tag_cube_pose.json"
        extrinsic_path.write_text(
            '{"T_world_tracking":{"pos":[1,2,3],"rot_xyzw":[0,0,0,1]}}',
            encoding="utf-8",
        )
        bus = EventBus()
        output = bus.subscribe("tracking/detections")
        runner = FakeRunner()
        manager = FoundationPoseManager(
            bus,
            FoundationPoseConfig(objects=["TomatoSauce"]),
            runner=runner,
            mobile_world_extrinsic_path=extrinsic_path,
        )
        mask = np.ones((4, 6), dtype=np.uint8) * 255
        manager._handle_command(
            {"source": "quest/quest", "object_name": "TomatoSauce", "mask": mask}
        )
        tracking_camera = np.eye(4)
        tracking_camera[0, 3] = 0.5
        packet = TimeStampedPacket(
            t_host=1.0,
            source="quest/quest",
            payload={
                "camera": {"K": np.eye(3), "depth_scale_m": 0.001},
                "frames": {
                    "color": np.zeros((4, 6, 3), dtype=np.uint8),
                    "depth": np.ones((4, 6), dtype=np.uint16) * 1000,
                },
                "T_tracking_cam": tracking_camera,
            },
        )
        await manager._handle_frame(packet)
        detection = output.get_nowait()
        # Manager inputs have already crossed the Y-down ingress boundary.
        assert np.allclose(detection.pose_world[:3, 3], [1.5, 2.0, 3.0])
        assert runner.calls[0]["source"] == "quest/quest"
        manager._executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_cnos_mask_re_registers_after_two_consistent_disagreements() -> None:
    async def scenario() -> None:
        bus = EventBus()
        output = bus.subscribe("tracking/detections")
        runner = FakeRunner()
        recovery_mask = np.zeros((4, 6), dtype=np.uint8)
        recovery_mask[1:3, 2:5] = 255
        recovery = FakeRecoveryRunner(
            [
                [CnosMask("TomatoSauce", recovery_mask, 0.8, (2, 1, 4, 2))],
                [CnosMask("TomatoSauce", recovery_mask, 0.8, (2, 1, 4, 2))],
            ]
        )
        cfg = FoundationPoseConfig(
            objects=["TomatoSauce"],
            recovery_enable=True,
            recovery_interval_s=0.5,
        )
        manager = FoundationPoseManager(
            bus, cfg, runner=runner, recovery_runner=recovery
        )
        manager._T_world_camera["realsense/cam"] = np.eye(4)
        initial_mask = np.ones((4, 6), dtype=np.uint8) * 255
        manager._handle_command(
            {
                "source": "realsense/cam",
                "object_name": "TomatoSauce",
                "mask": initial_mask,
            }
        )

        def packet(timestamp: float) -> TimeStampedPacket:
            return TimeStampedPacket(
                t_host=timestamp,
                source="realsense/cam",
                payload={
                    "camera": {"K": np.eye(3), "depth_scale_m": 0.001},
                    "frames": {
                        "color": np.zeros((4, 6, 3), dtype=np.uint8),
                        "depth": np.ones((4, 6), dtype=np.uint16) * 1000,
                    },
                },
            )

        await manager._handle_frame(packet(1.0))
        await manager._handle_frame(packet(1.6))
        await manager._handle_frame(packet(2.2))

        detections = [output.get_nowait(), output.get_nowait(), output.get_nowait()]
        assert [call["mask"] is not None for call in runner.calls] == [
            True,
            False,
            True,
        ]
        assert runner.calls[2]["mask"] is recovery_mask
        assert len(recovery.calls) == 2
        assert [item.object_name for item in detections] == [
            "TomatoSauce",
            "TomatoSauce",
            "TomatoSauce",
        ]
        manager._executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_cnos_mask_agreeing_with_cad_projection_keeps_tracking() -> None:
    async def scenario() -> None:
        bus = EventBus()
        runner = FakeRunner()
        recovery_mask = np.zeros((100, 100), dtype=np.uint8)
        recovery_mask[47:54, 47:54] = 255
        recovery = FakeRecoveryRunner(
            [[CnosMask("TomatoSauce", recovery_mask, 0.8, (47, 47, 53, 53))]]
        )
        cfg = FoundationPoseConfig(
            objects=["TomatoSauce"],
            recovery_enable=True,
            recovery_interval_s=0.5,
        )
        manager = FoundationPoseManager(
            bus, cfg, runner=runner, recovery_runner=recovery
        )
        manager._T_world_camera["realsense/cam"] = np.eye(4)
        manager._handle_command(
            {
                "source": "realsense/cam",
                "object_name": "TomatoSauce",
                "mask": recovery_mask,
            }
        )

        def packet(timestamp: float) -> TimeStampedPacket:
            return TimeStampedPacket(
                t_host=timestamp,
                source="realsense/cam",
                payload={
                    "camera": {
                        "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
                        "depth_scale_m": 0.001,
                    },
                    "frames": {
                        "color": np.zeros((100, 100, 3), dtype=np.uint8),
                        "depth": np.ones((100, 100), dtype=np.uint16) * 1000,
                    },
                },
            )

        await manager._handle_frame(packet(1.0))
        pose_world = np.eye(4)
        pose_world[2, 3] = 1.0
        manager._last_pose_world[("realsense/cam", "TomatoSauce")] = pose_world
        await manager._handle_frame(packet(1.6))

        assert [call["mask"] is not None for call in runner.calls] == [True, False]
        assert len(recovery.calls) == 1
        manager._executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_multi_object_visualization_contains_every_detection_once() -> None:
    async def scenario() -> None:
        bus = EventBus()
        detections = bus.subscribe("tracking/detections")
        visualization = bus.subscribe("visualization/frames")
        runner = FakeRunner()
        objects = ["Tuna", "Corn", "TomatoSauce"]
        manager = FoundationPoseManager(
            bus,
            FoundationPoseConfig(objects=objects),
            runner=runner,
        )
        manager._T_world_camera["realsense/cam"] = np.eye(4)
        manager.set_visualization_stream(enabled=True)
        for index, object_name in enumerate(objects):
            mask = np.zeros((80, 120), dtype=np.uint8)
            mask[10:30, 10 + index * 25:30 + index * 25] = 255
            manager._handle_command(
                {
                    "source": "realsense/cam",
                    "object_name": object_name,
                    "mask": mask,
                }
            )
        packet = TimeStampedPacket(
            t_host=1.0,
            source="realsense/cam",
            payload={
                "camera": {
                    "K": [[100.0, 0.0, 60.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
                    "depth_scale_m": 0.001,
                },
                "frames": {
                    "color": np.zeros((80, 120, 3), dtype=np.uint8),
                    "depth": np.ones((80, 120), dtype=np.uint16) * 1000,
                },
            },
        )

        await manager._handle_frame(packet)

        emitted = [detections.get_nowait() for _ in objects]
        frame = visualization.get_nowait()
        assert [item.object_name for item in emitted] == objects
        assert frame["stream"] == "foundationpose/all"
        assert frame["objects"] == objects
        assert visualization.empty()
        manager._executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_confirmed_cnos_misses_stop_tracking_until_confirmed_recovery() -> None:
    async def scenario() -> None:
        bus = EventBus()
        output = bus.subscribe("tracking/detections")
        runner = FakeRunner()
        recovery_mask = np.zeros((100, 100), dtype=np.uint8)
        recovery_mask[45:56, 45:56] = 255
        recovery = FakeRecoveryRunner(
            [
                [],
                [],
                [CnosMask("TomatoSauce", recovery_mask, 0.8, (45, 45, 55, 55))],
                [CnosMask("TomatoSauce", recovery_mask, 0.8, (45, 45, 55, 55))],
            ]
        )
        cfg = FoundationPoseConfig(
            objects=["TomatoSauce"],
            recovery_enable=True,
            recovery_interval_s=0.5,
            recovery_misses_before_lost=2,
            recovery_miss_confidence_decay=0.35,
            recovery_confirmations=2,
        )
        manager = FoundationPoseManager(
            bus, cfg, runner=runner, recovery_runner=recovery
        )
        manager._T_world_camera["realsense/cam"] = np.eye(4)
        manager._handle_command(
            {
                "source": "realsense/cam",
                "object_name": "TomatoSauce",
                "mask": recovery_mask,
            }
        )

        def packet(timestamp: float) -> TimeStampedPacket:
            return TimeStampedPacket(
                t_host=timestamp,
                source="realsense/cam",
                payload={
                    "camera": {
                        "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
                        "depth_scale_m": 0.001,
                    },
                    "frames": {
                        "color": np.zeros((100, 100, 3), dtype=np.uint8),
                        "depth": np.ones((100, 100), dtype=np.uint16) * 1000,
                    },
                },
            )

        for timestamp in (1.0, 1.6, 2.2, 2.8, 3.4):
            await manager._handle_frame(packet(timestamp))

        emitted = []
        while not output.empty():
            emitted.append(output.get_nowait())
        assert len(emitted) == 3
        assert emitted[1].score == 0.35
        assert [call["mask"] is not None for call in runner.calls] == [
            True,
            False,
            True,
        ]
        assert runner.reset_calls == [("realsense/cam", "TomatoSauce")]
        assert len(recovery.calls) == 4
        manager._executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_recovery_cannot_claim_region_reserved_by_other_object() -> None:
    async def scenario() -> None:
        bus = EventBus()
        status = bus.subscribe("tracking/foundationpose/status")
        runner = FakeRunner()
        corn_mask = np.zeros((100, 100), dtype=np.uint8)
        corn_mask[45:56, 45:56] = 255
        tomato_mask = np.zeros((100, 100), dtype=np.uint8)
        tomato_mask[75:86, 75:86] = 255
        false_tomato_mask = corn_mask.copy()
        recovery = FakeRecoveryRunner(
            [[CnosMask("TomatoSauce", false_tomato_mask, 0.8, (45, 45, 55, 55))]]
        )
        cfg = FoundationPoseConfig(
            objects=["Corn", "TomatoSauce"],
            recovery_enable=True,
            recovery_interval_s=0.5,
        )
        manager = FoundationPoseManager(
            bus, cfg, runner=runner, recovery_runner=recovery
        )
        manager._T_world_camera["realsense/cam"] = np.eye(4)
        for object_name, mask in (("Corn", corn_mask), ("TomatoSauce", tomato_mask)):
            manager._handle_command(
                {
                    "source": "realsense/cam",
                    "object_name": object_name,
                    "mask": mask,
                }
            )

        def packet(timestamp: float) -> TimeStampedPacket:
            return TimeStampedPacket(
                t_host=timestamp,
                source="realsense/cam",
                payload={
                    "camera": {
                        "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
                        "depth_scale_m": 0.001,
                    },
                    "frames": {
                        "color": np.zeros((100, 100, 3), dtype=np.uint8),
                        "depth": np.ones((100, 100), dtype=np.uint16) * 1000,
                    },
                },
            )

        await manager._handle_frame(packet(1.0))
        corn_pose = np.eye(4)
        corn_pose[2, 3] = 1.0
        tomato_pose = np.eye(4)
        tomato_pose[0, 3] = 0.30
        tomato_pose[2, 3] = 1.0
        manager._last_pose_world[("realsense/cam", "Corn")] = corn_pose
        manager._last_pose_world[("realsense/cam", "TomatoSauce")] = tomato_pose
        await manager._handle_frame(packet(1.6))

        second_frame_calls = runner.calls[2:]
        assert len(second_frame_calls) == 2
        assert all(call["mask"] is None for call in second_frame_calls)
        status_items = []
        while not status.empty():
            status_items.append(status.get_nowait())
        occupied = [
            item
            for item in status_items
            if item.get("reason") == "occupied_by_other_object"
        ]
        assert len(occupied) == 1
        assert occupied[0]["object_name"] == "TomatoSauce"
        assert occupied[0]["occupied_by"] == "Corn"
        manager._executor.shutdown(wait=True)

    asyncio.run(scenario())
