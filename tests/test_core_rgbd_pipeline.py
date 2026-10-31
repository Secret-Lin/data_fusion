import asyncio

import numpy as np

from core.event_bus import EventBus
from core.frame_sync import FrameGroupAssigner
from core.types import CameraModel, RGBDFrame, TimeStampedPacket, rgbd_frame_from_packet


def test_legacy_rgbd_packet_has_explicit_metadata() -> None:
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.ones((3, 4), dtype=np.uint16)
    packet = TimeStampedPacket(
        10.0,
        "realsense/cam0",
        {"camera": {"K": np.eye(3), "depth_scale_m": 0.002}, "frames": {"color": color, "depth": depth}},
        t_capture=9.5,
        t_arrival=10.0,
        frame_index=12,
        sync_group=7,
    )

    frame = rgbd_frame_from_packet(packet)

    assert frame.frame_index == 12
    assert frame.sync_group == 7
    assert frame.camera.depth_scale == 0.002
    assert frame.color_bgr is color


def test_typed_rgbd_payload_is_accepted_without_copy() -> None:
    frame = RGBDFrame(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint16),
        CameraModel(np.eye(3)),
        "realsense/cam0",
        1.0,
        1.1,
    )
    assert rgbd_frame_from_packet(TimeStampedPacket(1.1, frame.source_id, frame)) is frame


def test_sync_groups_nearby_distinct_sources() -> None:
    assigner = FrameGroupAssigner(0.02)
    assert assigner.assign("cam0", 1.0) == 0
    assert assigner.assign("cam1", 1.01) == 0
    assert assigner.assign("cam0", 1.02) == 1
    assert assigner.assign("cam1", 1.08) == 2


def test_event_bus_latest_is_observable_and_reliable_is_lossless() -> None:
    async def scenario() -> None:
        bus = EventBus()
        latest = bus.subscribe("frames", policy="latest")
        reliable = bus.subscribe("poses", maxsize=2)
        await bus.publish("frames", 1)
        await bus.publish("frames", 2)
        await bus.publish("poses", 1)
        await bus.publish("poses", 2)
        assert await latest.get() == 2
        assert latest.dropped == 1
        assert [await reliable.get(), await reliable.get()] == [1, 2]
        assert bus.metrics()["frames"][0]["dropped"] == 1

    asyncio.run(scenario())
