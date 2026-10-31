import asyncio

import cv2
import numpy as np

from adapters.quest_adapter import QuestRGBDAdapter, quest_protocol
from config import QuestConfig


def _frame_payload() -> bytes:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:, :, 1] = 127
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpg = encoded.tobytes()
    values = [
        quest_protocol.QUEST_MAGIC_QCAM, 1, quest_protocol.QUEST_HEADER_SIZE,
        42, 1234, 3, 2, 3, 2,
        100.0, 101.0, 1.0, 0.5, 3, 2,
        1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0, 90, len(jpg), 0,
    ]
    return quest_protocol.QUEST_HEADER_STRUCT.pack(*values) + jpg


def test_sensor_command_gates_stream_and_frame_matches_replay_schema() -> None:
    loop = asyncio.new_event_loop()
    adapter = QuestRGBDAdapter(loop, asyncio.Queue(), QuestConfig())
    assert not adapter.streaming
    adapter._handle_command(None, {"cmd": "sensor_toggle", "active": True})
    assert not adapter.streaming
    adapter._world_tracking = np.eye(4)
    adapter._state = "ready"
    adapter._handle_command(None, {"cmd": "sensor_toggle", "active": True})
    assert adapter.streaming

    packet = adapter._decode_frame(_frame_payload(), 10.5)
    assert packet.source == "quest/quest"
    assert packet.frame_index == 42
    assert packet.t_capture == 10.5
    assert packet.payload["frames"]["color"].shape == (2, 3, 3)
    assert packet.payload["frames"]["depth"].shape == (2, 3)
    assert packet.payload["camera"]["depth_scale_m"] == 0.001
    assert np.allclose(packet.payload["T_tracking_cam"][:3, 3], [1.0, -2.0, 3.0])
    assert np.allclose(packet.payload["T_world_cam"][:3, 3], [1.0, -2.0, 3.0])
    assert packet.payload["coordinate_system"] == "y_down"

    adapter._handle_command(None, {"cmd": "sensor_toggle", "active": False})
    assert not adapter.streaming
    loop.close()
