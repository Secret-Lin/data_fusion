import asyncio
import json

import numpy as np

from core.event_bus import EventBus
from managers.tracking_manager import TrackingManager


def test_record_init_extrinsic_loads_and_mobile_tag_correction_is_bounded(tmp_path) -> None:
    path = tmp_path / "tag_cube_pose.json"
    path.write_text(
        json.dumps({"T_world_tracking": {"pos": [1, 2, 3], "rot_xyzw": [0, 0, 0, 1]}, "quality": 100}),
        encoding="utf-8",
    )
    loop = asyncio.new_event_loop()
    manager = TrackingManager(
        EventBus(),
        loop,
        quest_world_extrinsic_path=str(path),
        quest_tag_correction_alpha=0.5,
    )
    assert manager._sensor_mode("quest/quest") == "mobile"
    assert np.allclose(manager._T_world_quest[:3, 3], [1, 2, 3])

    observed = manager._T_world_quest.copy()
    observed[0, 3] += 0.1
    manager._correct_quest_world_extrinsic(observed, "quest/quest")
    assert np.isclose(manager._T_world_quest[0, 3], 1.05)
    manager._dope_executor.shutdown()
    manager._apriltag_executor.shutdown()
    manager._vo_executor.shutdown()
    loop.close()


def test_fixed_sensor_never_falls_back_to_identity() -> None:
    loop = asyncio.new_event_loop()
    manager = TrackingManager(EventBus(), loop, sensor_modes={"cam": "fixed"})
    assert manager._sensor_mode("cam") == "fixed"
    assert manager._lookup_T_world_cam("cam") is None
    manager._correct_quest_world_extrinsic(np.eye(4), "quest/quest")
    assert manager._T_world_quest is None
    manager._dope_executor.shutdown()
    manager._apriltag_executor.shutdown()
    manager._vo_executor.shutdown()
    loop.close()
