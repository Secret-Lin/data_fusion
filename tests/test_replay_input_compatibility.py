import asyncio
import json

import h5py
import numpy as np

from config import BackendConfig
from core.event_bus import EventBus
from managers.replay_input_manager import ReplayInputManager


def _write_realsense(path, times, frame_indices, value) -> None:
    with h5py.File(path, "w") as stream:
        stream.create_dataset("color", data=np.full((len(times), 2, 3, 3), value, np.uint8))
        stream.create_dataset("depth", data=np.full((len(times), 2, 3), 1000, np.uint16))
        stream.create_dataset("t_host", data=np.asarray(times, np.float64))
        stream.create_dataset("frame_index", data=np.asarray(frame_indices, np.int64))


def _write_quest(path) -> None:
    with h5py.File(path, "w") as stream:
        stream.create_dataset("color", data=np.full((1, 2, 3, 3), [1, 2, 3], np.uint8))
        stream.create_dataset("depth", data=np.full((1, 2, 3), 800, np.uint16))
        stream.create_dataset("t_host", data=np.asarray([1.006], np.float64))
        stream.create_dataset("frame_index", data=np.asarray([91], np.int64))
        for name, value in (("fx", 100.0), ("fy", 101.0), ("cx", 1.0), ("cy", 0.5)):
            stream.create_dataset(name, data=np.asarray([value], np.float32))
        stream.create_dataset("camera_position", data=np.zeros((1, 3), np.float32))
        stream.create_dataset("camera_rotation_xyzw", data=np.asarray([[0, 0, 0, 1]], np.float32))
        stream.attrs["color_format"] = "bgr8_jpeg_decoded"


def test_recorder_schema_replays_all_frames_with_timestamp_groups(tmp_path) -> None:
    async def scenario() -> None:
        (tmp_path / "rgbd").mkdir()
        (tmp_path / "quest").mkdir()
        _write_realsense(tmp_path / "rgbd" / "cam_a.h5", [1.0, 1.033], [10, 12], 10)
        _write_realsense(tmp_path / "rgbd" / "cam_b.h5", [1.005, 1.070], [20, 25], 20)
        _write_quest(tmp_path / "quest" / "quest.h5")
        (tmp_path / "camera_metadata.json").write_text(
            json.dumps({name: {"K": np.eye(3).tolist(), "depth_scale_m": 0.001} for name in ("cam_a", "cam_b")}),
            encoding="utf-8",
        )
        (tmp_path / "tag_cube_pose.json").write_text(
            json.dumps(
                {
                    "T_world_tracking": {
                        "pos": [1.0, 2.0, 3.0],
                        "rot_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "quality": 100,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "tag_cube_extrinsics.json").write_text(
            json.dumps(
                {
                    "coordinate_system": "y_down",
                    "T_world_cam": {
                        "cam_a": np.eye(4).tolist(),
                        "cam_b": np.eye(4).tolist(),
                    },
                }
            ),
            encoding="utf-8",
        )

        bus = EventBus()
        queues = {
            topic: bus.subscribe(topic)
            for topic in ("realsense/cam_a/stream", "realsense/cam_b/stream", "quest/quest/stream")
        }
        manager = ReplayInputManager(
            bus,
            BackendConfig(),
            session_dir=tmp_path,
            fps_override=1e9,
            sync_tolerance_s=0.015,
        )
        assert np.isclose(
            manager.recorded_fps_by_source["realsense/cam_a"],
            1.0 / 0.033,
        )
        assert np.isclose(
            manager.recorded_fps_by_source["realsense/cam_b"],
            1.0 / 0.065,
        )
        await manager.run()

        cam_a = [queues["realsense/cam_a/stream"].get_nowait() for _ in range(2)]
        cam_b = [queues["realsense/cam_b/stream"].get_nowait() for _ in range(2)]
        quest = queues["quest/quest/stream"].get_nowait()
        assert [packet.frame_index for packet in cam_a] == [10, 12]
        assert [packet.frame_index for packet in cam_b] == [20, 25]
        assert cam_a[0].sync_group == cam_b[0].sync_group == quest.sync_group
        assert len({cam_a[1].sync_group, cam_b[1].sync_group}) == 2
        assert quest.frame_index == 91
        assert quest.payload["frames"]["color"][0, 0].tolist() == [1, 2, 3]
        assert np.allclose(quest.payload["T_world_tracking"][:3, 3], [1, 2, 3])
        assert np.allclose(quest.payload["T_tracking_cam"], np.eye(4))
        assert np.allclose(quest.payload["T_world_cam"][:3, 3], [1, 2, 3])
        assert quest.payload["coordinate_system"] == "y_down"
        assert np.allclose(cam_a[0].payload["T_world_cam"], np.eye(4))
        assert cam_a[0].payload["coordinate_system"] == "y_down"

    asyncio.run(scenario())
