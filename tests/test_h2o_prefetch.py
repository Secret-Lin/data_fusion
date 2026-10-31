import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.h2o.prefetch import prefetch_rgbd


class Sequence:
    def camera_intrinsics(self, camera):
        return np.eye(3)

    def load_rgbd(self, camera, frame):
        # Completion order differs from delivery order with multiple readers.
        if camera == "cam0":
            time.sleep(0.005)
        if camera == "cam0" and frame == 1:
            raise FileNotFoundError("missing depth")
        return SimpleNamespace(frame_id=frame, camera_id=camera,
                               depth_m=np.full((2, 3), frame + 1, np.float32))


def collect(workers):
    rows = []
    with prefetch_rgbd(Sequence(), list(range(4)), ["cam0", "cam1"],
                       workers=workers, prefetch_frames=2) as batches:
        for frame, cameras in batches:
            for camera, future in cameras.items():
                try:
                    value = future.result()
                    rows.append((frame, camera, value.depth_m.tolist()))
                except FileNotFoundError as exc:
                    rows.append((frame, camera, str(exc)))
    return rows


def test_prefetch_preserves_camera_frame_order_and_error_isolation():
    assert collect(4) == collect(0)
    assert not any(t.name.startswith("h2o-rgbd") for t in threading.enumerate())


def test_prefetch_cleans_up_when_consumer_fails():
    with pytest.raises(RuntimeError, match="consumer stopped"):
        with prefetch_rgbd(Sequence(), list(range(40)), ["cam0", "cam1"],
                           workers=2, prefetch_frames=2) as batches:
            next(batches)
            raise RuntimeError("consumer stopped")
    assert not any(t.name.startswith("h2o-rgbd") for t in threading.enumerate())


def test_bad_intrinsics_do_not_abort_other_sensors():
    class BadCamera(Sequence):
        def camera_intrinsics(self, camera):
            if camera == "cam0":
                raise ValueError("invalid intrinsics")
            return super().camera_intrinsics(camera)

    with prefetch_rgbd(BadCamera(), [0, 1], ["cam0", "cam1"], workers=2) as batches:
        for frame, cameras in batches:
            with pytest.raises(ValueError, match="invalid intrinsics"):
                cameras["cam0"].result()
            assert cameras["cam1"].result().frame_id == frame
