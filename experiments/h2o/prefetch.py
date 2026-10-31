"""Bounded RGB-D read-ahead; delivery preserves frame and camera order."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Iterator

from experiments.h2o.dataset import H2ORGBDFrame, H2OSequence


class _DeferredFrame:
    def __init__(self, load: Callable[[], H2ORGBDFrame]):
        self._load = load

    def result(self) -> H2ORGBDFrame:
        return self._load()


@contextmanager
def prefetch_rgbd(
    sequence: H2OSequence, frame_ids: list[int], cameras: list[str], *,
    workers: int = 4, prefetch_frames: int = 4,
) -> Iterator[Iterator[tuple[int, dict[str, Future | _DeferredFrame]]]]:
    """Decode on CPU workers only; all tracking remains on the calling thread.

    The queue holds at most ``prefetch_frames`` future frame groups plus the
    group currently being consumed. Exceptions surface at that camera's
    ``result()`` call, so a missing RGB-D pair cannot discard other cameras.
    workers=0 uses the original synchronous loading order for comparisons.
    """
    if workers < 0 or prefetch_frames < 1:
        raise ValueError("workers must be nonnegative and prefetch_frames positive")
    if workers == 0:
        yield ((frame_id, {
            camera: _DeferredFrame(lambda c=camera, f=frame_id: sequence.load_rgbd(c, f))
            for camera in cameras
        }) for frame_id in frame_ids)
        return

    # Populate the immutable intrinsics cache before launching reader threads.
    intrinsic_errors: dict[str, Exception] = {}
    for camera in cameras:
        try:
            sequence.camera_intrinsics(camera)
        except Exception as exc:
            # Relay the original exception per sensor/frame just like a decode
            # failure. One invalid camera must not abort the complete take.
            intrinsic_errors[camera] = exc
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="h2o-rgbd")
    pending = deque()
    remaining = iter(frame_ids)

    def submit(camera: str, frame_id: int) -> Future:
        if camera in intrinsic_errors:
            future = Future()
            future.set_exception(intrinsic_errors[camera])
            return future
        return executor.submit(sequence.load_rgbd, camera, frame_id)

    def enqueue() -> bool:
        frame_id = next(remaining, None)
        if frame_id is None:
            return False
        pending.append((frame_id, {
            camera: submit(camera, frame_id)
            for camera in cameras
        }))
        return True

    def consume():
        for _ in range(prefetch_frames):
            if not enqueue():
                break
        while pending:
            item = pending.popleft()
            enqueue()
            yield item

    try:
        yield consume()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
