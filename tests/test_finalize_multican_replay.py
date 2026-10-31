from __future__ import annotations

import numpy as np

from tools.finalize_multican_replay import _frame_index_at_time


def test_frame_index_at_time_uses_capture_timestamps() -> None:
    timestamps = np.asarray([10.00, 10.03, 10.10, 10.11], dtype=np.float64)

    assert _frame_index_at_time(timestamps, 9.99) == -1
    assert _frame_index_at_time(timestamps, 10.00) == 0
    assert _frame_index_at_time(timestamps, 10.09) == 1
    assert _frame_index_at_time(timestamps, 10.10) == 2
    assert _frame_index_at_time(timestamps, 11.00) == 3
