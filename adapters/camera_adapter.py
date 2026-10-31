# adapters/camera_adapter.py

import cv2

from common.structured_log import slog

from adapters import BaseBlockingAdapter


class CameraAdapter(BaseBlockingAdapter):
    def __init__(self, loop, out_queue, device_index: int = 0, source_name: str = "cam0"):
        super().__init__(loop, out_queue, source_name)
        self.cap = cv2.VideoCapture(device_index)
        slog(
            component=self.__class__.__name__,
            event="init",
            source=self.source_name,
            device_index=device_index,
        )

    def _read_from_sdk_blocking(self):
        ok, frame = self.cap.read()
        if not ok:
            # TODO: 更完善的错误处理
            return None
        return frame
