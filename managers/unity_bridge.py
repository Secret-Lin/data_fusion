# managers/unity_bridge.py

import asyncio
import base64
import json
import cv2

from common.structured_log import slog

from core.event_bus import EventBus

class UnityBridgeManager:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.marker_vis_queue = bus.subscribe("hl2_marker_vis")
        # TODO: setup TCP/WebSocket to Unity here

    async def run(self):
        slog(
            component=self.__class__.__name__,
            event="run",
            topic_in="hl2_marker_vis",
        )
        while True:
            msg = await self.marker_vis_queue.get()
            img = msg["image_bgr"]

            # resize and encode as JPEG
            h, w, _ = img.shape
            new_w = 320
            new_h = int(new_w * h / w)
            resized = cv2.resize(img, (new_w, new_h))

            ok, buf = cv2.imencode(".jpg", resized)
            if not ok:
                continue

            encoded = base64.b64encode(buf).decode("utf-8")
            video_data = {"image": encoded}
            outer_msg = {"type": 3, "data": json.dumps(video_data)}
            line = json.dumps(outer_msg) + "\n"

            # TODO: send `line` to Unity via socket
            # self._send_message_to_unity(line)
