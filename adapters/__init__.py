# adapters/__init__.py

import asyncio
import threading
import time
from typing import Callable

from common.structured_log import slog
from core.types import TimeStampedPacket


class BaseBlockingAdapter:
    """在 *线程* 中调用阻塞 SDK，将数据转为 TimeStampedPacket 推进 asyncio 队列。"""

    def __init__(self,
                 loop: asyncio.AbstractEventLoop,
                 out_queue: asyncio.Queue,
                 source_name: str) -> None:
        self.loop = loop
        self.out_queue = out_queue
        self.source_name = source_name
        self._stop_flag = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        slog(
            component=self.__class__.__name__,
            event="start",
            source=self.source_name,
        )
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        slog(
            component=self.__class__.__name__,
            event="stop",
            source=self.source_name,
        )
        self._stop_flag = True

    # ==== 需要子类实现 ====
    def _read_from_sdk_blocking(self):
        """从 HL2SS/NatNet/相机 SDK 中阻塞式读取一帧数据。"""
        raise NotImplementedError

    # ==== 内部线程逻辑 ====
    def _worker(self) -> None:
        while not self._stop_flag:
            raw = self._read_from_sdk_blocking()
            t_host = time.monotonic()
            pkt = TimeStampedPacket(
                t_host=t_host,
                source=self.source_name,
                payload=raw,
            )
            # 把数据线程安全地送进 asyncio 队列
            self.loop.call_soon_threadsafe(self.out_queue.put_nowait, pkt)
