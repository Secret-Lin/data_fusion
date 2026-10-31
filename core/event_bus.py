import asyncio
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List


class QueuePolicy(str, Enum):
    RELIABLE = "reliable"
    LATEST = "latest"
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"


class EventQueue(asyncio.Queue):
    def __init__(self, maxsize: int, policy: QueuePolicy) -> None:
        super().__init__(maxsize=maxsize)
        self.policy = policy
        self.dropped = 0


class EventBus:
    """In-process async bus with explicit bounded-queue behavior."""

    def __init__(self) -> None:
        self._queues: Dict[str, List[EventQueue]] = defaultdict(list)

    def subscribe(self, topic: str, *, maxsize: int = 0, policy: QueuePolicy | str = QueuePolicy.RELIABLE) -> EventQueue:
        selected = QueuePolicy(policy)
        if selected is QueuePolicy.LATEST and maxsize == 0:
            maxsize = 1
        queue = EventQueue(maxsize=maxsize, policy=selected)
        self._queues[topic].append(queue)
        return queue

    async def publish(self, topic: str, msg: Any) -> None:
        for queue in tuple(self._queues.get(topic, ())):
            if queue.policy is QueuePolicy.RELIABLE:
                await queue.put(msg)
                continue
            if queue.full():
                if queue.policy is QueuePolicy.DROP_NEWEST:
                    queue.dropped += 1
                    continue
                try:
                    queue.get_nowait()
                    queue.task_done()
                    queue.dropped += 1
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(msg)

    def metrics(self) -> dict[str, list[dict[str, int | str]]]:
        return {topic: [{"policy": q.policy.value, "size": q.qsize(), "maxsize": q.maxsize, "dropped": q.dropped} for q in queues] for topic, queues in self._queues.items()}
