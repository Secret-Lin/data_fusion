# managers/navigation_manager.py

import asyncio

from common.structured_log import slog

from core.event_bus import EventBus
from core.types import FusedState


class NavigationManager:
    """基于当前 fused state & 术前数据生成导航/引导信息。"""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.state_queue = bus.subscribe("fused_state")
        # 也可以订阅 pre-op / map / HL2 空间锚点等

    async def run(self) -> None:
        slog(
            component=self.__class__.__name__,
            event="run",
            topic_state="fused_state",
            topic_guidance="nav_guidance",
        )
        while True:
            state: FusedState = await self.state_queue.get()
            guidance = self._plan_guidance(state)
            await self.bus.publish("nav_guidance", guidance)

    def _plan_guidance(self, state: FusedState):
        """根据当前位姿计算下一步导航指令 / 虚拟引导几何。"""
        # TODO: 这里填你的导航逻辑（路径规划、虚拟光带、箭头等）
        guidance = {
            "t_host": state.t_host,
            "target_pose": state.pose_world,
            "type": "overlay_path",
        }
        return guidance
