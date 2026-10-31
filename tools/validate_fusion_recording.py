from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from config import BackendConfig
from core.event_bus import EventBus
from core.types import DetectionResult
from managers.fusion_manager import FusionManager


async def _run(path: Path) -> dict[str, int | float]:
    manager = FusionManager(EventBus(), asyncio.get_running_loop(), BackendConfig())
    accepted = 0
    rejected = 0
    shared_state_observations = 0
    previous_step: dict[str, int] = {}
    last_pose: np.ndarray | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        pose = record.get("pose_world")
        if pose is None:
            continue
        result = await manager._on_pose_message(
            DetectionResult(
                t_host=float(record["t_host"]),
                source=str(record["source"]),
                pose_world=np.asarray(pose, dtype=np.float64),
                score=float(record.get("score", 1.0)),
                reproj_error=record.get("reproj_error"),
            )
        )
        if result is None:
            rejected += 1
            continue
        object_name, step = result
        if previous_step.get(object_name) == step:
            shared_state_observations += 1
        previous_step[object_name] = step
        system = manager._gtsam_instances[object_name]
        system.update()
        last_pose = system.estimate_tool_pose(step)
        if last_pose is None or not np.isfinite(last_pose).all():
            raise RuntimeError(f"Invalid fused pose at step {step}")
        if not np.isclose(np.linalg.det(last_pose[:3, :3]), 1.0, atol=1e-5):
            raise RuntimeError(f"Invalid fused rotation at step {step}")
        accepted += 1

    return {
        "input_records": accepted + rejected,
        "accepted": accepted,
        "rejected": rejected,
        "fused_states": sum(manager._steps.values()),
        "shared_state_observations": shared_state_observations,
        "last_translation_norm_m": (
            0.0 if last_pose is None else float(np.linalg.norm(last_pose[:3, 3]))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args.recording)), indent=2))


if __name__ == "__main__":
    main()
