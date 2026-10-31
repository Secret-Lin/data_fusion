"""
Core utilities for MR/AR navigation backend.

This package provides:
- EventBus: in-process async publish/subscribe
- Common data types: TimeStampedPacket, DetectionResult, FusedState
"""

from .event_bus import EventBus
from .types import TimeStampedPacket, DetectionResult, FusedState

__all__ = [
    "EventBus",
    "TimeStampedPacket",
    "DetectionResult",
    "FusedState",
]
