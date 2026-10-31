"""
Manager layer for MR/AR navigation backend.

This package contains high-level managers that implement:
- SensorManager: acquisition of camera / HL2 / OTS streams
- CalibrationManager: intrinsic / extrinsic calibration and coord alignment
- TrackingManager: DOPE-like 6D pose estimation
- FusionManager: multi-sensor fusion and time synchronization
- NavigationManager: navigation and guidance logic
- UnityBridgeManager: communication with Unity MR application / HMD
"""

from importlib import import_module


_EXPORTS = {
    "SensorManager": "sensor_manager",
    "CalibrationManager": "calibration_manager",
    "TrackingManager": "tracking_manager",
    "FrontendObservationManager": "frontend_observation_manager",
    "FoundationPoseManager": "foundationpose_manager",
    "ReplayInputManager": "replay_input_manager",
    "FusionManager": "fusion_manager",
    "BaselineManager": "baseline_manager",
    "NavigationManager": "navigation_manager",
    "RecordManager": "record_manager",
    "UnityBridgeManager": "unity_bridge",
}


def __getattr__(name: str):
    mod_name = _EXPORTS.get(name)
    if mod_name is None:
        raise AttributeError(f"module 'managers' has no attribute '{name}'")
    mod = import_module(f".{mod_name}", __name__)
    obj = getattr(mod, name)
    globals()[name] = obj
    return obj


__all__ = [
    "SensorManager",
    "CalibrationManager",
    "TrackingManager",
    "FrontendObservationManager",
    "FoundationPoseManager",
    "ReplayInputManager",
    "FusionManager",
    "BaselineManager",
    "NavigationManager",
    "RecordManager",
    "UnityBridgeManager",
]
