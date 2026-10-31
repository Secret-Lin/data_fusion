#!/usr/bin/env python3
"""Utility to load camera latency JSON and determine hardware sync configuration."""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_camera_latency_json(json_path: str) -> dict:
    """Load camera latency results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_hardware_sync_groups(data: dict) -> Dict[int, List[str]]:
    """Get hardware sync groups from latency data.
    
    Returns dict mapping FPS -> list of camera serials that can sync at that FPS.
    """
    return data.get("hardware_sync_groups", {})


def get_best_fps_for_camera(data: dict, serial: str) -> int:
    """Get the best (lowest latency) FPS for a specific camera."""
    cameras = data.get("cameras", {})
    if serial in cameras:
        return cameras[serial].get("best_fps", 60)
    return 60


def get_camera_config_from_json(json_path: str) -> Tuple[list, Dict[int, List[str]]]:
    """Load camera configs and sync groups from JSON.
    
    Returns:
        - List of (serial, fps, latency_ms) tuples for each camera
        - Dict of FPS -> list of serials for hardware sync
    """
    data = load_camera_latency_json(json_path)
    
    cameras = data.get("cameras", {})
    configs = []
    
    for serial, cam_data in cameras.items():
        best_fps = cam_data.get("best_fps", 60)
        available = cam_data.get("available_fps", {})
        
        if str(best_fps) in available:
            latency = available[str(best_fps)]["avg_ms"]
        else:
            latency = 0.0
            
        configs.append({
            "serial": serial,
            "name": cam_data.get("name", "Unknown"),
            "fps": best_fps,
            "latency_ms": latency,
        })
    
    sync_groups = get_hardware_sync_groups(data)
    
    return configs, sync_groups


def print_sync_recommendation(json_path: str) -> None:
    """Print recommended hardware sync configuration."""
    configs, sync_groups = get_camera_config_from_json(json_path)
    
    print("=" * 60)
    print("HARDWARE SYNC RECOMMENDATION")
    print("=" * 60)
    
    # Find the best sync group: prefer higher FPS, but need at least 2 cameras
    # Priority: FPS (higher better), then number of cameras (more better)
    valid_groups = {fps: cams for fps, cams in sync_groups.items() if len(cams) >= 2}
    if not valid_groups:
        print("\nNo suitable hardware sync group found (need 2+ cameras at same FPS)")
        print("\nIndividual camera configs:")
        for cfg in configs:
            print(f"    CameraConfig(name=\"{cfg['name']}\", serial=\"{cfg['serial']}\", "
                  f"fps={cfg['fps']}, sync_mode=0, latency_ms={cfg['latency_ms']}),")
        return
        
    best_fps = max(valid_groups.keys(), key=lambda fps: (fps, len(sync_groups[fps])), default=None)
    
    if best_fps and len(sync_groups[best_fps]) > 1:
        print(f"\nRecommended: {best_fps}fps sync group")
        print(f"Cameras: {sync_groups[best_fps]}")
        
        print("\nConfig for config.py:")
        master = sync_groups[best_fps][0]
        for i, serial in enumerate(sync_groups[best_fps]):
            sync_mode = 1 if i == 0 else 2
            cfg = next((c for c in configs if c["serial"] == serial), None)
            if cfg:
                print(f"    CameraConfig(name=\"{cfg['name']}\", serial=\"{serial}\", "
                      f"fps={cfg['fps']}, sync_mode={sync_mode}, latency_ms={cfg['latency_ms']}),")
    else:
        print("\nNo suitable hardware sync group found (need 2+ cameras at same FPS)")
        print("\nIndividual camera configs:")
        for cfg in configs:
            print(f"    CameraConfig(name=\"{cfg['name']}\", serial=\"{cfg['serial']}\", "
                  f"fps={cfg['fps']}, sync_mode=0, latency_ms={cfg['latency_ms']}),")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python camera_latency_loader.py <json_path>")
        print("Example: python camera_latency_loader.py camera_latency.json")
        sys.exit(1)
    
    print_sync_recommendation(sys.argv[1])
