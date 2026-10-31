"""Read-only structural and video checks for completed real-data smoke runs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2


def check(root):
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "completed", manifest
    if manifest.get('evaluation_status') == 'reference_pending':
        result = {'common_targets': None, 'missing_sensor_methods': [], 'methods': {}}
    else:
        result = json.loads((root / "paired_comparison.json").read_text())
    predictions = json.loads((root / "predictions.json").read_text())
    if manifest.get('evaluation_status') == 'reference_pending':
        result['missing_sensor_methods'] = [k for k,v in predictions.items() if k.startswith('sensor_') and not v]
    videos = {}
    for camera, expected in manifest["frames_by_sensor"].items():
        cap = cv2.VideoCapture(str(root / f"{camera}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            assert image.size > 0
            frames += 1
        cap.release()
        assert frames == expected, (camera, frames, expected)
        videos[camera] = {"decoded_frames": frames, "fps": fps}
    print(json.dumps({"run": str(root), "videos": videos,
                      "evaluation_status": manifest.get('evaluation_status', 'legacy'),
                      "common_targets": result["common_targets"],
                      "missing_sensor_methods": result["missing_sensor_methods"],
                      "fused_observation_counts": dict(Counter(
                          r["observation_count"] for r in predictions["fusion"])),
                      "primary_reference_tagcube0_Tomato2": {
                          k: v.get("T_world_ydown_mesh_native_tagcube0_Tomato2")
                          for k, v in result["methods"].items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    for root in args.run_dirs:
        check(root)
