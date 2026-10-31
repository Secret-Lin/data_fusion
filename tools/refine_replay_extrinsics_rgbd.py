from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.rgbd_tag_alignment import (
    aggregate_fixed_tag_points,
    aggregate_fixed_tag_poses,
    align_common_tag_poses,
    estimate_quest_world_tracking_from_common_tags,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine replay extrinsics from common RGB-D AprilTag points"
    )
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--reference-serial", default="")
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--fixed-output", required=True, type=Path)
    parser.add_argument("--quest-output", required=True, type=Path)
    args = parser.parse_args()

    source_path = args.session / "tag_cube_extrinsics.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    transforms = {
        str(serial): np.asarray(value, dtype=np.float64).reshape(4, 4)
        for serial, value in source["T_world_cam"].items()
    }
    reference_serial = args.reference_serial or sorted(transforms)[0]
    if reference_serial not in transforms:
        raise KeyError(f"Unknown reference serial {reference_serial}")

    tag_pose_maps = {
        serial: aggregate_fixed_tag_poses(
            args.session / "rgbd" / f"{serial}.h5",
            max_frames=args.max_frames,
        )
        for serial in transforms
    }
    point_maps = {
        serial: aggregate_fixed_tag_points(
            args.session / "rgbd" / f"{serial}.h5",
            max_frames=args.max_frames,
        )
        for serial in transforms
    }
    T_world_reference = transforms[reference_serial]
    refined = {reference_serial: T_world_reference.copy()}
    fixed_quality: dict[str, dict[str, float | int | str]] = {
        reference_serial: {
            "method": "reference_camera_unchanged",
            "visible_tags": len(tag_pose_maps[reference_serial]),
        }
    }
    for serial in sorted(transforms):
        if serial == reference_serial:
            continue
        reference_target, quality = align_common_tag_poses(
            tag_pose_maps[serial], tag_pose_maps[reference_serial]
        )
        common_points = sorted(
            set(point_maps[serial]) & set(point_maps[reference_serial])
        )
        if len(common_points) >= 3:
            source_points = np.stack(
                [point_maps[serial][key] for key in common_points]
            )
            target_points = np.stack(
                [point_maps[reference_serial][key] for key in common_points]
            )
            rotation = reference_target[:3, :3]
            translation = np.mean(target_points, axis=0) - rotation @ np.mean(
                source_points, axis=0
            )
            reference_target[:3, 3] = translation
            residuals = np.linalg.norm(
                (rotation @ source_points.T).T + translation - target_points,
                axis=1,
            )
            quality.update(
                {
                    "translation_method": "common_rgbd_tag_centroid",
                    "common_rgbd_points": len(common_points),
                    "rgbd_residual_median_m": float(np.median(residuals)),
                    "rgbd_residual_p95_m": float(np.percentile(residuals, 95)),
                }
            )
        refined[serial] = T_world_reference @ reference_target
        fixed_quality[serial] = {
            "method": "common_tag_multiframe_pnp",
            **quality,
        }

    fixed_payload = {
        "coordinate_system": "y_down",
        "convention": "column vectors; X_world = T_world_cam @ X_cam",
        "world_frame": "tag_cube",
        "generated_from": "common AprilTag multi-frame PnP alignment",
        "reference_serial": reference_serial,
        "T_world_cam": {
            serial: value.tolist() for serial, value in refined.items()
        },
        "T_world_tag": {
            str(tag_id): (T_world_reference @ pose).tolist()
            for tag_id, pose in tag_pose_maps[reference_serial].items()
        },
        "quality": fixed_quality,
    }
    args.fixed_output.parent.mkdir(parents=True, exist_ok=True)
    args.fixed_output.write_text(
        json.dumps(fixed_payload, indent=2), encoding="utf-8"
    )

    quest_path = args.session / "quest" / "quest.h5"
    T_world_tracking, quest_quality = estimate_quest_world_tracking_from_common_tags(
        quest_path,
        tag_pose_maps[reference_serial],
        T_world_reference,
        max_frames=args.max_frames,
    )
    quest_payload = {
        "coordinate_system": "y_down",
        "convention": "column vectors; X_world = T_world_tracking @ X_tracking",
        "generated_from": "common AprilTag multi-frame PnP alignment",
        "reference_serial": reference_serial,
        "T_world_tracking": T_world_tracking.tolist(),
        "samples": int(quest_quality["samples"]),
        "quality": quest_quality,
    }
    args.quest_output.parent.mkdir(parents=True, exist_ok=True)
    args.quest_output.write_text(
        json.dumps(quest_payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"fixed": fixed_payload, "quest": quest_payload}, indent=2))


if __name__ == "__main__":
    main()
