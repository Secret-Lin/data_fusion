from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from adapters.cnos_runner import CnosRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--frames",
        type=str,
        default="",
        help="Comma-separated frame indices; overrides --frame",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cnos_smoke"))
    parser.add_argument("--reference-mask-dir", type=Path, default=None)
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.45)
    args = parser.parse_args()
    frame_indices = (
        [int(value.strip()) for value in args.frames.split(",") if value.strip()]
        if args.frames
        else [int(args.frame)]
    )
    with h5py.File(args.h5, "r") as handle:
        frame_colors = [np.asarray(handle["color"][index]) for index in frame_indices]
        reference_color = np.asarray(handle["color"][args.reference_frame])
        color_format = str(handle.attrs.get("color_format", "")).lower()
    if color_format.startswith("rgb"):
        frame_colors = [
            cv2.cvtColor(color, cv2.COLOR_RGB2BGR) for color in frame_colors
        ]
        reference_color = cv2.cvtColor(reference_color, cv2.COLOR_RGB2BGR)

    runner = CnosRunner(
        cnos_root="thirdparty/CNOS",
        template_dir="assets/cnos_templates/hope_cans",
        sam_model_type="vit_b",
        sam_checkpoint=(
            "thirdparty/CNOS/pretrained/segment-anything/sam_vit_b_01ec64.pth"
        ),
        dino_model_name="dinov2_vitb14",
        dino_root="thirdparty/dinov2",
        dino_checkpoint=(
            "thirdparty/dinov2/weights/dinov2_vitb14_pretrain.pth"
        ),
        device="cuda",
        image_width=640,
        confidence_threshold=float(args.confidence),
        min_mask_fraction=3e-4,
        max_proposals=100,
    )
    objects = ["Tuna", "Corn", "TomatoSauce"]
    runner.prepare(objects)
    if args.reference_mask_dir is not None:
        source_prefix = (
            "quest_quest"
            if args.h5.stem.casefold() == "quest"
            else f"realsense_{args.h5.stem}"
        )
        for object_name in objects:
            mask_path = (
                args.reference_mask_dir / f"{source_prefix}__{object_name}.png"
            )
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            runner.add_reference(reference_color, object_name, mask)
    palette = {
        "Tuna": (0, 255, 255),
        "Corn": (255, 0, 255),
        "TomatoSauce": (0, 255, 0),
    }
    for frame_index, color in zip(frame_indices, frame_colors):
        detections = runner.detect(color, objects)
        candidates = sorted(
            runner.last_candidates,
            key=lambda item: (str(item["object_name"]), -float(item["score"])),
        )
        output_dir = (
            args.output_dir
            if len(frame_indices) == 1
            else args.output_dir / f"frame_{frame_index:06d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        vis = color.copy()
        report = []
        for detection in detections:
            mask_path = output_dir / f"{detection.object_name}.png"
            cv2.imwrite(str(mask_path), detection.mask)
            overlay = vis.copy()
            overlay[detection.mask > 0] = palette[detection.object_name]
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0.0)
            x1, y1, x2, y2 = detection.bbox_xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), palette[detection.object_name], 3)
            cv2.putText(
                vis,
                f"{detection.object_name} {detection.score:.3f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                palette[detection.object_name],
                2,
                cv2.LINE_AA,
            )
            report.append(
                {
                    "object_name": detection.object_name,
                    "score": detection.score,
                    "bbox_xyxy": list(detection.bbox_xyxy),
                    "mask_pixels": int(np.count_nonzero(detection.mask)),
                }
            )
        cv2.imwrite(str(output_dir / "visualization.jpg"), vis)
        (output_dir / "results.json").write_text(
            json.dumps({"detections": report, "candidates": candidates}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"frame": frame_index, "detections": report}, indent=2))


if __name__ == "__main__":
    main()
