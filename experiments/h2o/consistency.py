"""Measure same-frame cross-camera H2O annotation consistency, with no noise.

Pose labels are evaluation references, NOT independently measured tracker
outputs. Agreement between these labels is not evidence of physical accuracy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from experiments.h2o.dataset import H2OSequence
from experiments.h2o.evaluation import allocate_results, write_json
from experiments.h2o.geometry import rotation_distance_deg


def discover_takes(subject: Path) -> list[Path]:
    takes = []
    for group in sorted(subject.iterdir()):
        if not group.is_dir():
            continue
        for take in sorted(group.iterdir(), key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name)):
            if take.is_dir() and take.name.isdigit():
                H2OSequence.discover_cameras(take)
                takes.append(take)
    if not takes:
        raise ValueError(f"No valid H2O takes found below {subject}")
    return takes


def compare_poses(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {
        "translation_mm": float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000),
        "rotation_deg": rotation_distance_deg(a[:3, :3], b[:3, :3]),
        "raw_rotation_frobenius": float(np.linalg.norm(a[:3, :3] - b[:3, :3])),
    }


def statistics(rows: list[dict]) -> dict:
    result = {"n": len(rows)}
    for key in ("translation_mm", "rotation_deg", "raw_rotation_frobenius"):
        values = np.asarray([r[key] for r in rows], dtype=float)
        for name, value in (("mean", np.mean(values) if len(values) else None),
                            ("p50", np.percentile(values, 50) if len(values) else None),
                            ("p95", np.percentile(values, 95) if len(values) else None),
                            ("max", np.max(values) if len(values) else None)):
            result[f"{key}_{name}"] = float(value) if value is not None else None
    return result


def analyze_take(take: Path, *, fps: float, max_frames: int = 0) -> tuple[dict, list[dict], list[dict]]:
    cameras = H2OSequence.discover_cameras(take)
    sequence = H2OSequence(take, cameras)
    available = {cam: set(sequence.available_frame_ids(cam)) for cam in cameras}
    frames = sorted(set().union(*available.values()))
    if max_frames:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"No complete annotation pairs below {take}")
    pairs, poses, inconsistent_ids = [], [], []
    annotated_frames = multi_view_frames = 0
    per_camera = {cam: {"n_pose_files": len(available[cam]), "n_annotated": 0,
                        "n_background": 0, "n_missing_in_selected_frames": 0,
                        "max_rotation_orthogonality_defect": 0.0} for cam in cameras}
    for frame in frames:
        samples = []
        for cam in cameras:
            if frame not in available[cam]:
                per_camera[cam]["n_missing_in_selected_frames"] += 1
                continue
            sample = sequence.load(cam, frame)
            if sample.object_id == 0:
                per_camera[cam]["n_background"] += 1
                continue  # A nonzero label denotes annotation, NOT visibility.
            per_camera[cam]["n_annotated"] += 1
            matrix = sample.T_world_object
            defect = float(np.linalg.norm(matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3)))
            per_camera[cam]["max_rotation_orthogonality_defect"] = max(
                defect, per_camera[cam]["max_rotation_orthogonality_defect"])
            poses.append({"frame_id": frame, "t_relative_s": (frame - frames[0]) / fps,
                          "camera_id": cam, "object_name": sample.object_name,
                          "pose_world": matrix.tolist(), "kind": "unmodified_annotation"})
            samples.append(sample)
        annotated_frames += bool(samples)
        multi_view_frames += len(samples) >= 2
        if len({s.object_id for s in samples}) > 1:
            inconsistent_ids.append({"frame_id": frame, "ids": {s.camera_id: s.object_id for s in samples}})
        for a, b in combinations(samples, 2):
            if a.object_id != b.object_id:
                continue
            pairs.append({"frame_id": frame, "t_relative_s": (frame - frames[0]) / fps,
                          "camera_a": a.camera_id, "camera_b": b.camera_id,
                          "object_name": a.object_name,
                          **compare_poses(a.T_world_object, b.T_world_object)})
    grouped = defaultdict(list)
    for row in pairs:
        grouped[f"{row['camera_a']}--{row['camera_b']}"].append(row)
    report = {"sequence": str(take), "cameras": cameras, "n_frames": len(frames),
              "n_annotated_frames": annotated_frames, "n_multiview_frames": multi_view_frames,
              "n_world_poses": len(poses), "per_camera": per_camera,
              "identity_mismatches": inconsistent_ids, "all_pairs": statistics(pairs),
              "per_pair": {key: statistics(rows) for key, rows in grouped.items()}}
    return report, pairs, poses


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


def run(subject: Path, output: Path, *, fps: float = 30.0, max_frames: int = 0,
        max_takes: int = 0, reference_camera: str = "cam0") -> dict:
    takes = discover_takes(subject)
    if max_takes:
        takes = takes[:max_takes]
    reports, all_pairs = [], []
    for index, take in enumerate(takes):
        key = take.relative_to(subject).as_posix()
        print(f"[TAKE {index + 1}/{len(takes)}] {key}", flush=True)
        report, pairs, poses = analyze_take(take, fps=fps, max_frames=max_frames)
        report["take"] = key
        directory = output / "takes" / "_".join(take.relative_to(subject).parts)
        directory.mkdir(parents=True, exist_ok=False)
        write_jsonl_gz(directory / "world_annotations.jsonl.gz", poses)
        write_jsonl_gz(directory / "pairwise_differences.jsonl.gz", pairs)
        write_json(directory / "report.json", report)
        reports.append(report)
        all_pairs.extend({"take": key, **r} for r in pairs)
        print(f"  frames={report['n_annotated_frames']}, pairs={len(pairs)}, "
              f"max_translation_mm={report['all_pairs']['translation_mm_max']}", flush=True)
    by_pair, by_object = defaultdict(list), defaultdict(list)
    for row in all_pairs:
        by_pair[f"{row['camera_a']}--{row['camera_b']}"].append(row)
        by_object[row["object_name"]].append(row)
    pairs_summary = {key: statistics(rows) for key, rows in sorted(by_pair.items())}
    summary = {"scope": "Same-frame annotation consistency, NOT independent observation/absolute accuracy",
               "subject_root": str(subject), "artificial_noise": False, "artificial_dropout": False,
               "n_takes": len(reports), "n_annotated_frames": sum(r["n_annotated_frames"] for r in reports),
               "n_world_poses": sum(r["n_world_poses"] for r in reports),
               "n_identity_mismatch_frames": sum(len(r["identity_mismatches"]) for r in reports),
               "reference_camera": reference_camera,
               "rotation_metric": "Nearest SO(3) per stored rotation; atan2 geodesic. No world alignment fitted.",
               "all_pairs": statistics(all_pairs), "per_pair": pairs_summary,
               "per_object": {key: statistics(rows) for key, rows in sorted(by_object.items())},
               "per_take": reports}
    write_json(output / "summary.json", summary)
    fields = ["camera_pair", *statistics([]).keys()]
    with (output / "pairwise_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"camera_pair": pair, **stats} for pair, stats in pairs_summary.items())
    text = "# H2O annotation consistency (no artificial noise)\n\n"
    text += (f"{len(reports)} takes; {summary['n_annotated_frames']} annotated time frames; "
             f"{summary['n_world_poses']} per-camera world annotations; {len(all_pairs)} same-frame pairs.\n\n")
    text += ("These poses are dataset ground-truth annotations expressed in different camera frames, "
             "not independently estimated tracking observations. They are converted using "
             "T_world_object = T_world_camera @ T_camera_object. All comparisons use the same frame "
             "and object; object motion between frames is not counted as error. "
             "No noise, dropout, delay, temporal smoothing or fitted world alignment is added. "
             "Frame times are reconstructed from frame ids and the stated frame rate.\n\n")
    text += ("Angles are computed after projecting each slightly rounded stored rotation onto SO(3). "
             "This removes non-orthogonality numerical artifacts, not physical rotation disagreement. "
             "Raw world matrices remain unchanged in world_annotations.jsonl.gz; raw rotation-matrix "
             "differences are also retained.\n\n")
    text += "| Camera pair | N | Translation P50 mm | Translation P95 mm | Translation max mm | Rotation P95 deg | Rotation max deg |\n"
    text += "|---|---:|---:|---:|---:|---:|---:|\n"
    for pair, metric in pairs_summary.items():
        text += (f"| {pair} | {metric['n']} | {metric['translation_mm_p50']:.6e} | "
                 f"{metric['translation_mm_p95']:.6e} | {metric['translation_mm_max']:.6e} | "
                 f"{metric['rotation_deg_p95']:.6e} | {metric['rotation_deg_max']:.6e} |\n")
    text += ("\nNear-zero inter-view disagreement does not establish near-zero physical error. "
             "The views can share the same annotation and systematic error. "
             "H2O annotations are produced using DenseFusion selection and multi-view ICP; "
             "real fusion evaluation requires independently inferred RGB-D poses, scored against these labels.\n\n"
             "Sources: [H2O annotation method, section 3.2](https://arxiv.org/html/2104.11181v2), "
             "[official data format](https://github.com/taeinkwon/h2odataset).\n")
    (output / "summary.md").write_text(text, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", type=Path, default=Path("data/h2o/subject1"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-takes", type=int, default=0)
    args = parser.parse_args(argv)
    if not np.isfinite(args.fps) or args.fps <= 0 or min(args.max_frames, args.max_takes) < 0:
        parser.error("fps must be positive and limits nonnegative")
    subject = args.subject_root.resolve()
    discover_takes(subject)  # Validate input before allocating an output directory.
    output = args.output.resolve() if args.output else allocate_results(subject.parent)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty; choose a new results_N: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", {"subject_root": str(subject), "fps": args.fps,
               "max_frames": args.max_frames, "max_takes": args.max_takes, "artificial_noise": False})
    print(f"Results: {output}", flush=True)
    result = run(subject, output, fps=args.fps, max_frames=args.max_frames, max_takes=args.max_takes)
    print(f"Completed {result['n_takes']} takes. Report: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
