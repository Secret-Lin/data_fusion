"""Exploratory multi-view RGB-D verification of saved H2O pose candidates.

Each camera compares ALL current world-pose candidates in its own image with
the official FoundationPose ranker. Average within-image ranks select a pose;
raw logits from different images are never compared directly. This experiment
does not change production fusion, run register/track_one, or use GT to rank.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata

from adapters.foundationpose_runner import FoundationPoseRunner
from adapters.h2o_foundationpose import H2OModelRepository
from experiments.h2o.dataset import H2OSequence
from experiments.h2o.evaluation import allocate_results, read_jsonl, score_poses, summarize, write_json
from experiments.h2o.foundationpose_results import discover_result_takes
from tools.run_h2o_foundationpose import _write_jsonl
from tools.run_h2o_foundationpose_ablation import code_fingerprints


def aggregate_view_ranks(score_matrix: np.ndarray) -> np.ndarray:
    """Rank 0 is best; ties receive average rank and missing values rank last."""
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or scores.shape[1] < 1:
        raise ValueError("Expected a nonempty view-by-candidate matrix")
    valid_views = np.isfinite(scores).any(axis=1)
    if not valid_views.any():
        raise ValueError("No scored views")
    ranks = [rankdata(-np.where(np.isfinite(row), row, -np.inf), method="average") - 1
             for row in scores[valid_views]]
    return np.mean(ranks, axis=0) / max(scores.shape[1] - 1, 1)


def run_take(args, source: Path, output: Path, runner: FoundationPoseRunner) -> dict:
    fingerprints = code_fingerprints()
    metadata = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    raw_path = source / "tracking_before_fusion.jsonl"
    raw = read_jsonl(raw_path)
    frames = sorted({r["frame_id"] for r in raw})[::args.stride]
    if args.frames:
        frames = [f for f in frames if f in {int(x) for x in args.frames.split(",")}]
    if args.max_frames:
        frames = frames[:args.max_frames]
    frame_set = set(frames)
    raw = [r for r in raw if r["frame_id"] in frame_set]
    if not raw:
        raise ValueError(f"No poses in {source}")
    by_frame = defaultdict(list)
    for row in raw:
        by_frame[row["frame_id"]].append(row)
    cameras = sorted({r["camera_id"] for r in raw})
    sequence = H2OSequence(metadata["sequence"], cameras)
    runner.reset()
    runner.prepare([metadata["object_name"]])
    output.mkdir(parents=True, exist_ok=True)
    predictions, durations, diagnostics, errors = [], [], [], []
    started = time.perf_counter()
    with (output / "ranks.jsonl").open("w", encoding="utf-8") as stream:
        for index, frame_id in enumerate(frames, 1):
            candidates = sorted(by_frame[frame_id], key=lambda r: r["camera_id"])
            world_poses = np.asarray([r["pose_world"] for r in candidates])
            matrix, per_view = [], []
            frame_started = time.perf_counter()
            for view in candidates:
                frame = sequence.load_rgbd(view["camera_id"], frame_id)
                camera_poses = np.linalg.inv(np.asarray(view["T_world_camera"])) @ world_poses
                values = np.full(len(candidates), np.nan)
                indices = np.flatnonzero(camera_poses[:, 2, 3] > 0.01)
                scores = runner.rank_poses(
                    source="h2o/reranking", object_name=view["object_name"],
                    color_bgr=frame.color_bgr, depth_m=frame.depth_m, camera_matrix=frame.K,
                    poses_cam=camera_poses[indices],
                ) if len(indices) else []
                for candidate_index, score in zip(indices, scores):
                    if score.raw_logit is not None:
                        values[candidate_index] = score.raw_logit
                    else:
                        errors.append({"frame_id": frame_id, "camera_id": view["camera_id"], **asdict(score)})
                matrix.append(values)
                per_view.append({"camera_id": view["camera_id"], "logits": [float(v) if np.isfinite(v) else None for v in values]})
            matrix = np.asarray(matrix)
            if not np.isfinite(matrix).any():
                errors.append({"frame_id": frame_id, "error": "All candidate scores unavailable"})
                continue
            ranks = aggregate_view_ranks(matrix)
            best = int(np.argmin(ranks))
            predictions.append(candidates[best])
            duration = 1000 * (time.perf_counter() - frame_started)
            durations.append(duration)
            diagnostic = {"frame_id": frame_id, "candidate_cameras": [r["camera_id"] for r in candidates],
                          "views": per_view, "mean_rank": ranks.tolist(),
                          "selected_camera": candidates[best]["camera_id"], "elapsed_ms": duration}
            diagnostics.append(diagnostic)
            stream.write(json.dumps(diagnostic, allow_nan=False) + "\n")
            stream.flush()
            if index == 1 or index % 50 == 0 or index == len(frames):
                print(f"  frames {index}/{len(frames)}, {time.perf_counter()-started:.1f}s, "
                      f"selected={candidates[best]['camera_id']}", flush=True)
    _write_jsonl(output / "rank_selected_poses.jsonl", predictions)
    # Only the evaluator receives GT; no learned camera priorities or thresholds.
    gt = [r for r in read_jsonl(source / "ground_truth.jsonl") if r["frame_id"] in frame_set]
    points = np.load(source / "mesh_points.npy")
    methods = {camera: [r for r in raw if r["camera_id"] == camera] for camera in cameras}
    methods["stored_fusion"] = [r for r in read_jsonl(source / "tracking_after_fusion.jsonl") if r["frame_id"] in frame_set]
    methods["multi_view_rank_selection_exploratory"] = predictions
    report = {"input": str(source), "sequence": metadata["sequence"], "init_mode": metadata.get("init_mode"),
              "n_frames": len(gt), "status": "scoring_errors" if errors else "completed", "errors": errors,
              "input_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(), "code_sha256": fingerprints,
              "production_changed": False, "gt_used_for_selection": False,
              "selection": "Mean normalized within-view ranks; tie uses sorted camera order; no temporal factors",
              "selected_camera_counts": dict(Counter(r["camera_id"] for r in predictions)),
              "frame_compute_ms_mean_including_rgbd_io": float(np.mean(durations)) if durations else None,
              "frame_compute_ms_p95_including_rgbd_io": float(np.percentile(durations, 95)) if durations else None,
              "methods": {name: summarize(score_poses(gt, rows, mesh_points=points)) for name, rows in methods.items()}}
    write_json(output / "summary.json", report)
    lines = ["# Multi-view candidate-rank diagnostic", "", f"Input: `{source}`", "",
             "Exploratory pose selection from cached candidates; no tracking rerun or production-fusion change. "
             "GT is used only for evaluation. Latency includes RGB-D I/O and five GPU ranking calls per frame.", "",
             "| Method | Translation mean (mm) | P95 (mm) | Rotation P95 (deg) | ADD P95 (mm) |",
             "|---|---:|---:|---:|---:|"]
    for name, m in report["methods"].items():
        values = [m[k] for k in ("translation_mean_mm", "translation_p95_mm", "rotation_p95_deg", "mesh_add_p95_mm")]
        lines.append(f"| {name} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in values) + " |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, default=repo / "artifacts" / "h2o_rank_audit")
    parser.add_argument("--takes", default="")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--frames", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--mesh-source", type=Path, default=Path("data/h2o/object.zip"))
    parser.add_argument("--mesh-cache", type=Path, default=repo / "artifacts" / "h2o_models")
    parser.add_argument("--foundationpose-root", type=Path, default=repo / "thirdparty" / "FoundationPose")
    args = parser.parse_args()
    if args.stride < 1 or args.max_frames < 0:
        raise ValueError("Invalid frame limits")
    takes = discover_result_takes(args.run_dir.resolve(), args.takes)
    output = allocate_results(args.output_parent.resolve())
    print(f"Results: {output}", flush=True)
    write_json(output / "run_manifest.json", {"arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                                            "code_sha256": code_fingerprints()})
    repository = H2OModelRepository(args.mesh_source, extraction_root=args.mesh_cache, scale_to_m=1.0)
    runner = FoundationPoseRunner(args.foundationpose_root, model_repository=repository)
    reports = []
    for key, source in takes:
        print(f"[TAKE] {key}", flush=True)
        reports.append(run_take(args, source, output if len(takes) == 1 else output / "takes" / key, runner))
    if len(takes) > 1:
        write_json(output / "summary.json", {"input": str(args.run_dir), "reports": reports})
    print(f"Report: {output / 'summary.json'}", flush=True)
    return int(any(r["status"] != "completed" for r in reports))


if __name__ == "__main__":
    raise SystemExit(main())
