"""Evaluate official rank scores on cached poses without repeating tracking.

The scorer only receives RGB-D, intrinsics, the CAD model and estimated camera
poses. H2O object annotations are used afterwards by the evaluator. Ranking
logits are diagnostic, not calibrated confidences; production fusion weights
are unchanged. Results are written to a fresh results_N directory.
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
from scipy.stats import rankdata, spearmanr

from adapters.foundationpose_runner import FoundationPoseRunner
from adapters.h2o_foundationpose import H2OModelRepository
from experiments.h2o.dataset import H2OSequence
from experiments.h2o.evaluation import allocate_results, read_jsonl, score_poses, summarize, write_json
from experiments.h2o.foundationpose_results import discover_result_takes
from tools.run_h2o_foundationpose_ablation import code_fingerprints


def score_statistics(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get("raw_logit") is not None]
    result = {"n": len(rows), "n_scored": len(valid),
              "status_counts": dict(Counter(r["status"] for r in rows))}
    if not valid:
        return result
    scores = np.array([r["raw_logit"] for r in valid])
    successes = np.array([r["success_5cm_10deg"] for r in valid], dtype=bool)
    result.update(min=float(scores.min()), max=float(scores.max()),
                  p50=float(np.median(scores)), n_unique=int(len(np.unique(scores))))
    for key in ("translation_error_m", "rotation_error_deg", "mesh_add_mm"):
        errors = np.array([r[key] for r in valid])
        rho = float(spearmanr(scores, errors).statistic) if np.ptp(scores) > 0 and np.ptp(errors) > 0 else None
        result[f"spearman_{key}"] = rho if rho is not None and np.isfinite(rho) else None
    n_good, n_bad = int(successes.sum()), int((~successes).sum())
    result["n_good_5cm_10deg"] = n_good
    result["n_bad_5cm_10deg"] = n_bad
    result["auc_good_pose"] = (float((rankdata(scores)[successes].sum() - n_good * (n_good + 1) / 2)
                                    / (n_good * n_bad)) if n_good and n_bad else None)
    durations = [r["elapsed_ms"] for r in valid]
    result["scorer_ms_mean"] = float(np.mean(durations))
    result["scorer_ms_p95"] = float(np.percentile(durations, 95))
    return result


def depth_fit_statistics(rows: list[dict]) -> dict:
    """Summarize optional CAD/RGB-D rendered-depth diagnostics only."""
    fits = [r.get("depth_fit") for r in rows
            if isinstance(r.get("depth_fit"), dict)
            and r["depth_fit"].get("status") == "ok"]
    result = {
        "n": len(rows),
        "n_valid": len(fits),
        "n_errors": sum(
            isinstance(r.get("depth_fit"), dict)
            and r["depth_fit"].get("status") == "error" for r in rows
        ),
    }
    if not fits:
        return result
    for key in ("observed_overlap_fraction", "depth_inlier_fraction",
                "depth_abs_error_median_m", "depth_abs_error_p95_m",
                "occluded_or_depth_mismatch_fraction"):
        values = np.asarray([f[key] for f in fits if f.get(key) is not None], dtype=float)
        if len(values):
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_p50"] = float(np.percentile(values, 50))
            result[f"{key}_p95"] = float(np.percentile(values, 95))
    # These are descriptive correlations only.  No threshold or weight is
    # fitted here; the held-out evaluation still owns all acceptance claims.
    valid_rows = [r for r in rows if isinstance(r.get("depth_fit"), dict)
                  and r["depth_fit"].get("status") == "ok"
                  and r["depth_fit"].get("depth_inlier_fraction") is not None
                  and r.get("translation_error_m") is not None]
    if len(valid_rows) >= 3:
        quality = np.asarray([r["depth_fit"]["depth_inlier_fraction"] for r in valid_rows])
        for error_key in ("translation_error_m", "rotation_error_deg", "mesh_add_mm"):
            if all(r.get(error_key) is not None for r in valid_rows):
                error = np.asarray([r[error_key] for r in valid_rows])
                rho = spearmanr(quality, error).statistic
                result[f"spearman_depth_inlier_fraction_{error_key}"] = (
                    float(rho) if np.isfinite(rho) else None
                )
    return result


def run_take(args: argparse.Namespace, source: Path, output: Path,
             runner: FoundationPoseRunner) -> dict:
    fingerprints = code_fingerprints()
    metadata = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    raw_path = source / "tracking_before_fusion.jsonl"
    raw = read_jsonl(raw_path)
    frames = sorted({r["frame_id"] for r in raw})[::args.stride]
    if args.frames:
        requested = {int(x) for x in args.frames.split(",")}
        frames = [f for f in frames if f in requested]
    if args.max_frames:
        frames = frames[:args.max_frames]
    target_ids = set(frames)
    raw = [r for r in raw if r["frame_id"] in target_ids]
    if not raw:
        raise ValueError(f"No cached poses to score in {source}")
    sequence = H2OSequence(metadata["sequence"], sorted({r["camera_id"] for r in raw}))
    runner.reset()
    runner.prepare(sorted({r["object_name"] for r in raw}))
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    diagnostics = []
    # Flush every result so expensive scoring remains inspectable on failure.
    with (output / "scores.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(raw, 1):
            frame = sequence.load_rgbd(row["camera_id"], row["frame_id"])
            score = runner.score_pose(
                source="h2o/cached_scoring", object_name=row["object_name"],
                color_bgr=frame.color_bgr, depth_m=frame.depth_m,
                camera_matrix=frame.K, pose_cam=np.asarray(row["pose_cam"]),
            )
            item = {"frame_id": row["frame_id"], "camera_id": row["camera_id"],
                    "object_name": row["object_name"], **asdict(score)}
            if getattr(args, "depth_fit", False):
                item["depth_fit"] = runner.measure_depth_fit(
                    source="h2o/cached_scoring", object_name=row["object_name"],
                    depth_m=frame.depth_m, camera_matrix=frame.K,
                    pose_cam=np.asarray(row["pose_cam"]),
                )
            stream.write(json.dumps(item, allow_nan=False) + "\n")
            stream.flush()
            diagnostics.append(item)
            if index == 1 or index % 100 == 0 or index == len(raw):
                print(f"  scores {index}/{len(raw)}, {time.perf_counter()-started:.1f}s, "
                      f"status={score.status}, logit={score.raw_logit}", flush=True)

    # Read GT only after inference; do not select thresholds or fit weights.
    gt = [r for r in read_jsonl(source / "ground_truth.jsonl") if r["frame_id"] in target_ids]
    points = np.load(source / "mesh_points.npy")
    by_camera = defaultdict(list)
    for row in raw:
        by_camera[row["camera_id"]].append(row)
    errors = {}
    methods = {}
    for camera, poses in by_camera.items():
        scores = score_poses(gt, poses, mesh_points=points)
        methods[camera] = summarize(scores)
        errors.update({(camera, r["frame_id"]): r for r in scores})
    evaluated = [{**r, **errors[(r["camera_id"], r["frame_id"]) ]} for r in diagnostics]
    best = {}
    for pose, score in zip(raw, diagnostics):
        if score["raw_logit"] is None:
            continue
        frame_id = pose["frame_id"]
        if frame_id not in best or score["raw_logit"] > best[frame_id][0]:
            best[frame_id] = (score["raw_logit"], pose)
    methods["best_single_pose_rank_diagnostic"] = summarize(score_poses(
        gt, [value[1] for value in best.values()], mesh_points=points))
    fused = [r for r in read_jsonl(source / "tracking_after_fusion.jsonl") if r["frame_id"] in target_ids]
    methods["stored_fusion"] = summarize(score_poses(gt, fused, mesh_points=points))
    camera_choices = Counter(value[1]["camera_id"] for value in best.values())
    translations = np.array([r["pose_world"] for r in gt])[:, :3, 3]
    reference = np.array(gt[0]["pose_world"])
    angular = [score_poses([r], [{**r, "pose_world": reference.tolist()}])[0]["rotation_error_deg"] for r in gt]
    report = {
        "status": "completed" if all(r["status"] == "single_pose_rank" for r in diagnostics) else "scoring_errors",
        "input": str(source), "sequence": metadata["sequence"], "n_frames": len(gt),
        "input_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "code_sha256": fingerprints, "init_mode": metadata.get("init_mode"),
        "tracking_rerun": False, "fusion_weights_changed": False,
        "note": "Official single-pose rank logits, +100 removed. Cross-frame/camera probabilities are uncalibrated. "
                "Best-rank selector is an exploratory diagnostic, not a deployed fusion method. No GT fitting.",
        "statistics": score_statistics(evaluated),
        "depth_fit_statistics": depth_fit_statistics(evaluated) if getattr(args, "depth_fit", False) else None,
        "per_camera_statistics": {c: score_statistics([r for r in evaluated if r["camera_id"] == c]) for c in by_camera},
        "per_camera_depth_fit_statistics": {
            c: depth_fit_statistics([r for r in evaluated if r["camera_id"] == c])
            for c in by_camera
        } if getattr(args, "depth_fit", False) else None,
        "best_rank_camera_counts": dict(camera_choices), "methods": methods,
        "gt_motion": {"axis_range_m": np.ptp(translations, axis=0).tolist(),
                      "max_displacement_from_first_m": float(np.linalg.norm(translations-translations[0], axis=1).max()),
                      "max_rotation_from_first_deg": max(angular)},
    }
    write_json(output / "summary.json", report)
    lines = ["# Cached FoundationPose rank-score audit", "", f"Input: `{source}`", "", report["note"], "",
             f"Frames: {len(gt)}; scored: {report['statistics']['n_scored']}/{len(raw)}; status: {report['status']}", "",
             "| Method | Translation mean (mm) | Translation P95 (mm) | Rotation P95 (deg) | ADD P95 (mm) |",
             "|---|---:|---:|---:|---:|"]
    for name, m in methods.items():
        vals = [m[k] for k in ("translation_mean_mm", "translation_p95_mm", "rotation_p95_deg", "mesh_add_p95_mm")]
        lines.append(f"| {name} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in vals) + " |")
    lines += ["", "See summary.json for score distributions, timing, error correlations and GT motion."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, default=repo / "artifacts" / "h2o_score_audit")
    parser.add_argument("--mesh-source", type=Path, default=Path("data/h2o/object.zip"))
    parser.add_argument("--mesh-cache", type=Path, default=repo / "artifacts" / "h2o_models")
    parser.add_argument("--foundationpose-root", type=Path, default=repo / "thirdparty" / "FoundationPose")
    parser.add_argument("--takes", default="")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frames", default="", help="Optional diagnostic frame ids, e.g. 0,100,300,700")
    parser.add_argument("--depth-fit", action="store_true",
                        help="Render CAD pose and report RGB-D depth-fit diagnostics; diagnostic only")
    args = parser.parse_args()
    if args.stride < 1 or args.max_frames < 0:
        raise ValueError("stride must be positive and max-frames nonnegative")
    takes = discover_result_takes(args.run_dir.resolve(), args.takes)
    output = allocate_results(args.output_parent.resolve())
    print(f"Results: {output}", flush=True)
    write_json(output / "run_manifest.json", {"arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                                            "code_sha256": code_fingerprints()})
    repository = H2OModelRepository(args.mesh_source, extraction_root=args.mesh_cache, scale_to_m=1.0)
    runner = FoundationPoseRunner(args.foundationpose_root, model_repository=repository)
    reports = []
    for key, source in takes:
        destination = output if len(takes) == 1 else output / "takes" / key
        print(f"[TAKE] {key}", flush=True)
        reports.append(run_take(args, source, destination, runner))
    if len(takes) > 1:
        write_json(output / "summary.json", {"input": str(args.run_dir), "reports": reports})
    print(f"Report: {output / 'summary.json'}", flush=True)
    return int(any(r["status"] != "completed" for r in reports))


if __name__ == "__main__":
    raise SystemExit(main())
