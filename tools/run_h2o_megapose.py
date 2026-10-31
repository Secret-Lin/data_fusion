"""H2O single-camera MegaPose RGB-D temporal refinement -> production fusion.

Default initialization reuses only each camera's saved FoundationPose first pose.
Subsequent frames never read GT to initialize, correct or reinitialize tracking.
This inherits GT-mask-assisted initialization, not an independent detector test.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import closing, redirect_stdout
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from adapters.h2o_foundationpose import H2OModelRepository
from adapters.m3t_runner import validate_pose
from adapters.megapose_runner import MegaPoseRunner, prepare_megapose
from experiments.h2o.ablation import VARIANTS, build_config, run_fusion
from experiments.h2o.consistency import discover_takes
from experiments.h2o.dataset import H2OSequence
from experiments.h2o.evaluation import allocate_results, read_jsonl, score_poses, summarize, write_json
from experiments.h2o.prefetch import prefetch_rgbd
from tools.run_h2o_foundationpose import _metrics_by_camera, _write_jsonl
from tools.run_h2o_foundationpose_ablation import code_fingerprints


def initial_poses(path: Path, frame_id: int, cameras: list[str], object_name: str) -> dict:
    rows = [r for r in read_jsonl(path) if int(r["frame_id"]) == frame_id]
    selected = {}
    for row in rows:
        camera = row["camera_id"]
        if camera not in cameras:
            continue
        if camera in selected or row["object_name"] != object_name:
            raise ValueError("Duplicate or mismatched first-frame cached pose")
        selected[camera] = validate_pose(row["pose_cam"])
    if set(selected) != set(cameras):
        raise ValueError(f"Missing first-frame seed poses for {set(cameras) - set(selected)}: {path}")
    return selected


def provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = [root / "adapters/megapose_runner.py", root / "adapters/megapose_transform.py",
             root / "adapters/m3t_runner.py", root / "tools/download_megapose.py", Path(__file__)]
    paths += sorted((root / "thirdparty/megapose6d/src/megapose").rglob("*.py"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_take(args, take: Path, output: Path, models: H2OModelRepository) -> dict:
    output.mkdir(parents=True)
    cameras = H2OSequence.discover_cameras(take)
    sequence = H2OSequence(take, cameras)
    frames = sequence.common_frame_ids()
    if args.max_frames:
        frames = frames[:args.max_frames]
    if not frames:
        raise ValueError("No complete H2O frame group")
    first = sequence.load(cameras[0], frames[0])
    name = first.object_name
    if first.object_id == 0:
        raise ValueError("First frame is background")
    model, mesh = models.load_mesh(name)
    points = np.asarray(mesh.vertices)
    if len(points) > 2048:
        points = points[np.linspace(0, len(points) - 1, 2048).astype(int)]
    np.save(output / "mesh_points.npy", points)
    seed_info = {"mode": args.init_mode}
    if args.init_mode == "foundationpose_first_pose":
        seed_root = args.seed_run_dir or args.subject_root.parent / "results_all_subjects" / args.subject_root.name / "results_0"
        seed_path = seed_root / "takes" / "_".join(take.relative_to(args.subject_root).parts) / "tracking_before_fusion.jsonl"
        seeds = initial_poses(seed_path, frames[0], cameras, name)
        seed_meta = json.loads((seed_path.parent / "summary.json").read_text(encoding="utf-8"))
        seed_info.update(path=str(seed_path), sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest(),
                         source_init_mode=seed_meta.get("init_mode"), first_frame_only=True)
    else:
        seeds = {camera: sequence.load(camera, frames[0]).T_camera_object for camera in cameras}
        seed_info.update(warning="Full GT pose at first frame, stronger initialization than a GT mask")
    write_json(output / "initialization.json", {**seed_info, "poses_cam": {k: v.tolist() for k, v in seeds.items()}})
    # Export loaded textures too (includes the known H2O cocoa MTL alias repair).
    mesh_path = args.mesh_cache / "megapose" / name / (name + ".obj")
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    if not mesh_path.is_file():
        mesh.export(mesh_path)
    runner = MegaPoseRunner(mesh_path, name, data_dir=args.data_dir,
                            iterations=args.iterations, render_workers=args.render_workers)
    raw, gt, errors = [], [], []
    started = time.perf_counter()
    with closing(runner), prefetch_rgbd(sequence, frames, cameras, workers=args.io_workers, prefetch_frames=4) as prefetched:
        for index, (frame_id, images) in enumerate(prefetched):
            reference = sequence.load(cameras[0], frame_id)
            if reference.object_id == 0:
                continue
            if reference.object_name != name:
                raise ValueError("Object changed within take")
            gt.append({"frame_id": frame_id, "t_host": frame_id / args.fps,
                       "object_name": name, "pose_world": reference.T_world_object.tolist()})
            for camera in cameras:
                try:
                    image = images[camera].result()
                    sample = sequence.load(camera, frame_id)
                    tick = time.perf_counter()
                    pose, quality = runner.process(camera, image.color_bgr, image.depth_m, image.K,
                                                   seeds[camera] if index == 0 else None)
                    raw.append({"frame_id": frame_id, "camera_id": camera, "object_name": name,
                                "t_host": frame_id / args.fps, "arrival_time": frame_id / args.fps,
                                "source": f"h2o/{camera}/megapose/{name}", "detector": "megapose",
                                "observation_kind": "megapose_rgbd_temporal_refinement", "score": 1.0,
                                "score_kind": "constant_not_calibrated", "quality_metrics": quality,
                                "mode": "initialize" if index == 0 else "track",
                                "pose_cam": pose.tolist(), "T_world_camera": sample.T_world_camera.tolist(),
                                "pose_world": (sample.T_world_camera @ pose).tolist(),
                                "tracking_and_quality_ms": (time.perf_counter() - tick) * 1000})
                except Exception as exc:
                    errors.append({"frame_id": frame_id, "camera_id": camera, "error": str(exc)})
                    print(f"[{camera} frame {frame_id}] ERROR: {exc}", flush=True)
            if index % 30 == 0 or index == len(frames) - 1:
                print(f"{take.name}: {index + 1}/{len(frames)} frames; {len(raw)} observations; {time.perf_counter()-started:.1f}s", flush=True)
    _write_jsonl(output / "ground_truth.jsonl", gt)
    _write_jsonl(output / "tracking_before_fusion.jsonl", raw)
    report = {"frontend": "megapose_rgbd_temporal_refinement", "init_mode": args.init_mode, "initialization": seed_info,
              "sequence": str(take), "subject_root": str(args.subject_root), "cameras": cameras,
              "object_name": name, "fps": args.fps, "n_frames": len(gt), "n_raw_outputs": len(raw),
              "errors": errors, "status": "observations_saved", "code_sha256": args.code_sha256,
              "frontend_sha256": args.frontend_sha256,
              "effective_fusion_config": asdict(build_config(VARIANTS["full"]).fusion),
              "quality_policy": "MegaPose rendered CAD depth fit; same residual/tolerance definitions as FP, uncalibrated",
              "timing_scope": "Offline tracking and quality measurement, not live end-to-end latency"}
    write_json(output / "summary.json", report)
    with (output / "fusion.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        fused, runtime = asyncio.run(run_fusion(raw, VARIANTS["full"], fps=args.fps))
    _write_jsonl(output / "tracking_after_fusion.jsonl", fused)
    report.update(status="completed" if raw and fused else "no_outputs", n_fused_outputs=len(fused),
                  per_camera=_metrics_by_camera(raw, gt, cameras, points),
                  fusion=summarize(score_poses(gt, fused, mesh_points=points)), fusion_runtime=runtime)
    report["code_changed_during_run"] = (args.code_sha256 != code_fingerprints()
                                         or args.frontend_sha256 != provenance())
    write_json(output / "summary.json", report)
    if not raw or not fused:
        raise RuntimeError(f"No usable MegaPose/fusion outputs; see {output / 'summary.json'}")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", type=Path, default=Path("data/h2o/subject1"))
    parser.add_argument("--take", help="Relative take, e.g. h1/0; omitted = all valid takes")
    parser.add_argument("--mesh-source", type=Path, default=Path("data/h2o/object.zip"))
    parser.add_argument("--mesh-cache", type=Path, default=root / "artifacts/h2o_models")
    parser.add_argument("--data-dir", type=Path, default=root / "artifacts/megapose_data")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--render-workers", type=int, default=1)
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--seed-run-dir", type=Path)
    parser.add_argument("--init-mode", choices=["foundationpose_first_pose", "oracle_gt_pose"], default="foundationpose_first_pose")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-takes", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--io-workers", type=int, default=4)
    parser.add_argument("--fps", type=float, default=30)
    args = parser.parse_args()
    if args.max_frames < 0 or args.max_takes < 0 or args.threads < 1 or args.iterations < 1 or args.render_workers < 1 or args.io_workers < 1 or args.fps <= 0:
        parser.error("Invalid limits/threads/fps")
    prepare_megapose(args.data_dir)
    import torch
    torch.set_num_threads(args.threads)
    args.subject_root = args.subject_root.resolve()
    available = discover_takes(args.subject_root)
    takes = [p for p in available if not args.take or p.relative_to(args.subject_root).as_posix() == args.take.replace('\\', '/')]
    if args.max_takes:
        takes = takes[:args.max_takes]
    if not takes:
        raise ValueError("No requested H2O takes")
    output = allocate_results(args.output_parent or args.subject_root.parent / "megapose_all_subjects" / args.subject_root.name)
    args.frontend_sha256 = provenance()
    args.code_sha256 = code_fingerprints()
    write_json(output / "run_manifest.json", {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                             "code_sha256": args.code_sha256, "frontend_sha256": args.frontend_sha256,
                                             "weights": json.loads((args.data_dir / "download_manifest.json").read_text())
                                             if (args.data_dir / "download_manifest.json").is_file() else None})
    print(f"Results: {output}; {len(takes)} takes; initialization={args.init_mode}", flush=True)
    models = H2OModelRepository(args.mesh_source, extraction_root=args.mesh_cache)
    completed, failed = [], []
    for i, take in enumerate(takes, 1):
        key = take.relative_to(args.subject_root).as_posix()
        print(f"[TAKE {i}/{len(takes)}] {key}", flush=True)
        try:
            result = run_take(args, take, output / "takes" / key.replace('/', '_'), models)
            completed.append({"take": key, **result})
        except Exception as exc:
            failed.append({"take": key, "error": str(exc)})
            print(f"FAILED {key}: {exc}", flush=True)
        write_json(output / "summary.json", {"frontend": "megapose_rgbd_temporal_refinement", "init_mode": args.init_mode,
                   "n_takes_requested": len(takes), "n_takes_completed": len(completed),
                   "n_takes_failed": len(failed), "per_take": completed, "failed_takes": failed})
    if failed:
        raise RuntimeError(f"{len(failed)} takes failed; see {output}")
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
