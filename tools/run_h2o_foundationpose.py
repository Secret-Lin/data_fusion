"""Run the current FoundationPose runner on H2O RGB-D sequences.

The default ``oracle_gt`` initialization is deliberately an integration smoke
test: it projects the H2O annotated pose and mesh to make the first-frame mask.
It must not be reported as an independent tracking benchmark. For a real
benchmark, provide an independent per-camera first-frame mask with
``--init-mode mask_dir``.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from adapters.foundationpose_runner import FoundationPoseRunner
from adapters.h2o_foundationpose import H2OModelRepository
from experiments.h2o.ablation import VARIANTS, build_config, run_fusion
from experiments.h2o.consistency import discover_takes
from experiments.h2o.dataset import H2OSequence
from experiments.h2o.evaluation import allocate_results, score_poses, summarize, write_json
from experiments.h2o.performance import StageTimings
from experiments.h2o.prefetch import prefetch_rgbd
from tools.run_h2o_foundationpose_ablation import code_fingerprints


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, default=_json_default, allow_nan=False) + "\n")


def _project_mesh_mask(
    mesh_vertices_m: np.ndarray,
    T_camera_object: np.ndarray,
    K: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    points = np.asarray(mesh_vertices_m, dtype=np.float64)
    pose = np.asarray(T_camera_object, dtype=np.float64)
    camera = (pose[:3, :3] @ points.T).T + pose[:3, 3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.01)
    if int(valid.sum()) < 3:
        raise ValueError("H2O GT pose projects fewer than three mesh vertices")
    image = camera[valid] @ np.asarray(K, dtype=np.float64).T
    uv = image[:, :2] / image[:, 2:3]
    height, width = shape
    uv[:, 0] = np.clip(uv[:, 0], 0, width - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, height - 1)
    hull = cv2.convexHull(np.round(uv).astype(np.int32))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if int(np.count_nonzero(mask)) < 30:
        raise ValueError("Projected H2O initialization mask is too small")
    return mask


def _mask_from_directory(mask_root: Path, camera_id: str, frame_id: int) -> np.ndarray:
    path = mask_root / camera_id / f"{frame_id:06d}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Missing independent initialization mask: {path}")
    if int(np.count_nonzero(mask)) < 30:
        raise ValueError(f"Initialization mask is empty: {path}")
    return mask


async def _run_fusion(rows: list[dict[str, Any]], *, fps: float = 30.0,
                      base_overrides: dict | None = None) -> list[dict[str, Any]]:
    predictions, _ = await run_fusion(rows, VARIANTS["full"], fps=fps, base_overrides=base_overrides)
    for row in predictions:
        row["t_host"] = row["capture_time_s"]
    return sorted(predictions, key=lambda row: row["frame_id"])


def _metrics_by_camera(
    rows: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    cameras: list[str],
    mesh_points: np.ndarray,
) -> dict[str, dict[str, Any]]:
    return {
        camera: summarize(
            score_poses(
                ground_truth,
                [row for row in rows if row["camera_id"] == camera],
                mesh_points=mesh_points,
            )
        )
        for camera in cameras
    }


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", type=Path, default=Path(r"data/h2o/subject1"))
    parser.add_argument(
        "--take",
        help="One sequence relative to subject root, e.g. h1/0. Omit to run every valid take.",
    )
    parser.add_argument(
        "--all-takes",
        action="store_true",
        help="Explicitly request every valid take (the default when --take is omitted).",
    )
    parser.add_argument("--cameras", default="all", help="all or comma-separated cam0,cam1,...")
    parser.add_argument("--mesh-source", type=Path, default=Path(r"data/h2o/object.zip"))
    parser.add_argument("--mesh-cache", type=Path, default=repo / "artifacts" / "h2o_models")
    parser.add_argument("--foundationpose-root", type=Path, default=repo / "thirdparty" / "FoundationPose")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output", type=Path)
    outputs.add_argument("--output-parent", type=Path, help="Default: subject root's parent; reserves results_N")
    parser.add_argument("--fusion-config", type=Path, help="JSON object overriding FusionConfig defaults")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--max-takes",
        type=int,
        default=0,
        help="Limit all-take mode to the first N takes; 0 means all takes.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--register-iterations", type=int, default=5)
    parser.add_argument("--track-iterations", type=int, default=2)
    parser.add_argument("--score-tracking", action="store_true",
                        help="Export uncalibrated official rank logits each frame; adds GPU work")
    parser.add_argument("--depth-fit-quality", action="store_true",
                        help="Render CAD depth and store q_fit diagnostics per observation; adds GPU work")
    parser.add_argument("--io-workers", type=int, default=4,
                        help="CPU RGB-D readers; 0 preserves synchronous reads (default: 4)")
    parser.add_argument("--prefetch-frames", type=int, default=4,
                        help="Bounded read-ahead groups, each containing all cameras (default: 4)")
    parser.add_argument("--torch-threads", type=int, default=4,
                        help="CPU threads for PyTorch host operations; 0 keeps its default (default: 4)")
    parser.add_argument(
        "--init-mode",
        choices=("oracle_gt", "mask_dir"),
        default="oracle_gt",
        help="oracle_gt is only an integration smoke test; mask_dir is the independent benchmark mode",
    )
    parser.add_argument("--mask-root", type=Path, help="Root containing <camera>/<frame>.png for mask_dir")
    return parser.parse_args()


def _run_take(
    args: argparse.Namespace,
    subject: Path,
    sequence_dir: Path,
    output: Path,
    repository: H2OModelRepository,
    runner: FoundationPoseRunner,
) -> dict[str, Any]:
    """Run one take and write its complete, self-contained result directory.

    The runner and model repository are shared by all takes in all-take mode so
    that FoundationPose networks are loaded once.  Estimator state is reset at
    the take boundary; otherwise a tracker could accidentally carry state from
    one independent H2O recording into the next one.
    """
    take_started = time.perf_counter()
    timings = StageTimings()
    output.mkdir(parents=True, exist_ok=True)
    runner.reset()
    discovered = H2OSequence.discover_cameras(sequence_dir)
    cameras = discovered if args.cameras.strip().lower() == "all" else [
        value.strip() for value in args.cameras.split(",") if value.strip()
    ]
    missing = sorted(set(cameras) - set(discovered))
    if missing:
        raise ValueError(f"Requested H2O cameras are unavailable: {missing}; available={discovered}")
    sequence = H2OSequence(sequence_dir, cameras)
    frame_ids = sequence.common_frame_ids()
    if args.max_frames:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise ValueError(f"No complete H2O frames in {sequence_dir}")

    first_sample = sequence.load(cameras[0], frame_ids[0])
    if first_sample.object_id == 0:
        raise ValueError("The selected H2O sequence starts with background")
    object_name = first_sample.object_name
    for camera in cameras:
        if sequence.load(camera, frame_ids[0]).object_name != object_name:
            raise ValueError("H2O cameras disagree on the first-frame object identity")

    model, mesh = repository.load_mesh(object_name)
    with timings.measure("prepare_networks_and_mesh"):
        runner.prepare([object_name])
    mesh_points = np.asarray(mesh.vertices, dtype=np.float64)
    if len(mesh_points) > 2048:
        mesh_points = mesh_points[np.linspace(0, len(mesh_points) - 1, 2048).astype(int)]
    np.save(output / "mesh_points.npy", mesh_points)

    ground_truth: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    init_masks = output / "initial_masks"
    init_masks.mkdir(parents=True, exist_ok=True)
    with prefetch_rgbd(sequence, frame_ids, cameras,
                       workers=getattr(args, "io_workers", 0),
                       prefetch_frames=getattr(args, "prefetch_frames", 4)) as prefetched:
        for frame_id, rgbd_frames in prefetched:
            reference = sequence.load(cameras[0], frame_id)
            if reference.object_id == 0:
                continue
            if reference.object_name != object_name:
                raise ValueError(f"Object identity changed at H2O frame {frame_id}")
            ground_truth.append(
                {
                    "frame_id": int(frame_id),
                    "t_host": float(frame_id) / args.fps,
                    "object_name": reference.object_name,
                    "pose_world": reference.T_world_object.tolist(),
                }
            )
            for camera in cameras:
                source = f"h2o/{camera}/foundationpose/{object_name}"
                try:
                    sample = sequence.load(camera, frame_id)
                    with timings.measure("rgbd_wait"):
                        frame = rgbd_frames[camera].result()
                    mask = None
                    if frame_id == frame_ids[0]:
                        if args.init_mode == "oracle_gt":
                            mask = _project_mesh_mask(
                                mesh.vertices,
                                sample.T_camera_object,
                                frame.K,
                                frame.color_bgr.shape[:2],
                            )
                        else:
                            if args.mask_root is None:
                                raise ValueError("--mask-root is required with --init-mode mask_dir")
                            mask = _mask_from_directory(args.mask_root.resolve(), camera, frame_id)
                        cv2.imwrite(str(init_masks / f"{camera}_{frame_id:06d}.png"), mask)
                    pose_started = time.perf_counter()
                    estimate = runner.process(
                        source=source,
                        object_name=object_name,
                        color_bgr=frame.color_bgr,
                        depth_m=frame.depth_m,
                        camera_matrix=frame.K,
                        mask=mask,
                    )
                    if estimate is None:
                        raise RuntimeError("FoundationPose returned no estimate")
                    timings.add(estimate.mode, time.perf_counter() - pose_started)
                    quality_metrics = None
                    if getattr(args, "depth_fit_quality", False):
                        quality_started = time.perf_counter()
                        quality_metrics = runner.measure_depth_fit(
                            source=source,
                            object_name=object_name,
                            depth_m=frame.depth_m,
                            camera_matrix=frame.K,
                            pose_cam=estimate.pose_cam,
                        )
                        timings.add("depth_fit", time.perf_counter() - quality_started)
                        if quality_metrics.get("status") != "ok":
                            raise RuntimeError(
                                "FoundationPose depth-fit diagnostic failed: "
                                f"{quality_metrics.get('error', 'unknown error')}"
                            )
                    raw_rows.append(
                        {
                            "frame_id": int(frame_id),
                            "camera_id": camera,
                            "t_host": float(frame_id) / args.fps,
                            "arrival_time": float(frame_id) / args.fps,
                            "observation_kind": "foundationpose_rgbd",
                            "object_name": estimate.object_name,
                            "mode": estimate.mode,
                            "score": float(estimate.score),
                            "score_kind": "legacy_constant_not_calibrated",
                            "scoring": asdict(estimate.scoring) if getattr(estimate, "scoring", None) else None,
                            "quality_metrics": quality_metrics,
                            "source": source,
                            "pose_cam": estimate.pose_cam.tolist(),
                            "T_world_camera": sample.T_world_camera.tolist(),
                            "pose_world": (sample.T_world_camera @ estimate.pose_cam).tolist(),
                        }
                    )
                except Exception as exc:  # keep other sensors running for a real dropout
                    errors.append({"frame_id": int(frame_id), "camera_id": camera, "error": str(exc)})
                    print(f"[WARN] {camera} frame {frame_id}: {exc}", flush=True)
            if frame_id == frame_ids[0] or frame_id == frame_ids[-1] or frame_id % 30 == 0:
                elapsed = time.perf_counter() - take_started
                rate = len(ground_truth) / elapsed if elapsed else 0.0
                print(f"frame {frame_id} ({len(raw_rows)} successful sensor outputs); "
                      f"{elapsed:.1f}s, {rate:.2f} synchronized frames/s", flush=True)

    # Persist costly RGB-D inference before running the optimizer. A fusion
    # failure must not force the user to repeat GPU inference.
    with timings.measure("save_observations"):
        _write_jsonl(output / "ground_truth.jsonl", ground_truth)
        _write_jsonl(output / "tracking_before_fusion.jsonl", raw_rows)
    write_json(output / "summary.json", {
        "status": "inference_saved_fusion_pending", "cameras": cameras,
        "object_name": object_name, "sequence": str(sequence_dir), "init_mode": args.init_mode,
        "fps": args.fps, "n_frames": len(ground_truth), "n_raw_outputs": len(raw_rows), "errors": errors,
    })
    with timings.measure("fusion"), (output / "fusion.log").open("w", encoding="utf-8") as fusion_log, redirect_stdout(fusion_log):
        fused_rows = asyncio.run(_run_fusion(raw_rows, fps=args.fps,
                                            base_overrides=getattr(args, "fusion_overrides", None)))
    with timings.measure("evaluation"):
        raw_metrics = _metrics_by_camera(raw_rows, ground_truth, cameras, mesh_points)
        fused_metrics = summarize(score_poses(ground_truth, fused_rows, mesh_points=mesh_points))
    _write_jsonl(output / "tracking_after_fusion.jsonl", fused_rows)
    performance = timings.report(time.perf_counter() - take_started, len(ground_truth), len(raw_rows))
    performance["io_workers"] = getattr(args, "io_workers", 0)
    performance["prefetch_frames"] = getattr(args, "prefetch_frames", 4)
    write_json(output / "performance.json", performance)
    report = {
        "scope": "H2O RGB-D FoundationPose integration; scores use H2O annotations",
        "status": "completed" if raw_rows and fused_rows else "no_outputs",
        "independent_accuracy": args.init_mode == "mask_dir",
        "oracle_initialization_warning": args.init_mode == "oracle_gt",
        "subject_root": str(subject),
        "sequence": str(sequence_dir),
        "cameras": cameras,
        "object_name": object_name,
        "init_mode": args.init_mode,
        "fps": args.fps,
        "code_sha256": code_fingerprints(),
        "effective_fusion_config": asdict(build_config(VARIANTS["full"], getattr(args, "fusion_overrides", None)).fusion),
        "mesh": {"source": str(args.mesh_source.resolve()), "path": str(model.mesh_path), "scale_to_m": model.scale_to_m},
        "depth_conversion": "H2O uint16 millimetres -> FoundationPose float32 metres",
        "score_policy": "Legacy score=1.0 register/0.9 track; optional scoring contains raw ranking logits with the vendor +100 offset removed, never probabilities or fusion weights",
        "depth_fit_quality": bool(getattr(args, "depth_fit_quality", False)),
        "performance": performance,
        "n_frames": len(ground_truth),
        "n_raw_outputs": len(raw_rows),
        "n_fused_outputs": len(fused_rows),
        "errors": errors,
        "per_camera": raw_metrics,
        "fusion": fused_metrics,
    }
    write_json(output / "summary.json", report)
    lines = [
        "# H2O FoundationPose integration",
        "",
        f"- Sequence: `{sequence_dir}`",
        f"- Object: `{object_name}`",
        f"- Cameras: `{', '.join(cameras)}`",
        f"- Initialization: `{args.init_mode}`",
        f"- Frames: `{len(ground_truth)}`; raw outputs: `{len(raw_rows)}`; fused outputs: `{len(fused_rows)}`",
        "",
        "`oracle_gt` projects the H2O annotated pose into a first-frame mesh mask and is only a pipeline smoke test; use `mask_dir` for an independent tracking evaluation.",
        "",
        "## Per-camera metrics",
        "",
    ]
    for camera, metrics in raw_metrics.items():
        lines.append(
            f"- `{camera}`: coverage={metrics['coverage']:.3f}, "
            f"translation P95={metrics.get('translation_p95_mm')} mm, "
            f"rotation P95={metrics.get('rotation_p95_deg')} deg, "
            f"ADD P95={metrics.get('mesh_add_p95_mm')} mm"
        )
    lines.extend(
        [
            "",
            "## Fusion metrics",
            "",
            f"- coverage={fused_metrics['coverage']:.3f}, translation P95={fused_metrics.get('translation_p95_mm')} mm, rotation P95={fused_metrics.get('rotation_p95_deg')} deg, ADD P95={fused_metrics.get('mesh_add_p95_mm')} mm",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not raw_rows or not fused_rows:
        raise RuntimeError(f"No {'FoundationPose' if not raw_rows else 'fusion'} outputs in {sequence_dir}; see {output / 'summary.json'}")
    print(f"Completed {sequence_dir.relative_to(subject).as_posix()}. Report: {output / 'summary.md'}")
    return report


def _weighted_metric(
    reports: list[dict[str, Any]],
    section: str,
    metric: str,
    weight_key: str,
) -> float | None:
    weighted_sum = 0.0
    total_weight = 0
    for report in reports:
        value = report.get(section, {}).get(metric)
        weight = int(report.get(weight_key, 0))
        if value is None or weight <= 0:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight else None


def _write_all_takes_summary(
    output: Path,
    subject: Path,
    args: argparse.Namespace,
    completed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> None:
    """Write a subject-level index without concatenating independent takes."""
    per_take = []
    for report in completed:
        sequence = Path(report["sequence"])
        key = sequence.relative_to(subject).as_posix()
        per_take.append(
            {
                "take": key,
                "object_name": report.get("object_name"),
                "cameras": report.get("cameras", []),
                "n_frames": report.get("n_frames", 0),
                "n_raw_outputs": report.get("n_raw_outputs", 0),
                "n_fused_outputs": report.get("n_fused_outputs", 0),
                "fusion": report.get("fusion", {}),
                "report": str(output / "takes" / "_".join(sequence.relative_to(subject).parts) / "summary.json"),
            }
        )
    summary = {
        "scope": "All valid H2O RGB-D takes under one subject, each evaluated independently",
        "subject_root": str(subject),
        "output_root": str(output),
        "all_takes": True,
        "init_mode": args.init_mode,
        "oracle_initialization_warning": args.init_mode == "oracle_gt",
        "max_frames_per_take": args.max_frames,
        "n_takes_requested": len(completed) + len(failed),
        "n_takes_completed": len(completed),
        "n_takes_failed": len(failed),
        "failed_takes": failed,
        "per_take": per_take,
        "fusion_weighted_mean": {
            "translation_mean_mm": _weighted_metric(completed, "fusion", "translation_mean_mm", "n_fused_outputs"),
            "rotation_mean_deg": _weighted_metric(completed, "fusion", "rotation_mean_deg", "n_fused_outputs"),
            "mesh_add_mean_mm": _weighted_metric(completed, "fusion", "mesh_add_mean_mm", "n_fused_outputs"),
            "coverage": _weighted_metric(completed, "fusion", "coverage", "n_frames"),
        },
        "aggregation_note": (
            "Each take keeps its own frame/time index and is never concatenated with another take. "
            "The subject-level means are weighted by the number of valid outputs; P50/P95 remain per-take "
            "in each takes/*/summary.json because pooled raw errors are not reconstructed here."
        ),
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# H2O FoundationPose: all takes",
        "",
        f"- Subject: `{subject}`",
        f"- Completed: `{len(completed)}` / `{len(completed) + len(failed)}`",
        f"- Failed: `{len(failed)}`",
        f"- Initialization: `{args.init_mode}`",
        "",
        "Each take is processed independently. Results are under `takes/<group>_<take>/`; "
        "the subject-level report is an index and weighted-mean summary.",
        "",
        "| Take | Object | Frames | Raw outputs | Fused outputs | Fusion translation P95 (mm) | Fusion rotation P95 (deg) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in per_take:
        fusion = item["fusion"]
        lines.append(
            f"| `{item['take']}` | `{item['object_name']}` | {item['n_frames']} | "
            f"{item['n_raw_outputs']} | {item['n_fused_outputs']} | "
            f"{fusion.get('translation_p95_mm')} | {fusion.get('rotation_p95_deg')} |"
        )
    if failed:
        lines.extend(["", "## Failed takes", ""])
        for item in failed:
            lines.append(f"- `{item['take']}`: {item['error']}")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.max_frames < 0 or args.max_takes < 0 or args.fps <= 0:
        raise ValueError("max-frames/max-takes must be nonnegative and fps must be positive")
    if args.take and args.all_takes:
        raise ValueError("Use either --take or --all-takes, not both")
    if args.io_workers < 0 or args.prefetch_frames < 1 or args.torch_threads < 0:
        raise ValueError("worker/thread counts must be nonnegative and prefetch-frames positive")
    subject = args.subject_root.resolve()
    if not subject.is_dir():
        raise ValueError(f"H2O subject root does not exist: {subject}")
    args.fusion_overrides = json.loads(args.fusion_config.read_text(encoding="utf-8")) if args.fusion_config else {}
    build_config(VARIANTS["full"], args.fusion_overrides)
    if args.output:
        output = args.output.resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Output is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = allocate_results(args.output_parent.resolve() if args.output_parent else subject.parent)
    print(f"Results: {output}", flush=True)
    import torch

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    hardware = {"logical_cpus": os.cpu_count(), "torch_cpu_threads": torch.get_num_threads(),
                "cuda_available": torch.cuda.is_available()}
    if hardware["cuda_available"]:
        device = torch.cuda.get_device_properties(0)
        hardware.update(gpu=device.name, gpu_memory_gib=device.total_memory / 2**30)
    print(f"Hardware: {hardware}; RGB-D readers={args.io_workers}, "
          f"prefetch={args.prefetch_frames} frame groups", flush=True)
    write_json(output / "run_manifest.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code_sha256": code_fingerprints(),
        "effective_fusion_config": asdict(build_config(VARIANTS["full"], args.fusion_overrides).fusion),
        "hardware": hardware,
    })

    if args.take:
        relative_take = Path(args.take.replace("\\", "/"))
        sequence_dir = (subject / relative_take).resolve()
        if not sequence_dir.is_dir() or subject not in sequence_dir.parents:
            raise ValueError(f"Take must be below subject root: {args.take}")
        takes = [sequence_dir]
        all_takes_mode = False
    else:
        takes = discover_takes(subject)
        if args.max_takes:
            takes = takes[: args.max_takes]
        if not takes:
            raise ValueError(f"No H2O takes found below {subject}")
        all_takes_mode = True

    repository = H2OModelRepository(
        args.mesh_source,
        extraction_root=args.mesh_cache,
        scale_to_m=1.0,
    )
    runner = FoundationPoseRunner(
        args.foundationpose_root,
        model_repository=repository,
        register_iterations=args.register_iterations,
        track_iterations=args.track_iterations,
        score_tracking=args.score_tracking,
    )

    if not all_takes_mode:
        _run_take(args, subject, takes[0], output, repository, runner)
        return 0

    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    total = len(takes)
    for index, sequence_dir in enumerate(takes, start=1):
        key = sequence_dir.relative_to(subject).as_posix()
        take_output = output / "takes" / "_".join(sequence_dir.relative_to(subject).parts)
        print(f"[TAKE {index}/{total}] {key}", flush=True)
        try:
            report = _run_take(args, subject, sequence_dir, take_output, repository, runner)
            completed.append(report)
        except Exception as exc:
            failed.append({"take": key, "error": str(exc)})
            print(f"[ERROR] {key}: {exc}", flush=True)
    _write_all_takes_summary(output, subject, args, completed, failed)
    print(f"Completed all-take run. Report: {output / 'summary.md'}")
    if failed:
        print(f"Completed with {len(failed)} failed take(s); see summary.json", flush=True)
    return 1 if failed or not completed else 0


if __name__ == "__main__":
    raise SystemExit(main())
