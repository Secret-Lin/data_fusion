"""Read and summarize single-take or subject-level FoundationPose results."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.h2o.ablation import run_baselines
from experiments.h2o.evaluation import MetricAccumulator, read_jsonl, score_poses, summarize, write_json


def discover_result_takes(root: Path, takes: str = "") -> list[tuple[str, Path]]:
    if (root / "ground_truth.jsonl").is_file():
        found = [(root.name, root)]
    else:
        found = [(p.parent.name, p.parent) for p in sorted(root.glob("takes/*/ground_truth.jsonl"))]
    requested = {s.strip().replace("/", "_").replace("\\", "_") for s in takes.split(",") if s.strip()}
    if requested:
        missing = requested - {key for key, _ in found}
        if missing:
            raise ValueError(f"Unknown result takes: {sorted(missing)}")
        found = [(key, path) for key, path in found if key in requested]
    if not found:
        raise ValueError(f"No FoundationPose results below {root}")
    return found


def load_result_take(path: Path, max_frames: int = 0) -> tuple[list[dict], list[dict], list[str], np.ndarray | None, dict]:
    gt = read_jsonl(path / "ground_truth.jsonl")
    if max_frames:
        gt = gt[:max_frames]
    ids = {r["frame_id"] for r in gt}
    raw = [r for r in read_jsonl(path / "tracking_before_fusion.jsonl") if r["frame_id"] in ids]
    for row in raw:
        row.setdefault("arrival_time", row["t_host"])
    metadata = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    cameras = sorted(set(metadata.get("cameras", [])) | {r["camera_id"] for r in raw})
    points_path = path / "mesh_points.npy"
    points = np.load(points_path) if points_path.is_file() else None
    return gt, raw, cameras, points, metadata


def analyze_results(root: Path, output: Path) -> dict:
    pooled = defaultdict(MetricAccumulator)
    by_object = defaultdict(lambda: defaultdict(MetricAccumulator))
    per_take = []
    error_counts = Counter()
    for key, path in discover_result_takes(root):
        gt, raw, cameras, points, metadata = load_result_take(path)
        methods = run_baselines(raw, cameras)
        methods["stored_fusion"] = read_jsonl(path / "tracking_after_fusion.jsonl")
        metrics = {}
        for name, predictions in methods.items():
            scores = score_poses(gt, predictions, mesh_points=points)
            pooled[name].add(scores)
            by_object[metadata["object_name"]][name].add(scores)
            metrics[name] = summarize(scores)
        errors = metadata.get("errors", [])
        error_counts.update(e["error"] for e in errors)
        per_take.append({"take": key, "object_name": metadata["object_name"],
                         "n_frames": len(gt), "n_observations": len(raw), "n_errors": len(errors),
                         "status": "no_observations" if not raw else "has_outputs",
                         "methods": metrics})
    report = {
        "input": str(root), "n_takes": len(per_take),
        "n_takes_with_observations": sum(p["n_observations"] > 0 for p in per_take),
        "n_frames": sum(p["n_frames"] for p in per_take),
        "n_observations": sum(p["n_observations"] for p in per_take),
        "errors": dict(error_counts), "methods": {k: v.result() for k, v in pooled.items()},
        "per_object": {o: {k: v.result() for k, v in methods.items()} for o, methods in by_object.items()},
        "per_take": per_take,
        "evaluation_note": "Pooled frame errors, including empty takes as missing; GT used only for scoring. "
                           "Initialization protocol is inherited from the source run (oracle_gt is GT-mask-assisted). "
                           "Rotation and ADD do not quotient object symmetries. best_score ties select first camera.",
    }
    write_json(output / "summary.json", report)
    lines = ["# FoundationPose result analysis", "", f"Input: `{root}`", "",
             f"Takes with observations: {report['n_takes_with_observations']}/{report['n_takes']}; "
             f"target frames: {report['n_frames']}; raw observations: {report['n_observations']}.", "",
             report["evaluation_note"], "",
             "| Method | Coverage | Translation mean / P50 / P95 (mm) | Rotation P95 (deg) | ADD P95 (mm) | Failure rate (5 cm / 10 deg) |",
             "|---|---:|---:|---:|---:|---:|"]
    def fmt(x):
        return "—" if x is None else f"{x:.3f}"
    for name, m in report["methods"].items():
        t = " / ".join(fmt(m[f"translation_{s}_mm"]) for s in ("mean", "p50", "p95"))
        lines.append(f"| {name} | {fmt(m['coverage'])} | {t} | {fmt(m['rotation_p95_deg'])} | "
                     f"{fmt(m['mesh_add_p95_mm'])} | {fmt(m['failure_rate_including_missing'])} |")
    lines += ["", "## Empty takes", ""]
    lines += [f"- {p['take']} ({p['object_name']}): {p['n_errors']} errors" for p in per_take if not p["n_observations"]]
    lines += ["", "## Error causes", ""] + [f"- {count}: {error}" for error, count in error_counts.items()]
    lines += ["", "## Worst stored-fusion translation P95", ""]
    for p in sorted((p for p in per_take if p["methods"]["stored_fusion"]["n"]),
                    key=lambda p: p["methods"]["stored_fusion"]["translation_p95_mm"], reverse=True)[:8]:
        m = p["methods"]["stored_fusion"]
        lines.append(f"- {p['take']} ({p['object_name']}): {m['translation_p95_mm']:.2f} mm; "
                     f"{m['observations_per_output_mean']:.2f} accepted sensors/frame")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
