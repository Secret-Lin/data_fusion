"""Run current fusion and ablations on cached H2O frontend observations.

Accepts either a single take or a subject result root. No artificial corruption
or RGB-D inference: every method receives the same saved observations.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from experiments.h2o.ablation import DEFAULT_VARIANTS, VARIANTS, FusionRunError, Variant, build_config, run_baselines, run_fusion, save_scores
from experiments.h2o.evaluation import MetricAccumulator, allocate_results, paired_delta, read_jsonl, score_poses, summarize, write_json
from experiments.h2o.foundationpose_results import discover_result_takes, load_result_take


def code_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = ("managers/fusion_manager.py", "config.py", "core/types.py", "experiments/h2o/ablation.py",
             "tools/run_h2o_foundationpose_ablation.py", "tools/run_h2o_foundationpose.py",
             "adapters/foundationpose_runner.py", "adapters/h2o_foundationpose.py", "core/pose_consensus.py",
             "experiments/h2o/dataset.py", "experiments/h2o/prefetch.py", "experiments/h2o/performance.py",
             "experiments/h2o/evaluation.py", "experiments/h2o/foundationpose_results.py",
             "tools/score_h2o_cached_poses.py",
             "tools/rerank_h2o_cached_poses.py",
             "thirdparty/FoundationPose/Utils.py",
             "thirdparty/FoundationPose/learning/training/predict_score.py")
    # Cached fusion can run without installing any tracking frontend. Keep
    # missing optional vendor sources explicit in provenance, not as fake hashes.
    return {
        name: "not_installed" if name.startswith("thirdparty/") and not (root / name).is_file()
        else hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in paths
    }


def compare_pose_outputs(first: list[dict], second: list[dict]) -> dict:
    """Check actual paired outputs, not just equality of aggregate errors."""
    keys = lambda rows: [(r["frame_id"], r.get("object_name")) for r in rows]
    same_keys = keys(first) == keys(second)
    report = {"first_count": len(first), "second_count": len(second),
              "same_ordered_keys": same_keys, "max_pose_matrix_delta": None,
              "max_covariance_delta": None, "active_sensor_changed_frames": None,
              "pose_and_covariance_identical": False}
    if not same_keys or not first:
        return report
    a, b = [np.asarray([r["pose_world"] for r in rows]) for rows in (first, second)]
    report["max_pose_matrix_delta"] = float(np.max(np.abs(a - b)))
    report["active_sensor_changed_frames"] = sum(
        x.get("active_sensors") != y.get("active_sensors") for x, y in zip(first, second))
    covariances = [r.get("covariance") for r in first + second]
    if all(c is not None for c in covariances):
        ca, cb = [np.asarray([r["covariance"] for r in rows]) for rows in (first, second)]
        report["max_covariance_delta"] = float(np.max(np.abs(ca - cb)))
        report["pose_and_covariance_identical"] = bool(np.array_equal(a, b) and np.array_equal(ca, cb))
    return report


def output_diagnostics(predictions: dict[str, list[dict]]) -> dict:
    pairs = (("full", "stored_fusion"), ("full", "reverse_order"),
             ("revision_checkpoint", "revision_checkpoint_reverse"))
    return {f"{a}_vs_{b}": compare_pose_outputs(predictions[a], predictions[b])
            for a, b in pairs if a in predictions and b in predictions}


def _write_report(output: Path, report: dict) -> None:
    write_json(output / "summary.json", report)
    lines = ["# H2O cached-observation fusion comparison", "", f"Input: `{report['input']}`", "",
             "No artificial noise, dropout or delay. GT is only used by the evaluator. "
             "Initialization protocol is inherited from the input run.", "",
             "Errors are pooled over target frames; empty observations count towards coverage and failure rate.", "",
             "Failure = translation > 50 mm OR rotation > 10 deg OR missing. "
             "Pose errors describe available outputs, not missing frames.", "",
             "| Method | Coverage | Translation mean (mm) | P50 (mm) | P95 (mm) | Rotation P95 (deg) | ADD P95 (mm) | Failure (%) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(x):
        return "—" if x is None else f"{x:.3f}"
    for name, item in report["methods"].items():
        if "metrics" not in item:
            lines.append(f"| {name} | {item.get('status')} | — | — | — | — | — | — |")
            continue
        m = item["metrics"]
        cols = [m.get(k) for k in ("coverage", "translation_mean_mm", "translation_p50_mm",
                                  "translation_p95_mm", "rotation_p95_deg", "mesh_add_p95_mm")]
        rate = m.get("failure_rate_including_missing")
        cols.append(None if rate is None else rate * 100)
        label = name if item.get("status") == "completed" else f"{name} [{item.get('status')}]"
        lines.append(f"| {label} | " + " | ".join(fmt(v) for v in cols) + " |")
    if report.get("failed_methods"):
        lines += ["", "## Failed variants (included in coverage/failure denominators)",
                  "", "Completed prefix outputs are retained; failed frame and remaining targets are missing. "
                  "Unexpected exceptions without a recoverable prefix count all targets as missing."]
        lines += [f"- {r.get('take', '')}/{r['method']}: {r['error']}" for r in report["failed_methods"]]
    if "failed_takes" in report:
        lines += ["", f"Takes: {report['n_takes']}; empty input takes: {report['n_empty_takes']}; "
                  f"failed computations: {len(report['failed_takes'])}."]
        lines += [f"- {r['take']}: {r['error']}" for r in report["failed_takes"]]
    if report.get("output_diagnostics"):
        lines += ["", "## Paired output diagnostics", ""]
        for name, check in report["output_diagnostics"].items():
            lines.append(f"- {name}: {check}")
    if report.get("code_changed_during_run"):
        lines += ["", "WARNING: source files changed during this run; do not treat it as a frozen-code experiment."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _run_take(args: argparse.Namespace, run_dir: Path, output: Path,
                    base_overrides: dict) -> tuple[dict, dict[str, list[dict]]]:
    gt, observations, cameras, mesh_points, metadata = load_result_take(run_dir, args.max_frames)
    if not gt:
        raise ValueError(f"Empty ground truth: {run_dir}")
    quality_sidecar = getattr(args, "quality_sidecar", None)
    if quality_sidecar:
        sidecar_rows = read_jsonl(Path(quality_sidecar).resolve())
        by_key = {
            (int(row["frame_id"]), str(row["camera_id"]), str(row.get("object_name", ""))): row
            for row in sidecar_rows
        }
        enriched = []
        for row in observations:
            copy = dict(row)
            fit = by_key.get((int(row["frame_id"]), str(row["camera_id"]),
                              str(row.get("object_name", ""))))
            if fit is not None and isinstance(fit.get("depth_fit"), dict):
                metrics = dict(copy.get("quality_metrics") or {})
                metrics.update({
                    key: value for key, value in fit["depth_fit"].items()
                    if key != "status" and value is not None
                })
                if fit["depth_fit"].get("depth_inlier_fraction") is not None:
                    metrics["q_fit"] = float(fit["depth_fit"]["depth_inlier_fraction"])
                copy["quality_metrics"] = metrics
            enriched.append(copy)
        observations = enriched
    variants = [VARIANTS[name] for name in dict.fromkeys(args.variants.split(","))]
    if args.leave_one_out:
        variants += [Variant(f"without_{camera}", f"Exclude {camera}", exclude_camera=camera) for camera in cameras]
    output.mkdir(parents=True, exist_ok=True)
    methods, scored, failed_methods = {}, {}, []
    predictions_by_method = run_baselines(observations, cameras)
    target_ids = {r["frame_id"] for r in gt}
    stored = run_dir / "tracking_after_fusion.jsonl"
    if stored.is_file():
        predictions_by_method["stored_fusion"] = [r for r in read_jsonl(stored) if r["frame_id"] in target_ids]
    for name, predictions in predictions_by_method.items():
        rows = score_poses(gt, predictions, mesh_points=mesh_points)
        save_scores(output / name, predictions, rows)
        scored[name] = rows
        methods[name] = {"status": "completed", "metrics": summarize(rows)}
    for variant in variants:
        if variant.inactive_reason:
            methods[variant.name] = {"status": "not_applicable", "reason": variant.inactive_reason}
            continue
        effective_config = asdict(build_config(variant, base_overrides).fusion)
        status = "completed"
        with (output / f"{variant.name}.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            try:
                predictions, runtime = await run_fusion(observations, variant, fps=args.fps, base_overrides=base_overrides)
            except Exception as exc:
                traceback.print_exc(file=log)
                status = "failed"
                predictions = exc.predictions if isinstance(exc, FusionRunError) else []
                failure = {"method": variant.name, "error": str(exc), "exception_type": type(exc).__name__,
                           "log": str(output / f"{variant.name}.log")}
                if isinstance(exc, FusionRunError):
                    failure.update(exc.failure)
                else:
                    failure["policy"] = "stop_variant; all_targets_missing_without_recoverable_prefix"
                failed_methods.append(failure)
                runtime = {"failure": failure}
        rows = score_poses(gt, predictions, mesh_points=mesh_points)
        save_scores(output / variant.name, predictions, rows)
        scored[variant.name] = rows
        methods[variant.name] = {"status": status, "variant": asdict(variant),
                                 "effective_fusion_config": effective_config,
                                 "metrics": summarize(rows), "runtime": runtime}
        if variant.name in {"full", "reverse_order", "revision_checkpoint", "revision_checkpoint_reverse"}:
            predictions_by_method[variant.name] = predictions
        print(f"  {variant.name} [{status}]: {len(predictions)}/{len(gt)} outputs; "
              f"P95={methods[variant.name]['metrics']['translation_p95_mm']} mm", flush=True)
    final_code = code_fingerprints()
    initial_code = getattr(args, "code_sha256", final_code)
    report = {"input": str(run_dir), "cameras": cameras, "object_name": gt[0]["object_name"],
              "n_ground_truth": len(gt), "n_observations": len(observations), "fps": args.fps,
              "status": "completed_with_failures" if failed_methods else ("no_observations" if not observations else "completed"),
              "failed_methods": failed_methods,
              "fusion_protocol": "h2o_checkpoint_default_v2",
              "init_mode": metadata.get("init_mode"),
              "frontend": metadata.get("frontend", "foundationpose"),
              "source_initialization": metadata.get("initialization"),
              "source_frontend_sha256": metadata.get("frontend_sha256"),
              "code_sha256": initial_code,
              "code_changed_during_run": initial_code != final_code,
              "artificial_noise": False, "artificial_dropout": False, "artificial_delay": False,
              "quality_sidecar": str(Path(quality_sidecar).resolve()) if quality_sidecar else None,
              "methods": methods,
              "output_diagnostics": output_diagnostics(predictions_by_method),
              "paired_full_minus": {name: paired_delta(scored["full"], rows) for name, rows in scored.items() if name != "full"}}
    _write_report(output, report)
    return report, scored


async def _run(args: argparse.Namespace) -> dict:
    args.code_sha256 = code_fingerprints()
    root = args.run_dir.resolve()
    takes = discover_result_takes(root, args.takes)
    if args.max_takes:
        takes = takes[:args.max_takes]
    if getattr(args, "quality_sidecar", None) and len(takes) != 1:
        raise ValueError("--quality-sidecar currently requires exactly one selected result take")
    base_overrides = json.loads(args.fusion_config.read_text(encoding="utf-8")) if args.fusion_config else {}
    build_config(VARIANTS["full"], base_overrides)
    if args.output:
        output = args.output.resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Output is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = allocate_results(args.output_parent.resolve() if args.output_parent else root.parent)
    args.output = output
    print(f"Results: {output}", flush=True)
    if (root / "ground_truth.jsonl").is_file():
        report, _ = await _run_take(args, root, output, base_overrides)
        return report
    pooled = defaultdict(MetricAccumulator)
    inactive, completed, failed, failed_methods = {}, [], [], []
    for index, (key, path) in enumerate(takes, 1):
        print(f"[TAKE {index}/{len(takes)}] {key}", flush=True)
        try:
            report, scored = await _run_take(args, path, output / "takes" / key, base_overrides)
            for name, rows in scored.items():
                pooled[name].add(rows)
            inactive.update({name: item for name, item in report["methods"].items() if item["status"] == "not_applicable"})
            failed_methods.extend({"take": key, **item} for item in report.get("failed_methods", []))
            completed.append({"take": key, "status": report["status"], "n_ground_truth": report["n_ground_truth"],
                              "n_observations": report["n_observations"],
                              "output_diagnostics": report["output_diagnostics"],
                              "report": str(output / "takes" / key / "summary.json")})
        except Exception as exc:
            failed.append({"take": key, "error": str(exc)})
            print(f"[ERROR] {key}: {exc}", flush=True)
    report = {"input": str(root), "n_takes": len(takes), "n_empty_takes": sum(r["status"] == "no_observations" for r in completed),
              "failed_takes": failed, "failed_methods": failed_methods,
              "fusion_protocol": "h2o_checkpoint_default_v2",
              "per_take": completed, "max_frames_per_take": args.max_frames,
              "code_sha256": args.code_sha256,
              "code_changed_during_run": args.code_sha256 != code_fingerprints(),
              "aggregation_note": "Pooled exact frame errors including failed-variant missing suffixes. "
                                  "Input/scoring failures are separate failed_takes and make the report incomplete. "
                                  "Any failure causes nonzero exit after all variants/takes are attempted.",
              "methods": {**{name: {"status": "completed_with_failures" if any(
                  r['method'] == name for r in failed_methods) else "completed",
                  "n_failed_takes": sum(r['method'] == name for r in failed_methods),
                  "metrics": pool.result()} for name, pool in pooled.items()}, **inactive}}
    _write_report(output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", type=Path)
    output_group.add_argument("--output-parent", type=Path)
    parser.add_argument("--takes", default="", help="Comma-separated h1/0,k1/5; default all stored takes")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-takes", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--fusion-config", type=Path, help="JSON FusionConfig overrides; ablation overrides apply last")
    parser.add_argument("--quality-sidecar", type=Path,
                        help="Optional cached score_h2o --depth-fit scores.jsonl; adds q_fit only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0 or args.max_frames < 0 or args.max_takes < 0:
        raise ValueError("fps must be positive; frame/take limits must be nonnegative")
    names = set(args.variants.split(","))
    if "full" not in names or names - set(VARIANTS):
        raise ValueError(f"variants must include full and be selected from {list(VARIANTS)}")
    report = asyncio.run(_run(args))
    print(f"Report: {args.output / 'summary.md'}")
    return 1 if report.get("failed_takes") or report.get("failed_methods") else 0


if __name__ == "__main__":
    raise SystemExit(main())
