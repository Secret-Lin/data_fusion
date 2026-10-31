"""Evaluate independent pose-fusion baselines on saved H2O tracker observations.

No RGB-D, frontend inference, artificial corruption or GT-guided estimation.
All-subject mode is explicit; the default single-run mode is bounded by callers.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import traceback

import numpy as np

from experiments.fusion_baselines import METHODS, BaselineConfig, BaselineRunError, canonical_groups, run_baseline
from experiments.h2o.ablation import FusionRunError, VARIANTS, build_config, read_gzip, run_baselines, run_fusion, save_scores
from experiments.h2o.evaluation import (
    MetricAccumulator, allocate_results, bootstrap_mean_ci, fingerprint, metric_table,
    paired_delta, score_poses, summarize, write_json,
)
from experiments.h2o.foundationpose_results import discover_result_takes, load_result_take
from tools.run_h2o_foundationpose_ablation import code_fingerprints


SOURCES = {"foundationpose": "results_all_subjects", "m3t": "m3t_all_subjects", "megapose": "megapose_all_subjects"}


def source_hashes() -> dict[str, str]:
    output = code_fingerprints()
    root = Path(__file__).resolve().parents[1]
    for path in ("experiments/fusion_baselines.py", "tools/run_h2o_fusion_baselines.py",
                 "experiments/configs/fusion_baselines_v1.json", "tools/run_h2o_fusion_baselines.ps1"):
        output[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
    return output


def planned_jobs(args: argparse.Namespace) -> list[dict]:
    roots = []
    if args.run_dir:
        roots.append((args.frontend, args.run_dir.resolve().parent.name, args.run_dir.resolve()))
    else:
        for frontend in args.frontends:
            for subject in args.subjects:
                directory = args.data_root.resolve() / SOURCES[frontend] / f"subject{subject}" / args.run_name
                summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                if (summary.get("n_takes_requested", 0) <= 0 or summary.get("n_takes_failed") != 0
                        or summary.get("n_takes_completed") != summary["n_takes_requested"]):
                    raise ValueError(f"Incomplete frontend cache: {directory}")
                discovered = discover_result_takes(directory)
                if len(discovered) != summary["n_takes_requested"]:
                    raise ValueError(f"Cache take count mismatch: {directory}")
                roots.append((frontend, f"subject{subject}", directory))
    jobs = []
    for frontend, subject, directory in roots:
        takes = discover_result_takes(directory, args.takes)
        if args.max_takes:
            takes = takes[:args.max_takes]
        for name, path in takes:
            for required in ("ground_truth.jsonl", "tracking_before_fusion.jsonl", "summary.json"):
                if not (path / required).is_file():
                    raise FileNotFoundError(path / required)
            jobs.append({"frontend": frontend, "subject": subject, "take": name, "input": str(path),
                         "relative_output": f"{frontend}/{subject}/takes/{name}"})
    return jobs


def common_scores(scored: dict[str, list[dict]]) -> dict[str, list[dict]]:
    shared = set.intersection(*[{r["frame_id"] for r in rows if r["present"]} for rows in scored.values()])
    return {name: [r for r in rows if r["frame_id"] in shared] for name, rows in scored.items()}


def run_take(job: dict, output: str, settings: dict, max_frames: int, initial_code: dict) -> dict:
    directory = Path(output) / job["relative_output"]
    directory.mkdir(parents=True, exist_ok=False)
    path = Path(job["input"])
    gt, raw, cameras, points, metadata = load_result_take(path, max_frames)
    if not gt:
        raise ValueError(f"No target frames: {path}")
    cfg = BaselineConfig(**settings)
    noise = build_config(VARIANTS["full"]).fusion
    # One final observation per physical camera/group for EVERY estimator,
    # including production and arithmetic baselines. No GT values enter here.
    groups = canonical_groups(raw, cfg.max_group_span_s)
    observations = [row for group in groups for row in group]
    by_frame = {}
    for row in observations:
        old = by_frame.setdefault(row["frame_id"], row["object_name"])
        if old != row["object_name"]:
            raise ValueError("H2O scorer requires one annotated object per frame")
    observation_hash = fingerprint(observations)
    # Constant/zero frontend scores are not calibrated probabilities. For the
    # legacy score-weighted comparator, an all-zero group gets uniform weights.
    # This does NOT change observations supplied to any new estimator or ours.
    score_rows = []
    zero_score_groups = 0
    for group in groups:
        if sum(float(r["score"]) for r in group) <= 0:
            zero_score_groups += 1
            score_rows.extend({**r, "score": 1.} for r in group)
        else:
            score_rows.extend(group)
    base_predictions = run_baselines(score_rows, cameras)
    names = list(base_predictions) + list(METHODS) + ["ours"]
    methods, scored = {}, {}
    for name in names:
        predictions, runtime, failure = [], {}, None
        with (directory / f"{name}.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            try:
                if name in base_predictions:
                    predictions = base_predictions[name]
                elif name == "ours":
                    predictions, runtime = asyncio.run(run_fusion(observations, VARIANTS["full"], fps=30))
                else:
                    predictions, runtime = run_baseline(observations, name, cfg, noise)
                rows = score_poses(gt, predictions, mesh_points=points)
            except Exception as exc:
                traceback.print_exc(file=log)
                failure = {"error": str(exc), "exception": type(exc).__name__}
                if isinstance(exc, (BaselineRunError, FusionRunError)):
                    predictions = exc.predictions
                    failure.update(exc.failure)
                else:
                    predictions = []
                    failure["policy"] = "all targets missing; no recoverable method prefix"
                rows = score_poses(gt, predictions, mesh_points=points)
        save_scores(directory / name, predictions, rows)
        scored[name] = rows
        methods[name] = {"status": "failed" if failure else "completed", "metrics": summarize(rows),
                         "failure": failure, "runtime": runtime,
                         "prediction_only_outputs": sum(bool(r.get("prediction_only")) for r in predictions),
                         "batch_iteration_limit_outputs": sum(bool(r.get("diagnostics", {}).get("hit_iteration_limit")) for r in predictions)}
    common = common_scores(scored)
    report = {**job, "n_targets": len(gt), "n_raw": len(raw), "n_observations": len(observations),
              "n_duplicate_replaced": len(raw)-len(observations), "observations_sha256": observation_hash,
              "zero_score_groups_uniform_simple_baseline": zero_score_groups,
              "targets_sha256": fingerprint(gt), "baseline_config": settings,
              "production_config": asdict(noise), "methods": methods,
              "common_output_metrics": {name: summarize(rows) for name, rows in common.items()},
              "paired_ours_minus": {name: paired_delta(scored["ours"], rows) for name, rows in scored.items() if name != "ours"},
              "initialization": {k: metadata.get(k) for k in ("init_mode", "initialization", "frontend_sha256")},
              "code_changed_during_run": source_hashes() != initial_code,
              "inputs_mutated": observation_hash != fingerprint(observations),
              "status": "failed_methods" if any(v["failure"] for v in methods.values()) else "completed"}
    write_json(directory / "summary.json", report)
    return report


def summarize_reports(output: Path, reports: list[dict], failures: list[dict], manifest: dict) -> dict:
    pooled = defaultdict(lambda: defaultdict(MetricAccumulator))
    matched = defaultdict(lambda: defaultdict(MetricAccumulator))
    take_deltas = defaultdict(lambda: defaultdict(list))
    for report in reports:
        directory = output / report["relative_output"]
        scores = {name: read_gzip(directory / name / "scores.jsonl.gz") for name in report["methods"]}
        common = common_scores(scores)
        for name, rows in scores.items():
            pooled[report["frontend"]][name].add(rows)
            matched[report["frontend"]][name].add(common[name])
        for name, pair in report["paired_ours_minus"].items():
            take_deltas[report["frontend"]][name].append({"subject": report["subject"], "take": report["take"], **pair})
    paired = {}
    for frontend, methods in take_deltas.items():
        paired[frontend] = {}
        for name, deltas in methods.items():
            fields = {}
            for metric in ("mean_translation_delta_mm", "mean_rotation_delta_deg"):
                values = [d[metric] for d in deltas if d[metric] is not None]
                fields[metric] = {"n_paired_takes": len(values), "mean": float(np.mean(values)) if values else None,
                                  "ci95_take_bootstrap": bootstrap_mean_ci(values),
                                  "ours_better": sum(v < 0 for v in values),
                                  "ours_worse": sum(v > 0 for v in values), "equal": sum(v == 0 for v in values)}
            paired[frontend][name] = {"summary": fields, "takes": deltas}
    result = {**manifest, "n_completed_takes": len(reports), "failed_takes": failures,
              "failed_methods": [{"frontend": r["frontend"], "subject": r["subject"], "take": r["take"],
                                   "method": name, **m["failure"]}
                                  for r in reports for name, m in r["methods"].items() if m["failure"]],
              "methods_by_frontend": {f: {n: v.result() for n, v in methods.items()} for f, methods in pooled.items()},
              "common_output_metrics_by_frontend": {f: {n: v.result() for n, v in methods.items()} for f, methods in matched.items()},
              "paired_ours_minus": paired,
              "code_changed_during_run": source_hashes() != manifest["code_sha256"] or any(r["code_changed_during_run"] for r in reports),
              "inputs_mutated": any(r["inputs_mutated"] for r in reports)}
    result["status"] = "completed" if not (failures or result["failed_methods"] or result["code_changed_during_run"] or result["inputs_mutated"]) else "incomplete_or_failed"
    write_json(output / "summary.json", result)
    lines = ["# Independent cached-pose fusion baselines", "", f"Status: {result['status']}", "",
             "No frontend inference, artificial corruption or GT-driven selection. Existing initialization is inherited.",
             "Failure = >50 mm OR >10 degrees OR missing. Means/quantiles below condition on available output; coverage is always shown.",
             "Common-output and take-paired descriptive bootstrap results are in summary.json. Missing comparisons are not wins.",
             "These are standard-method implementations, not official published tracker benchmark reproductions. No live-latency claim.", ""]
    for frontend, methods in result["methods_by_frontend"].items():
        lines += [f"## {frontend}", "", metric_table(methods), ""]
    lines += [f"Failed takes: {len(failures)}; failed methods: {len(result['failed_methods'])}.",
              "Do not interpret lower conditional error after a method failure as an advantage."]
    (output / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=Path)
    mode.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--frontend", choices=list(SOURCES), default="foundationpose", help="Label for --run-dir")
    parser.add_argument("--data-root", type=Path, default=Path("data/h2o"))
    parser.add_argument("--run-name", default="results_0")
    parser.add_argument("--frontends", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--subjects", nargs="+", type=int, choices=range(1, 5), default=[1, 2, 3, 4])
    parser.add_argument("--takes", default="")
    parser.add_argument("--max-takes", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/fusion_baselines_v1.json"))
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts/independent_fusion_baselines"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.max_takes, args.max_frames) < 0 or not 1 <= args.workers <= 16:
        parser.error("Limits must be nonnegative and workers in [1,16]")
    if not re.fullmatch(r"results_\d+", args.run_name):
        parser.error("run-name must be results_N")
    args.frontends, args.subjects = list(dict.fromkeys(args.frontends)), list(dict.fromkeys(args.subjects))
    settings = asdict(BaselineConfig(**json.loads(args.config.read_text(encoding="utf-8"))))
    jobs = planned_jobs(args)
    print(f"Validated {len(jobs)} takes; max_frames={args.max_frames}; workers={args.workers}", flush=True)
    if args.dry_run:
        print("Dry run: no fusion, inference or result directories created.")
        return
    output = allocate_results(args.output_parent.resolve())
    manifest = {"protocol": "independent_pose_fusion_v1", "jobs": jobs, "n_requested_takes": len(jobs),
                "max_frames": args.max_frames, "workers": args.workers, "baseline_config": settings,
                "code_sha256": source_hashes(), "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "artificial_corruption": False, "frontend_rerun": False,
                "parameter_policy": "Fixed engineering settings; not calibrated on test GT; selected development data is not untouched test data",
                "output_semantics": "Final capture-group output; no claims about provisional outputs or live delay",
                "gap_policy": "No automatic outputs at entirely missing groups; ESKF can predict at observed groups when all gates reject up to configured max_prediction_gap_s"}
    write_json(output / "manifest.json", manifest)
    print(f"Results: {output}", flush=True)
    reports, failures = [], []
    def collect(job, report=None, error=None):
        if error is None:
            reports.append(report)
            status = report["status"]
        else:
            failures.append({**job, "error": str(error), "traceback": "".join(traceback.format_exception(error))})
            status = "input_or_scoring_failure"
        print(f"[{len(reports)+len(failures)}/{len(jobs)}] {job['relative_output']}: {status}", flush=True)
        write_json(output / "progress.json", {"n_completed": len(reports), "failed_takes": failures})
    if args.workers == 1:
        for job in jobs:
            try:
                collect(job, run_take(job, str(output), settings, args.max_frames, manifest["code_sha256"]))
            except Exception as exc:
                collect(job, error=exc)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_jobs = {executor.submit(run_take, job, str(output), settings, args.max_frames, manifest["code_sha256"]): job for job in jobs}
            for future in as_completed(future_jobs):
                job = future_jobs[future]
                try:
                    collect(job, future.result())
                except Exception as exc:
                    collect(job, error=exc)
    reports.sort(key=lambda r: r["relative_output"])
    report = summarize_reports(output, reports, failures, manifest)
    print(f"Report: {output / 'summary.md'}", flush=True)
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
