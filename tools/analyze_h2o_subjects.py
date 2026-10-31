"""Evaluate saved subject runs without invoking tracking or the fusion solver."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from experiments.h2o.ablation import run_baselines, write_gzip
from experiments.h2o.evaluation import MetricAccumulator, allocate_results, read_jsonl, score_poses, summarize, write_json
from experiments.h2o.foundationpose_results import discover_result_takes, load_result_take
from tools.audit_h2o_fusion import COMPARISON_METRICS, paired_take_summary, verify_metrics
from tools.run_h2o_foundationpose_ablation import code_fingerprints


def paired_comparisons(takes: list[dict]) -> dict:
    """Pair within subject/take; never join frame ids across independent takes."""
    names = sorted(set.intersection(*(set(take["methods"]) for take in takes)))
    result = {}
    for name in names:
        if name == "stored_fusion":
            continue
        result[name] = {}
        for metric in COMPARISON_METRICS:
            pairs = [(t["methods"]["stored_fusion"][metric], t["methods"][name][metric]) for t in takes]
            valid = [(a, b) for a, b in pairs if a is not None and b is not None]
            if valid:
                result[name][metric] = {
                    **paired_take_summary([a for a, _ in valid], [b for _, b in valid]),
                    "takes_without_metric": len(pairs) - len(valid),
                }
    return result


def checked_take_index(root: Path, summary: dict) -> list[tuple[str, Path]]:
    """Do not silently omit failed/missing takes while evaluating a completed run."""
    found = discover_result_takes(root)
    expected = {r["take"].replace("/", "_") for r in summary["per_take"]}
    if (summary["n_takes_failed"] or summary.get("failed_takes")
            or summary["n_takes_completed"] != summary["n_takes_requested"]
            or len(expected) != summary["n_takes_requested"]
            or len(summary["per_take"]) != len(expected)
            or {name for name, _ in found} != expected):
        raise ValueError(f"Incomplete/failed/duplicate take index: {root}; inspect source summary")
    return found


def analyze(root: Path, run_name: str, subjects: list[int], output: Path) -> dict:
    pools = defaultdict(lambda: defaultdict(MetricAccumulator))
    object_pools = defaultdict(lambda: defaultdict(MetricAccumulator))
    records, inputs = [], {}
    support = defaultdict(Counter)
    reference_code, reference_config, reference_arguments = None, None, None
    current_code = code_fingerprints()
    for number in subjects:
        subject = f"subject{number}"
        run = root / subject / run_name
        source = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        arguments = {k: v for k, v in manifest["arguments"].items()
                     if k not in ("subject_root", "output", "output_parent")}
        if reference_code is None:
            reference_code = manifest["code_sha256"]
            reference_config = manifest["effective_fusion_config"]
            reference_arguments = arguments
        if (manifest["code_sha256"] != reference_code
                or manifest["effective_fusion_config"] != reference_config
                or arguments != reference_arguments):
            raise ValueError(f"Subjects were run with different source/configuration: {subject}")
        inputs[subject] = {"run_dir": str(run), "summary_sha256": hashlib.sha256((run / "summary.json").read_bytes()).hexdigest()}
        takes = checked_take_index(run, source)
        for index, (key, path) in enumerate(takes, 1):
            gt, raw, cameras, points, metadata = load_result_take(path)
            if (metadata["code_sha256"] != reference_code
                    or metadata["effective_fusion_config"] != reference_config):
                raise ValueError(f"Mixed source/configuration within run: {subject}/{key}")
            fused = read_jsonl(path / "tracking_after_fusion.jsonl")
            if (len(gt) != metadata["n_frames"] or len(raw) != metadata["n_raw_outputs"]
                    or len(fused) != metadata["n_fused_outputs"] or not gt or points is None):
                raise ValueError(f"Saved result count/mesh mismatch: {path}")
            groups = ["all_subjects", subject, "development_subject1" if number == 1 else "new_subjects2_4"]
            methods = run_baselines(raw, cameras)
            methods["stored_fusion"] = fused
            item = {"subject": subject, "take": key, "object_name": metadata["object_name"],
                    "n_frames": len(gt), "n_observations": len(raw), "errors": metadata.get("errors", []), "methods": {}}
            directory = output / "takes" / subject / key
            directory.mkdir(parents=True)
            for name, predictions in methods.items():
                scores = score_poses(gt, predictions, mesh_points=points)
                metrics = summarize(scores)
                if name == "stored_fusion":
                    verify_metrics(metrics, metadata["fusion"], f"{subject}/{key}/fusion")
                    for group in groups:
                        support[group].update(r.get("observation_count", 0) for r in scores if r["present"])
                elif name.startswith("sensor_"):
                    verify_metrics(metrics, metadata["per_camera"][name.removeprefix("sensor_")], f"{subject}/{key}/{name}")
                for group in groups:
                    pools[group][name].add(scores)
                object_pools[metadata["object_name"]][name].add(scores)
                item["methods"][name] = metrics
                write_gzip(directory / f"{name}.scores.jsonl.gz", scores)
            records.append(item)
            write_json(directory / "summary.json", item)
            print(f"[{subject} {index}/{len(takes)}] {key}: evaluated saved poses", flush=True)
    group_reports = {}
    for group, methods in pools.items():
        selected = [t for t in records if group == "all_subjects" or t["subject"] == group
                    or (group == "development_subject1" and t["subject"] == "subject1")
                    or (group == "new_subjects2_4" and t["subject"] != "subject1")]
        group_reports[group] = {
            "n_takes": len(selected), "n_frames": sum(t["n_frames"] for t in selected),
            "n_observations": sum(t["n_observations"] for t in selected),
            "methods": {name: pool.result() for name, pool in methods.items()},
            "paired_fusion_minus": paired_comparisons(selected),
            "support_counts": dict(sorted(support[group].items())),
        }
    report = {"inputs": inputs, "code_sha256": reference_code,
              "runtime_source_matches_current": reference_code == current_code,
              "effective_fusion_config": reference_config, "arguments": reference_arguments,
              "groups": group_reports, "per_take": records,
              "per_object": {obj: {name: pool.result() for name, pool in methods.items()}
                             for obj, methods in object_pools.items()},
              "notes": ["Stored poses only; no tracking/fusion rerun, artificial corruption or GT alignment.",
                        "Subject1 is development data; subject2-4 are evaluated separately with the frozen configuration.",
                        "Per-frame score pools give exact quantiles; paired comparisons weight takes equally.",
                        "Bootstrap resamples takes, not frames; repeated subjects/objects remain dependent. Descriptive intervals only.",
                        "GT-mask-assisted first-frame initialization; rotation and ADD do not quotient symmetry."]}
    write_json(output / "summary.json", report)
    lines = ["# H2O saved-pose evaluation across subjects", ""]
    for group, data in group_reports.items():
        lines += [f"## {group}", "", f"{data['n_takes']} takes / {data['n_frames']} frames.", "",
                  "| Method | T mean mm | T P95 mm | R P95 deg | ADD P95 mm | Failure % |", "|---|---:|---:|---:|---:|---:|"]
        for name, metrics in data["methods"].items():
            values = [metrics[k] for k in ("translation_mean_mm", "translation_p95_mm", "rotation_p95_deg", "mesh_add_p95_mm")]
            values.append(metrics["failure_rate_including_missing"] * 100)
            lines.append(f"| {name} | " + " | ".join("N/A" if v is None else f"{v:.3f}" for v in values) + " |")
        lines.append("")
    lines += ["## Limits", ""] + [f"- {note}" for note in report["notes"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/h2o/results_all_subjects"))
    parser.add_argument("--run-name", default="results_0")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts/h2o_subjects_review"))
    args = parser.parse_args()
    output = allocate_results(args.output_parent.resolve())
    print(f"Report: {output}", flush=True)
    analyze(args.root.resolve(), args.run_name, list(dict.fromkeys(args.subjects)), output)
    print(f"Completed: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
