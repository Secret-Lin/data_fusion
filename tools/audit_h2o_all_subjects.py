"""Read-only audit and exact cross-subject pooling of saved H2O ablations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from experiments.h2o.evaluation import MetricAccumulator, allocate_results, write_json
from tools.audit_h2o_fusion import COMPARISON_METRICS, audit, paired_take_summary, write_markdown


def matched_comparison(records: list[dict], first: str, second: str) -> dict:
    result = {}
    for key in COMPARISON_METRICS:
        pairs = [(r["methods"][first][key], r["methods"][second][key]) for r in records]
        valid = [(a, b) for a, b in pairs if a is not None and b is not None]
        result[key] = ({**paired_take_summary([a for a, _ in valid], [b for _, b in valid]),
                        "takes_without_metric": len(pairs) - len(valid)} if valid else None)
    return result


def run(root: Path, run_name: str, output: Path) -> dict:
    audits, records = {}, []
    pools = defaultdict(lambda: defaultdict(MetricAccumulator))
    reference_code = None
    reference_methods = None
    reference_configs = None
    for number in (1, 2, 3, 4):
        subject = f"subject{number}"
        source = root / subject / run_name
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        input_summary = json.loads((Path(summary["input"]) / "summary.json").read_text(encoding="utf-8"))
        expected = {r["take"].replace("/", "_") for r in input_summary["per_take"]}
        actual = {r["take"] for r in summary["per_take"]}
        if (summary["max_frames_per_take"] != 0 or expected != actual
                or input_summary["n_takes_failed"] != 0
                or input_summary["n_takes_completed"] != input_summary["n_takes_requested"]
                or len(actual) != input_summary["n_takes_requested"]):
            raise ValueError(f"Not a complete all-frame/all-take ablation: {subject}")
        statuses = {k: v["status"] for k, v in summary["methods"].items()}
        if any(v not in ("completed", "not_applicable") for v in statuses.values()):
            raise ValueError(f"Unsuccessful method: {subject}")
        if number == 1:
            reference_code, reference_methods = summary["code_sha256"], statuses
        if reference_code != summary["code_sha256"] or reference_methods != statuses:
            raise ValueError(f"Mixed source/methods across subjects: {subject}")
        print(f"[{subject}] Verifying every saved score target and metric", flush=True)
        verified = audit(source)
        audits[subject] = verified
        folder = output / subject
        folder.mkdir()
        write_json(folder / "summary.json", verified)
        write_markdown(folder / "summary.md", verified)
        methods = list(verified["pooled_metrics"])
        for index, item in enumerate(summary["per_take"], 1):
            take = item["take"]
            directory = source / "takes" / take
            report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            configs = {k: v.get("effective_fusion_config") for k, v in report["methods"].items()}
            if reference_configs is None:
                reference_configs = configs
            if configs != reference_configs:
                raise ValueError(f"Mixed method configurations: {subject}/{take}")
            groups = ["all_subjects", subject, "development_subject1" if number == 1 else "new_subjects2_4"]
            for method in methods:
                with gzip.open(directory / method / "scores.jsonl.gz", "rt", encoding="utf-8") as stream:
                    rows = [json.loads(line) for line in stream if line.strip()]
                for group in groups:
                    pools[group][method].add(rows)
            records.append({"subject": subject, "take": take, "object": report["object_name"],
                            "methods": {k: report["methods"][k]["metrics"] for k in methods}})
            if index % 10 == 0 or index == summary["n_takes"]:
                print(f"[{subject}] Pooled {index}/{summary['n_takes']} takes", flush=True)
    groups = {}
    for group, method_pools in pools.items():
        selected = [r for r in records if group == "all_subjects" or r["subject"] == group
                    or (group == "development_subject1" and r["subject"] == "subject1")
                    or (group == "new_subjects2_4" and r["subject"] != "subject1")]
        comparisons = {"full_minus": {}, "checkpoint_minus": {}}
        for method in method_pools:
            if method != "full":
                comparisons["full_minus"][method] = matched_comparison(selected, "full", method)
        for method in ("full", "sensor_cam1", "mean_equal", "quality_consensus", "revision_checkpoint_quality"):
            comparisons["checkpoint_minus"][method] = matched_comparison(selected, "revision_checkpoint", method)
        groups[group] = {"n_takes": len(selected), "methods": {k: v.result() for k, v in method_pools.items()},
                         "comparisons": comparisons}
    result = {"input": str(root), "run_name": run_name, "code_sha256": reference_code,
              "effective_method_configs": reference_configs, "audits": audits,
              "groups": groups, "per_take": records,
              "notes": ["Read-only saved-score audit; no fusion/tracking rerun or GT-based selection.",
                        "Exact frame-pooled quantiles; paired macro differences weight takes equally.",
                        "Subject1 was development; candidate diagnostics also inspected selected subject2-4 takes.",
                        "Take bootstrap intervals are descriptive; subjects/objects remain dependent.",
                        "No artificial corruption; GT-assisted initialization; rotation/ADD without symmetry quotient."]}
    write_json(output / "summary.json", result)
    lines = ["# H2O all-subject cached ablation audit", ""]
    for name, group in groups.items():
        lines += [f"## {name} ({group['n_takes']} takes)", "",
                  "| Method | T mean mm | T P95 mm | R P95 deg | ADD P95 mm | Failure % |",
                  "|---|---:|---:|---:|---:|---:|"]
        for method, m in group["methods"].items():
            values = [m[k] for k in ("translation_mean_mm", "translation_p95_mm", "rotation_p95_deg", "mesh_add_p95_mm")]
            values.append(m["failure_rate_including_missing"] * 100)
            lines.append(f"| {method} | " + " | ".join("N/A" if v is None else f"{v:.3f}" for v in values) + " |")
        lines.append("")
    lines += ["## Limitations", ""] + [f"- {note}" for note in result["notes"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/h2o/ablation_all_subjects"))
    parser.add_argument("--run-name", default="results_0")
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts/h2o_all_ablation_audit"))
    args = parser.parse_args()
    output = allocate_results(args.output_parent.resolve())
    print(f"Output: {output}", flush=True)
    run(args.root.resolve(), args.run_name, output)
    print(f"Completed: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
