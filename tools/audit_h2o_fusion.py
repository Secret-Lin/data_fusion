"""Audit completed H2O ablations without rerunning a tracker or fusion.

Reads saved per-frame scores and reports, re-pools metrics, and compares take-
level paired means. Bootstrap units are takes, never individual video frames.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.h2o.evaluation import MetricAccumulator, allocate_results, bootstrap_mean_ci, read_jsonl, write_json
from tools.run_h2o_foundationpose_ablation import code_fingerprints


COMPARISON_METRICS = (
    "translation_mean_mm", "rotation_mean_deg", "mesh_add_mean_mm",
    "failure_rate_including_missing",
)


def paired_take_summary(first: list[float], second: list[float]) -> dict:
    """Inputs must already be paired by take; negative delta favors first."""
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("Require nonempty matched one-dimensional take arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired take metrics must be finite")
    delta = a - b
    tolerance = 1e-6
    return {
        "n_takes": len(a), "first_better": int(np.sum(delta < -tolerance)),
        "ties": int(np.sum(np.abs(delta) <= tolerance)),
        "second_better": int(np.sum(delta > tolerance)),
        "macro_mean_delta": float(delta.mean()),
        "macro_median_delta": float(np.median(delta)),
        "macro_mean_bootstrap_ci95": bootstrap_mean_ci(delta),
        "bootstrap_unit": "take", "bootstrap_seed": 811, "bootstrap_replicates": 2000,
    }


def validate_targets(rows: list[dict], truth: list[dict], label: str) -> None:
    keys = lambda items: [(int(r["frame_id"]), r["object_name"]) for r in items]
    actual, expected = keys(rows), keys(truth)
    if len(set(actual)) != len(actual) or set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError(f"Duplicate, missing or mismatched evaluation targets: {label}")


def verify_metrics(actual: dict, expected: dict, label: str) -> None:
    for key, value in actual.items():
        other = expected.get(key)
        if isinstance(value, (int, float)) and isinstance(other, (int, float)):
            equal = bool(np.isclose(value, other, rtol=1e-10, atol=1e-10))
        else:
            equal = value == other
        if not equal:
            raise ValueError(f"Re-pooled metric mismatch: {label}/{key}: {value!r} != {other!r}")


def audit(root: Path) -> dict:
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entries = summary["per_take"]
    if summary.get("failed_takes") or summary.get("failed_methods") or summary.get("code_changed_during_run"):
        raise ValueError("Cannot validate a failed or mixed-code run as a complete experiment")
    if len(entries) != summary["n_takes"] or len({r["take"] for r in entries}) != len(entries):
        raise ValueError("Incomplete or duplicate take index")
    methods = [k for k, v in summary["methods"].items() if v["status"] == "completed"]
    if "full" not in methods:
        raise ValueError("Missing full fusion reference")
    pools = defaultdict(MetricAccumulator)
    by_object = defaultdict(lambda: defaultdict(MetricAccumulator))
    support_counts = Counter()
    reports, diagnostics = [], defaultdict(list)
    for entry in entries:
        take = entry["take"]
        directory = root / "takes" / take
        report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if report.get("code_changed_during_run") or report["code_sha256"] != summary["code_sha256"]:
            raise ValueError(f"Inconsistent source fingerprint in {take}")
        if any(report.get(k) is not False for k in ("artificial_noise", "artificial_dropout", "artificial_delay")):
            raise ValueError(f"Unexpected artificial intervention or unknown provenance in {take}")
        truth = read_jsonl(Path(report["input"]) / "ground_truth.jsonl")
        if summary.get("max_frames_per_take"):
            truth = truth[:summary["max_frames_per_take"]]
        if len(truth) != report["n_ground_truth"]:
            raise ValueError(f"Ground-truth count mismatch in {take}")
        for method in methods:
            with gzip.open(directory / method / "scores.jsonl.gz", "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            validate_targets(rows, truth, f"{take}/{method}")
            pool = MetricAccumulator()
            pool.add(rows)
            verify_metrics(pool.result(), report["methods"][method]["metrics"], f"{take}/{method}")
            pools[method].add(rows)
            by_object[report["object_name"]][method].add(rows)
            if method == "full":
                support_counts.update(int(r.get("observation_count", 0)) for r in rows if r["present"])
        for pair, check in report.get("output_diagnostics", {}).items():
            diagnostics[pair].append(check)
        reports.append((take, report))
    for method in methods:
        verify_metrics(pools[method].result(), summary["methods"][method]["metrics"], method)
    paired = {}
    for method in methods:
        if method == "full":
            continue
        paired[method] = {}
        for key in COMPARISON_METRICS:
            values = [(r["methods"]["full"]["metrics"].get(key), r["methods"][method]["metrics"].get(key))
                      for _, r in reports]
            valid = [(a, b) for a, b in values if a is not None and b is not None]
            if valid:
                paired[method][key] = {
                    **paired_take_summary([a for a, _ in valid], [b for _, b in valid]),
                    "takes_without_metric": len(values) - len(valid),
                }
    diagnostic_report = {}
    for name, checks in diagnostics.items():
        diagnostic_report[name] = {
            "n_takes": len(checks),
            "identical_pose_and_covariance_takes": sum(c["pose_and_covariance_identical"] for c in checks),
            "active_sensor_changed_frames": sum(c["active_sensor_changed_frames"] or 0 for c in checks),
            "max_pose_matrix_delta": max((c["max_pose_matrix_delta"] for c in checks
                                          if c["max_pose_matrix_delta"] is not None), default=None),
        }
    return {
        "input": str(root.resolve()), "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_takes": len(entries), "n_frames": pools["full"].n_expected,
        "n_observations": sum(r["n_observations"] for _, r in reports),
        "all_saved_score_targets_and_metrics_verified": True,
        "frozen_source_matches_current_code": summary["code_sha256"] == code_fingerprints(),
        "init_modes": sorted({r["init_mode"] for _, r in reports}),
        "pooled_metrics": {k: p.result() for k, p in pools.items()},
        "paired_full_minus": paired, "output_diagnostics": diagnostic_report,
        "full_support_counts": dict(sorted(support_counts.items())),
        "per_object_metrics": {obj: {k: pool.result() for k, pool in groups.items()}
                               for obj, groups in by_object.items()},
        "inactive_methods": {k: v for k, v in summary["methods"].items() if v["status"] != "completed"},
        "notes": [
            "Analysis only: no RGB-D inference, fusion rerun, corruption, alignment or parameter changes.",
            "Pooled quantiles are recomputed from saved frame scores; macro comparisons weight each take equally.",
            "Delta = full minus comparator; negative favors full. Failure deltas are rates, not percentages.",
            "Bootstrap resamples paired takes, not correlated frames; same-subject/object dependence remains.",
            "Intervals are descriptive development-set evidence, not independent-subject generalization or multiple-testing-adjusted claims.",
            "Pose equivalence diagnostics are inherited from the frozen run, not independently replayed here.",
            "GT-assisted first-frame initialization is not end-to-end automatic detection; rotation/ADD do not quotient symmetries.",
        ],
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = ["# H2O completed fusion audit", "", f"Input: `{report['input']}`", "",
             f"Verified {report['n_takes']} takes, {report['n_frames']} frames, {report['n_observations']} observations.", "",
             "## Pooled metrics", "",
             "| Method | Mean t mm | P95 t mm | P95 R deg | P95 ADD mm | Failure % |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, m in report["pooled_metrics"].items():
        values = [m[k] for k in ("translation_mean_mm", "translation_p95_mm", "rotation_p95_deg", "mesh_add_p95_mm")]
        values += [m["failure_rate_including_missing"] * 100]
        lines.append(f"| {name} | " + " | ".join("N/A" if v is None else f"{v:.3f}" for v in values) + " |")
    lines += ["", "## Paired take-mean comparisons", "",
              "Negative delta favors full. W/T/L = full better / tie / comparator better.", "",
              "| Comparator | Metric | W/T/L | Macro mean delta | Take-bootstrap 95% interval |",
              "|---|---|---:|---:|---|"]
    for name, metrics in report["paired_full_minus"].items():
        for key, m in metrics.items():
            ci = m["macro_mean_bootstrap_ci95"]
            interval = "N/A" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(f"| {name} | {key} | {m['first_better']}/{m['ties']}/{m['second_better']} | "
                         f"{m['macro_mean_delta']:.4f} | {interval} |")
    lines += ["", "## Scope and limitations", ""] + [f"- {note}" for note in report["notes"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts/h2o_fusion_audit"))
    args = parser.parse_args()
    report = audit(args.run_dir.resolve())
    output = allocate_results(args.output_parent.resolve())
    write_json(output / "summary.json", report)
    write_markdown(output / "summary.md", report)
    print(f"Verified {report['n_takes']} takes / {report['n_frames']} frames. Report: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
