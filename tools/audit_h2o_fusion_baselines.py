"""Audit completed baseline scores and archive a compact publication snapshot.

Read-only on experiment inputs; never runs inference or fusion. Recomputes
translation/rotation statistics from every saved score, not just summary means.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.h2o.evaluation import allocate_results
from tools.run_h2o_fusion_baselines import source_hashes


METHOD_LABELS = {
    "se3_gaussian_quality": "Gaussian / quality",
    "se3_huber_fixed": "Huber / fixed",
    "se3_huber_quality": "Huber / quality",
    "eskf_fixed": "ESKF / fixed",
    "eskf_quality": "ESKF / quality",
    "eskf_gated_fixed": "Gated ESKF / fixed",
    "eskf_gated_quality": "Gated ESKF / quality",
    "ours": "Ours",
}
FRONTENDS = {"foundationpose": "FoundationPose", "m3t": "M3T", "megapose": "MegaPose"}
FIELDS = ("translation_mean_mm", "translation_p50_mm", "translation_p95_mm",
          "rotation_mean_deg", "rotation_p50_deg", "rotation_p95_deg",
          "failure_rate_including_missing", "coverage")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: np.ndarray) -> dict:
    # This audit intentionally requires the observed completed/full-coverage run.
    assert values.ndim == 2 and values.shape[1] == 2 and np.isfinite(values).all()
    t, r = values.T
    assert np.all(t >= 0) and np.all((r >= 0) & (r <= 180.000001))
    return {"translation_mean_mm": float(np.mean(t)),
            "translation_p50_mm": float(np.percentile(t, 50)),
            "translation_p95_mm": float(np.percentile(t, 95)),
            "rotation_mean_deg": float(np.mean(r)),
            "rotation_p50_deg": float(np.percentile(r, 50)),
            "rotation_p95_deg": float(np.percentile(r, 95)),
            "failure_rate_including_missing": float(np.mean((t > 50) | (r > 10))),
            "coverage": 1.0}


def check_metrics(actual: dict, expected: dict) -> None:
    for field in FIELDS:
        assert np.isclose(actual[field], expected[field], rtol=1e-10, atol=1e-9), (field, actual[field], expected[field])


def write_table(summary: dict, target: Path) -> None:
    lines = [r"\begin{tabular}{llrrrrr}\toprule",
             r"Frontend & Method & T mean & T P95 & R mean & R P95 & Fail (\%) \\\midrule"]
    fields = ("translation_mean_mm", "translation_p95_mm", "rotation_mean_deg", "rotation_p95_deg", "failure_rate_including_missing")
    for index, (frontend, label) in enumerate(FRONTENDS.items()):
        if index:
            lines.append(r"\midrule")
        for method, name in METHOD_LABELS.items():
            m = summary["methods_by_frontend"][frontend][method]
            numbers = [m[f] * (100 if f.startswith("failure") else 1) for f in fields]
            lines.append(label + " & " + name + " & " + " & ".join(f"{x:.2f}" for x in numbers) + r" \\")
    lines.append(r"\bottomrule\end{tabular}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(run: Path, output_parent: Path) -> Path:
    summary = read_json(run / "summary.json")
    assert summary["status"] == "completed" and summary["max_frames"] == 0
    assert summary["n_requested_takes"] == summary["n_completed_takes"] == 552
    assert not summary["failed_takes"] and not summary["failed_methods"]
    assert not summary["code_changed_during_run"] and not summary["inputs_mutated"]
    assert summary["code_sha256"] == source_hashes(), "Estimator code no longer matches the run"
    jobs = summary["jobs"]
    assert len({j["relative_output"] for j in jobs}) == 552
    counts = Counter((j["frontend"], j["subject"]) for j in jobs)
    assert counts == Counter({(f, f"subject{i}"): n for f in FRONTENDS for i, n in enumerate((47, 43, 48, 46), 1)})
    pooled = defaultdict(list)
    totals = defaultdict(Counter)
    reports, hashes = [], {}
    total_rows = 0
    for index, job in enumerate(jobs):
        directory = run / job["relative_output"]
        report = read_json(directory / "summary.json")
        assert report["status"] == "completed" and not report["inputs_mutated"] and not report["code_changed_during_run"]
        assert report["baseline_config"] == summary["baseline_config"]
        assert len(report["methods"]) == 16
        frontend = job["frontend"]
        identities = None
        take_arrays = {}
        for method, details in report["methods"].items():
            assert details["status"] == "completed" and details["failure"] is None
            path = directory / method / "scores.jsonl.gz"
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            keys = [(r["frame_id"], r["object_name"]) for r in rows]
            assert len(keys) == len(set(keys)) == report["n_targets"]
            assert identities is None or keys == identities, "Different evaluation targets"
            identities = keys
            assert all(r["present"] for r in rows), "Missing outputs require a separate partial-coverage audit"
            values = np.array([[r["translation_error_m"] * 1000, r["rotation_error_deg"]] for r in rows])
            assert all(r["success_5cm_10deg"] == (r["translation_error_m"] <= .05 and r["rotation_error_deg"] <= 10) for r in rows)
            check_metrics(stats(values), details["metrics"])
            check_metrics(stats(values), report["common_output_metrics"][method])
            pooled[(frontend, method)].append(values)
            take_arrays[method] = values
            totals[(frontend, method)]["prediction_only"] += details["prediction_only_outputs"]
            totals[(frontend, method)]["iteration_limit"] += details["batch_iteration_limit_outputs"]
            hashes[str(path.relative_to(run))] = sha256(path)
            total_rows += len(rows)
        for method, pair in report["paired_ours_minus"].items():
            delta = np.mean(take_arrays["ours"] - take_arrays[method], axis=0)
            assert pair["n_common"] == report["n_targets"]
            assert np.allclose(delta, [pair["mean_translation_delta_mm"], pair["mean_rotation_delta_deg"]], rtol=1e-10, atol=1e-9)
        totals[(frontend, "input")]["duplicates"] += report["n_duplicate_replaced"]
        totals[(frontend, "input")]["zero_score_groups"] += report["zero_score_groups_uniform_simple_baseline"]
        reports.append(report)
        hashes[str((directory / "summary.json").relative_to(run))] = sha256(directory / "summary.json")
        if (index + 1) % 48 == 0:
            print(f"Audited {index+1}/552 takes", flush=True)
    for (frontend, method), parts in pooled.items():
        values = np.concatenate(parts)
        assert len(values) == 114329
        measured = stats(values)
        check_metrics(measured, summary["methods_by_frontend"][frontend][method])
        check_metrics(measured, summary["common_output_metrics_by_frontend"][frontend][method])
    for frontend, methods in summary["paired_ours_minus"].items():
        for method, paired in methods.items():
            for metric in ("mean_translation_delta_mm", "mean_rotation_delta_deg"):
                a = np.array([r["paired_ours_minus"][method][metric] for r in reports if r["frontend"] == frontend])
                p = paired["summary"][metric]
                assert len(a) == p["n_paired_takes"] == 184
                assert np.isclose(np.mean(a), p["mean"])
                assert (int(np.sum(a < 0)), int(np.sum(a > 0))) == (p["ours_better"], p["ours_worse"])
    output = allocate_results(output_parent)
    for name in ("summary.json", "summary.md", "manifest.json"):
        shutil.copy2(run / name, output / name)
        hashes[name] = sha256(run / name)
    (output / "take_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (output / "source_score_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    result = {"status": "passed", "run": str(run), "n_takes": 552, "n_score_rows": total_rows,
              "n_methods_per_take": 16, "n_targets_per_frontend": 114329,
              "diagnostics": {f"{f}/{m}": dict(c) for (f, m), c in totals.items()},
              "scope": "Recomputed saved-score summaries and pairwise means; no pose re-estimation or new frontend run"}
    (output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_table(summary, output / "independent_baselines_table.tex")
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    for name in summary["code_sha256"]:
        src = Path(__file__).resolve().parents[1] / name
        dest = output / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    files = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "SHA256.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Publication snapshot: {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, default=Path("paper/h2o_revision/data"))
    args = parser.parse_args()
    audit(args.run_dir.resolve(), args.output_parent.resolve())


if __name__ == "__main__":
    main()
