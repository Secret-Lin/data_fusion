"""Bounded cached-observation regression checks; never starts frontend inference."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.h2o.ablation import FusionRunError, VARIANTS, Variant, build_config, run_fusion, save_scores
from experiments.h2o.evaluation import allocate_results, score_poses, summarize, write_json
from experiments.h2o.foundationpose_results import load_result_take
from tools.run_h2o_foundationpose_ablation import code_fingerprints, compare_pose_outputs


CASES = [
    ("megapose_all_subjects", 1, "o2_3", 0),
    ("megapose_all_subjects", 2, "k2_3", 0),
    ("megapose_all_subjects", 4, "h2_0", 0),
    ("megapose_all_subjects", 1, "h1_0", 240),
    ("m3t_all_subjects", 1, "h1_0", 240),
    ("results_all_subjects", 1, "h1_0", 240),
]


async def check(root: Path, output_parent: Path) -> Path:
    output = allocate_results(output_parent.resolve())
    initial_code = code_fingerprints()
    cases = []
    variants = [
        Variant("legacy_full", "Previous offline default", {"enable_world_revision_checkpoint": False}),
        Variant("legacy_no_huber", "Reproduce the failure without checkpoint", {
            "enable_world_revision_checkpoint": False, "robust_huber_k": 0}),
        VARIANTS["full"], VARIANTS["no_huber"], VARIANTS["reverse_order"],
        Variant("no_huber_reverse", "No Huber, reverse same-frame order", {"robust_huber_k": 0}, reverse_order=True),
    ]
    for source, subject, take, limit in CASES:
        path = root / source / f"subject{subject}" / "results_0" / "takes" / take
        gt, observations, _, points, _ = load_result_take(path, limit)
        name = f"{source}_subject{subject}_{take}"
        directory = output / name
        directory.mkdir()
        methods, predictions_by_method = {}, {}
        print(f"CHECK {name}: {len(gt)} frames", flush=True)
        for variant in variants:
            failure = None
            with (directory / f"{variant.name}.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
                try:
                    predictions, runtime = await run_fusion(observations, variant, fps=30)
                except FusionRunError as exc:
                    predictions, runtime = exc.predictions, {"failure": exc.failure}
                    failure = str(exc)
            scores = score_poses(gt, predictions, mesh_points=points)
            save_scores(directory / variant.name, predictions, scores)
            methods[variant.name] = {
                "status": "failed" if failure else "completed", "error": failure,
                "metrics": summarize(scores), "runtime": runtime,
                "effective_fusion_config": asdict(build_config(variant).fusion),
                "max_output_position_norm_m": max((float(np.linalg.norm(np.asarray(p["pose_world"])[:3, 3]))
                                                    for p in predictions), default=None),
            }
            predictions_by_method[variant.name] = predictions
            print(f"  {variant.name}: {methods[variant.name]['status']} {len(predictions)}/{len(gt)}", flush=True)
        case = {"input": str(path), "name": name, "n_frames": len(gt), "max_frames": limit,
                "methods": methods,
                "full_order": compare_pose_outputs(predictions_by_method["full"], predictions_by_method["reverse_order"]),
                "no_huber_order": compare_pose_outputs(predictions_by_method["no_huber"], predictions_by_method["no_huber_reverse"])}
        write_json(directory / "summary.json", case)
        cases.append(case)
    report = {"cases": cases, "code_sha256": initial_code,
              "code_changed_during_run": code_fingerprints() != initial_code,
              "scope": "Three previously failing full takes plus three 240-frame normal segments; not a full-data evaluation",
              "artificial_corruption": False, "frontend_rerun": False}
    write_json(output / "summary.json", report)
    lines = ["# Checkpoint small-sample stability checks", "", report["scope"], "",
             "Legacy failures retain completed prefixes; remaining targets count as missing. "
             "This is not a live latency experiment.", "",
             "| Case | Method | Status | T mean mm | T P95 mm | R mean deg | R P95 deg | Failure % | Coverage % |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for case in cases:
        for name, method in case["methods"].items():
            m = method["metrics"]
            values = [m[k] for k in ("translation_mean_mm", "translation_p95_mm", "rotation_mean_deg", "rotation_p95_deg")]
            values += [100 * m["failure_rate_including_missing"], 100 * m["coverage"]]
            lines.append(f"| {case['name']} | {name} | {method['status']} | " + " | ".join(
                "N/A" if v is None else f"{v:.3f}" for v in values) + " |")
        lines += ["", f"Order checks {case['name']}: full={case['full_order']}; no_huber={case['no_huber_order']}", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/h2o"))
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts/h2o_checkpoint_checks"))
    args = parser.parse_args()
    print(asyncio.run(check(args.root, args.output_parent)))


if __name__ == "__main__":
    main()
