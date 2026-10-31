"""Compare execution-only H2O changes on identical ordered observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.h2o.evaluation import allocate_results, read_jsonl, write_json


def compare_files(baseline: Path, candidate: Path, filename: str, pose_key: str) -> dict:
    before, after = read_jsonl(baseline / filename), read_jsonl(candidate / filename)
    def key(row):
        return row["frame_id"], row.get("camera_id"), row["object_name"]
    if list(map(key, before)) != list(map(key, after)):
        raise ValueError(f"Observation count/order/identity changed: {filename}")
    if not before:
        raise ValueError(f"No observations to compare: {filename}")
    a, b = np.asarray([r[pose_key] for r in before]), np.asarray([r[pose_key] for r in after])
    translation = np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1) * 1000
    relative = a[:, :3, :3].transpose(0, 2, 1) @ b[:, :3, :3]
    angle = np.degrees(Rotation.from_matrix(relative).magnitude())
    semantic_keys = ("mode", "score", "t_host", "arrival_time", "quality_metrics",
                     "observation_count", "active_sensors", "source", "T_world_camera",
                     "integrity", "covariance", "online_extrinsics")
    changed = {k: sum(x.get(k) != y.get(k) for x, y in zip(before, after)) for k in semantic_keys}
    return {
        "n": len(before), "matrix_element_max_abs_delta": float(np.max(np.abs(a - b))),
        "translation_max_delta_mm": float(translation.max()),
        "rotation_max_delta_deg": float(angle.max()),
        "translation_p95_delta_mm": float(np.percentile(translation, 95)),
        "semantic_field_changed_rows": changed,
        "bitwise_identical_poses": bool(np.array_equal(a, b)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts/h2o_performance/comparisons"))
    args = parser.parse_args()
    report = {"baseline": str(args.baseline.resolve()), "candidate": str(args.candidate.resolve())}
    for label, filename, pose_key in (
        ("observations", "tracking_before_fusion.jsonl", "pose_world"),
        ("fusion", "tracking_after_fusion.jsonl", "pose_world"),
    ):
        report[label] = compare_files(args.baseline, args.candidate, filename, pose_key)
    summaries = [json.loads((p / "summary.json").read_text(encoding="utf-8"))
                 for p in (args.baseline, args.candidate)]
    report["errors_unchanged"] = summaries[0].get("errors") == summaries[1].get("errors")
    report["fusion_metrics"] = [s["fusion"] for s in summaries]
    report["passed"] = report["errors_unchanged"] and all(
        report[k]["translation_max_delta_mm"] < 0.01
        and report[k]["rotation_max_delta_deg"] < 0.01
        and not any(report[k]["semantic_field_changed_rows"].values())
        for k in ("observations", "fusion")
    )
    output = allocate_results(args.output_parent.resolve())
    write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report: {output / 'summary.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
