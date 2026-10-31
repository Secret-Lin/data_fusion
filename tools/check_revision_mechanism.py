"""Bounded, real-cache revision controls; no inference or production modifications.

R0 vs R2 isolates irreversible vs revisable support with checkpointing held fixed.
R1 vs R2 isolates ordinary replacement vs checkpointing with selection held fixed.
The local patches are sequential offline instrumentation, never a live adapter.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.pose_consensus import ConsensusResult, robust_pose_center, select_pose_consensus
from experiments.h2o.ablation import FusionRunError, Variant, build_config, read_gzip, run_fusion, save_scores
from experiments.h2o.evaluation import allocate_results, score_poses, summarize, write_json
from experiments.h2o.foundationpose_results import load_result_take
from managers.fusion_manager import FusionManager
from tools.run_h2o_foundationpose_ablation import code_fingerprints, compare_pose_outputs


MODES = ("R0", "R1", "R2")
CASES = (
    ("results_all_subjects", "FP", "h1_0", 120, 1.345),
    ("m3t_all_subjects", "M3T", "h1_0", 120, 1.345),
    ("megapose_all_subjects", "MP", "h1_0", 120, 1.345),
    ("megapose_all_subjects", "MP-noHuber", "o2_3", 90, 0.0),
)


def irreversible_selection(sensor_ids, admitted, decided, poses, covariances, **kwargs):
    """Accept a new camera iff compatible with every previously admitted camera.

    Decisions persist for this object/group only. Duplicate sensor packets replace
    their value but not their admission. Use exactly the production compatibility
    matrix and robust-center implementation; never inspect ground truth.
    """
    proposal = select_pose_consensus(poses, covariances, **kwargs)
    for i, sensor in enumerate(sensor_ids):
        if sensor in decided:
            continue
        if all(proposal.pair_consistent[i, sensor_ids.index(old)] for old in admitted):
            admitted.add(sensor)
        decided.add(sensor)
    selected = [i for i, sensor in enumerate(sensor_ids) if sensor in admitted]
    center = robust_pose_center([poses[i] for i in selected],
                                [covariances[i] for i in selected], kwargs["huber_k"])
    return ConsensusResult(selected, center, proposal.ambiguous, proposal.pair_consistent)


def variant(mode: str, reverse: bool, huber: float) -> Variant:
    if mode not in MODES:
        raise ValueError(mode)
    return Variant(f"{mode}_{'reverse' if reverse else 'forward'}", mode, {
        "enable_world_revision_checkpoint": mode != "R1",
        "robust_huber_k": huber,
    }, reverse_order=reverse)


async def run_control(observations: list[dict], mode: str, reverse=False, huber=1.345):
    trace = []

    class InstrumentedManager(FusionManager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.admitted = {}
            self.decided = {}
            self.previous_support = {}

        def _revise_world_observation(self, object_name, step, obs, sync_group):
            key = (object_name, step)
            if mode == "R0":
                def select(poses, covariances, **kwargs):
                    sensors = sorted(self._world_candidates[key])
                    return irreversible_selection(
                        sensors, self.admitted.setdefault(key, set()),
                        self.decided.setdefault(key, set()), poses, covariances, **kwargs)
                with patch("managers.fusion_manager.select_pose_consensus", select):
                    result = super()._revise_world_observation(object_name, step, obs, sync_group)
            else:
                result = super()._revise_world_observation(object_name, step, obs, sync_group)
            support = {o.sensor_id for o in self._step_observations[key]}
            previous = self.previous_support.get(key, set())
            trace.append({"object_name": object_name, "frame_id": sync_group,
                          "sensor_id": obs.sensor_id, "selected": sorted(support),
                          "removed": sorted(previous - support),
                          "added": sorted(support - previous),
                          "candidate_count": len(self._world_candidates[key])})
            self.previous_support[key] = support
            return result

    chosen = variant(mode, reverse, huber)
    # Scope is local to this awaited sequential run; no production module edits.
    with patch("experiments.h2o.ablation.FusionManager", InstrumentedManager):
        try:
            poses, runtime = await run_fusion(observations, chosen, fps=30)
            failure = None
        except FusionRunError as exc:
            poses, runtime, failure = exc.predictions, {}, exc.failure
    return poses, runtime, failure, trace


def write_table(output: Path, cases: list[dict]) -> None:
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Case & Mode & T mean & T95 & R mean & R95 & Fail (\%) & Cov. (\%) \\",
             r"\midrule"]
    for case in cases:
        for mode in MODES:
            m = case["methods"][f"{mode}_forward"]["metrics"]
            values = [m[k] for k in ("translation_mean_mm", "translation_p95_mm",
                                     "rotation_mean_deg", "rotation_p95_deg")]
            values += [100*m["failure_rate_including_missing"], 100*m["coverage"]]
            lines.append(case["label"] + " & " + mode + " & " + " & ".join(
                "--" if v is None else f"{v:.2f}" for v in values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output / "mechanism_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def check(root: Path, parent: Path) -> Path:
    output = allocate_results(parent.resolve())
    source_hashes = code_fingerprints()
    tool_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cases = []
    for source, label, take, limit, huber in CASES:
        path = root / source / "subject1" / "results_0" / "takes" / take
        gt, observations, _, points, _ = load_result_take(path, limit)
        if not gt or not observations:
            raise ValueError(f"Empty cache: {path}")
        # R0's control is defined for one observation per camera/group. Do not
        # silently generalize the experiment to changing same-camera packets.
        keys = [(r["object_name"], r["frame_id"], r["camera_id"]) for r in observations]
        if len(keys) != len(set(keys)):
            raise ValueError("This controlled protocol requires unique camera/group records")
        directory = output / label
        directory.mkdir()
        write_json(directory / "input_snapshot.json", {"targets": gt, "observations": observations})
        methods, all_poses = {}, {}
        input_hash = hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest()
        print(f"{label}: {len(gt)} targets / {len(observations)} observations", flush=True)
        for mode in MODES:
            for reverse in (False, True):
                chosen = variant(mode, reverse, huber)
                with (directory / f"{chosen.name}.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
                    poses, runtime, failure, trace = await run_control(observations, mode, reverse, huber)
                scores = score_poses(gt, poses, mesh_points=points)
                save_scores(directory / chosen.name, poses, scores)
                write_json(directory / chosen.name / "support_trace.json", trace)
                methods[chosen.name] = {
                    "metrics": summarize(scores), "failure": failure, "runtime": runtime,
                    "effective_config": asdict(build_config(chosen).fusion),
                    "support_retraction_events": sum(bool(r["removed"]) for r in trace),
                    "trace_scope": "Selection immediately before optimization; last event may fail",
                }
                all_poses[chosen.name] = poses
                print(f"  {chosen.name}: {len(poses)}/{len(gt)}; failed={failure is not None}", flush=True)
        if hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest() != input_hash:
            raise RuntimeError("Observation mutation")
        case = {"label": label, "input": str(path), "max_frames": limit,
                "n_targets": len(gt), "observations_sha256": input_hash,
                "huber": huber, "methods": methods,
                "order_checks": {mode: compare_pose_outputs(all_poses[f"{mode}_forward"],
                                                             all_poses[f"{mode}_reverse"])
                                 for mode in MODES}}
        cases.append(case)
        write_json(directory / "summary.json", case)
    report = {"cases": cases, "code_sha256": source_hashes, "tool_sha256": tool_hash,
              "code_changed_during_run": (code_fingerprints() != source_hashes or
                  hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != tool_hash),
              "scope": "Selected development diagnostics, not an unbiased full-dataset estimate",
              "frontend_rerun": False, "artificial_corruption": False,
              "gt_usage": "Inherited GT-assisted initialization; GT only scores fusion outputs",
              "controls": {"R0_vs_R2": "support revisability, checkpoint enabled in both",
                           "R1_vs_R2": "checkpoint, production selection shared in both"}}
    write_json(output / "summary.json", report)
    write_table(output, cases)
    return output


def reproduce_table(directory: Path) -> None:
    """Recompute every method's metrics from archived scores before typesetting."""
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    for case in report["cases"]:
        for name, method in case["methods"].items():
            scores = read_gzip(directory / case["label"] / name / "scores.jsonl.gz")
            if summarize(scores) != method["metrics"]:
                raise ValueError(f"Score/summary mismatch: {case['label']}/{name}")
    write_table(directory, report["cases"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/h2o"))
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts/revision_mechanism"))
    parser.add_argument("--report-only", type=Path, help="Verify scores and reproduce the table without fusion")
    args = parser.parse_args()
    if args.report_only:
        reproduce_table(args.report_only)
        return
    print(asyncio.run(check(args.root, args.output_parent)))


if __name__ == "__main__":
    main()
