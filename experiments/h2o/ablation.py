"""H2O fusion diagnostics on unmodified annotations, without artificial noise.

This is an offline fusion experiment, not a FoundationPose inference benchmark.
Production managers are reused without changing their online behavior.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from contextlib import redirect_stdout
import csv
from dataclasses import asdict, dataclass, field
import gzip
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from config import BackendConfig
from core.event_bus import EventBus
from core.types import DetectionResult
from experiments.h2o.public_comparison import PUBLIC_COMPARISON
from experiments.h2o.consistency import discover_takes
from experiments.h2o.dataset import H2OSequence
from experiments.h2o.evaluation import (
    MetricAccumulator, allocate_results, attach_keypoints, bootstrap_mean_ci, fingerprint, metric_table,
    paired_delta, read_jsonl, score_poses, summarize, write_json,
)
from managers.baseline_manager import _avg_quat_markley, _quat_wxyz_to_rot, _rot_to_quat_wxyz
from managers.fusion_manager import FusionManager


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)
    grouping: str = "capture"
    reverse_order: bool = False
    exclude_camera: str | None = None
    inactive_reason: str | None = None


VARIANTS = {
    v.name: v for v in [
        Variant("full", "Unmodified production world-pose fusion"),
        Variant("no_adaptive", "Disable quality scaling of observation covariance",
                {"use_adaptive_covariance": False}),
        Variant("no_huber", "Disable Huber in observation/motion factors and consensus center; retain gates",
                {"robust_huber_k": 0.0}),
        Variant("no_gate", "Retain Huber, disable innovation rejection thresholds",
                {"innovation_gate_trans_m": 1e30, "innovation_gate_rot_deg": 1e30,
                 "enable_mahalanobis_gate": False}),
        Variant("no_robust", "Disable Huber and innovation rejection together",
                {"robust_huber_k": 0.0, "innovation_gate_trans_m": 1e30,
                 "innovation_gate_rot_deg": 1e30, "enable_mahalanobis_gate": False}),
        Variant(
            "no_grouping",
            "Every observation constrains its own independent state",
            overrides={"enable_temporal_factors": False},
            grouping="separate",
        ),
        Variant("arrival_grouping", "Assign state by arrival time instead of capture time", grouping="arrival"),
        Variant("no_integrity", "Disable integrity computation; runtime negative control",
                {"enable_integrity_monitoring": False}),
        Variant("reverse_order", "Reverse same-arrival sensor order; sensitivity diagnostic", reverse_order=True),
        Variant("revision_checkpoint", "Re-solve consecutive state revisions from a pre-state checkpoint",
                {"enable_world_revision_checkpoint": True}),
        Variant("revision_checkpoint_reverse", "Checkpoint candidate with reversed same-arrival sensor order",
                {"enable_world_revision_checkpoint": True}, reverse_order=True),
        Variant("revision_checkpoint_quality", "Checkpoint candidate with RGB-D/CAD quality override",
                {"enable_world_revision_checkpoint": True, "enable_consensus_quality_override": True}),
        Variant("weak_initial_prior", "Weaken first-observation prior to sigma=1000",
                {"prior_rot_sigma": 1000.0, "prior_trans_sigma": 1000.0}),
        Variant(
            "no_temporal",
            "Disable temporal motion factors",
            {"enable_temporal_factors": False},
        ),
        Variant(
            "no_history",
            "Disable both motion factors and historical consensus prediction",
            {"enable_temporal_factors": False,
             "enable_consensus_history_prediction": False},
        ),
        Variant("no_online_calibration", "Disable online extrinsic optimization",
                {"enable_online_calibration": False}, inactive_reason=
                "Not applicable: supplied world poses bypass camera/extrinsic graph variables."),
        Variant("quality_consensus", "Enable calibrated-quality candidate override",
                {"enable_consensus_quality_override": True}),
    ]
}
DEFAULT_VARIANTS = ",".join(VARIANTS)
SCENARIOS = ("annotations",)


def build_config(variant: Variant, base_overrides: dict[str, Any] | None = None) -> BackendConfig:
    cfg = BackendConfig()
    cfg.calibration.tag_cube.result_path = ""
    # Offline H2O observations revise the same state once per arriving camera.
    # Start each revision from its pre-state solver, not an excluded view's
    # linearization. Do not change live defaults before live validation.
    cfg.fusion.enable_world_revision_checkpoint = True
    for key, value in {**(base_overrides or {}), **variant.overrides}.items():
        if not hasattr(cfg.fusion, key):
            raise ValueError(f"Unknown fusion configuration field: {key}")
        setattr(cfg.fusion, key, value)
    return cfg


def load_take(take: Path, max_frames: int, fps: float) -> tuple[list[dict], list[dict], list[str]]:
    cameras = H2OSequence.discover_cameras(take)
    sequence = H2OSequence(take, cameras)
    frames = sequence.common_frame_ids()
    if max_frames > 0:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"No complete frames in {take}")
    clean, ground_truth = [], []
    for frame in frames:
        samples = [sequence.load(camera, frame) for camera in cameras]
        annotated = [sample for sample in samples if sample.object_id != 0]
        if not annotated:
            continue
        if len({sample.object_id for sample in annotated}) != 1:
            raise ValueError(f"Inconsistent object identities at {take} frame {frame}")
        reference = annotated[0]
        timestamp = (frame - frames[0]) / fps
        ground_truth.append({"frame_id": frame, "t_host": timestamp,
                             "object_name": reference.object_name,
                             "pose_world": reference.T_world_object.tolist()})
        for sample in annotated:
            clean.append({"frame_id": frame, "t_host": timestamp,
                          "object_name": sample.object_name, "camera_id": sample.camera_id,
                          "pose_world": sample.T_world_object.tolist()})
    if not ground_truth:
        raise ValueError(f"No annotated object in {take}")
    attach_keypoints(ground_truth, take, cameras[0])
    return ground_truth, clean, cameras


def generate_observations(clean: list[dict], *, take_key: str = "", seed: int = 0,
                          scenario: str = "annotations") -> list[dict]:
    """Forward labels unchanged. Old key/seed arguments do not change the data."""
    if scenario not in SCENARIOS:
        raise ValueError("Artificial H2O scenarios have been removed; use annotations")
    output = []
    for row in clean:
        output.append({**row, "pose_world": np.asarray(row["pose_world"]).tolist(), "score": 1.0,
                       "arrival_time": row["t_host"], "synthetic_corruption": False,
                       "observation_kind": "unmodified_h2o_annotation",
                       "source": f"h2o/{row['camera_id']}/foundationpose/{row['object_name']}"})
    return sorted(output, key=lambda r: (r["arrival_time"], r["camera_id"]))


def observations_from_legacy(directory: Path, frames: set[int]) -> list[dict]:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report.get("synthetic_noise", True):
        raise ValueError("Cannot import legacy synthetic observations into a no-noise experiment")
    rows = read_jsonl(directory / "observations.jsonl")
    return [{**row, "camera_id": row["source"].split("/")[1], "arrival_time": row["t_host"],
             "observation_kind": "unmodified_h2o_annotation"}
            for row in rows if row["frame_id"] in frames]


class FusionRunError(RuntimeError):
    """A stopped solver run with only fully processed earlier frames retained."""

    def __init__(self, message: str, predictions: list[dict], failure: dict):
        super().__init__(message)
        self.predictions = predictions
        self.failure = failure


async def run_fusion(observations: list[dict], variant: Variant, *, fps: float,
                     base_overrides: dict[str, Any] | None = None) -> tuple[list[dict], dict]:
    """Drive the exact production handlers sequentially, without wall-time sleeps.

    Accepts pose messages; annotation mode deliberately supplies labels and
    therefore measures consistency, not independent inference accuracy.
    Timing is handler/optimizer CPU wall time, not RGB-D or online latency.
    """
    cfg = build_config(variant, base_overrides)
    bus = EventBus()
    output = bus.subscribe(cfg.fusion.topic_fused)
    manager = FusionManager(bus, asyncio.get_running_loop(), cfg)
    manager.configure_runtime(offline_replay=True)
    events = sorted([r for r in observations if r["camera_id"] != variant.exclude_camera],
                    key=lambda r: (r["arrival_time"], r["camera_id"]))
    if variant.reverse_order:
        events = sorted(events, key=lambda r: r["camera_id"], reverse=True)
        events.sort(key=lambda r: r["arrival_time"])
    predictions: dict[int, dict] = {}
    compute_ms, accepted, rejected = [], 0, 0
    origin = round(observations[0]["frame_id"] - observations[0]["t_host"] * fps) if observations else 0
    started = time.perf_counter()
    for index, row in enumerate(events):
        frame_id = int(row["frame_id"])
        target = frame_id
        if variant.grouping == "separate":
            group = index
        elif variant.grouping == "arrival":
            group = origin + round(row["arrival_time"] * fps)
            target = group
        else:
            group = frame_id
        detection = DetectionResult(t_host=float(row["t_host"]), source=row["source"],
                                    pose_world=np.asarray(row["pose_world"]), score=float(row["score"]),
                                    detector=row.get("detector", "foundationpose"), object_name=row["object_name"], sync_group=group,
                                    capture_time=float(row.get("capture_time", row["t_host"])),
                                    covariance=row.get("covariance"), quality_metrics=row.get("quality_metrics"))
        tick = time.perf_counter()
        result = await manager._on_pose_message(detection)
        if result is None:
            rejected += 1
            compute_ms.append((time.perf_counter() - tick) * 1000)
            continue
        accepted += 1
        await manager._update_and_publish(*result)
        elapsed = (time.perf_counter() - tick) * 1000
        compute_ms.append(elapsed)
        if output.empty():
            # Do not score a provisional output from an incomplete frame as the
            # final estimate, or continue using a potentially corrupted solver.
            retained = [p for frame, p in predictions.items() if frame < target]
            raise FusionRunError(
                f"Fusion failed to publish {variant.name} frame {frame_id}; see log",
                retained,
                {"frame_id": frame_id, "target_frame_id": target, "source": row["source"],
                 "event_index": index, "n_input": len(events),
                 "n_processed_events": index + 1,
                 "policy": "stop_variant; retain_completed_prefix; failed_frame_and_suffix_missing"})
        fused = output.get_nowait()
        if not output.empty():
            raise RuntimeError("Unexpected multiple outputs for one observation")
        predictions[target] = {"frame_id": target, "object_name": row["object_name"],
                               "pose_world": fused["T"], "observation_count": fused["observation_count"],
                               "integrity": fused.get("integrity", {}),
                               "prediction_only": bool(fused.get("prediction_only", False)),
                               "covariance": fused.get("covariance"),
                               "active_sensors": fused.get("active_sensors", []),
                               "online_extrinsics": fused.get("online_extrinsics", {}),
                               "final_input_arrival_s": row["arrival_time"],
                               "capture_time_s": row["t_host"]}
    return list(predictions.values()), {
        "n_input": len(events), "n_accepted": accepted, "n_rejected": rejected,
        "elapsed_s": time.perf_counter() - started,
        "compute_ms_mean_per_observation": float(np.mean(compute_ms)) if compute_ms else None,
        "compute_ms_p95_per_observation": float(np.percentile(compute_ms, 95)) if compute_ms else None,
        "timing_scope": "Sequential production handler + optimizer + optional integrity; excludes RGB-D, I/O and networking",
    }


def run_baselines(observations: list[dict], cameras: list[str]) -> dict[str, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    methods = {name: [] for name in ["best_score", "mean_equal", "mean_markley_weighted"] +
               [f"sensor_{camera}" for camera in cameras]}
    for row in observations:
        grouped[row["frame_id"]].append(row)
        methods[f"sensor_{row['camera_id']}"].append(row)
    for frame_id, rows in grouped.items():
        methods["best_score"].append(max(rows, key=lambda r: r["score"]))
        poses = [np.asarray(r["pose_world"]) for r in rows]
        quats = [_rot_to_quat_wxyz(T[:3, :3]) for T in poses]
        for name, weights in [("mean_equal", np.ones(len(rows))),
                              ("mean_markley_weighted", np.array([r["score"] for r in rows]))]:
            T = np.eye(4)
            T[:3, 3] = np.average([pose[:3, 3] for pose in poses], axis=0, weights=weights)
            T[:3, :3] = _quat_wxyz_to_rot(_avg_quat_markley(quats, list(weights)))
            methods[name].append({"frame_id": frame_id, "object_name": rows[0]["object_name"],
                                  "pose_world": T.tolist(), "observation_count": len(rows)})
    return methods


def write_gzip(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def read_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_scores(directory: Path, predictions: list[dict], rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_gzip(directory / "poses.jsonl.gz", predictions)
    write_gzip(directory / "scores.jsonl.gz", rows)


async def run_case(
    observations: list[dict], gt: list[dict], cameras: list[str], variants: list[Variant],
    output: Path, *, fps: float, manifest: dict,
) -> dict:
    case_signature = fingerprint({"observations": observations, "ground_truth": gt,
                                  "variants": [asdict(v) for v in variants], "manifest": manifest})
    report_path = output / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("case_signature") != case_signature:
            raise ValueError(f"Resume input/config mismatch: {output}")
        for method, result in report["methods"].items():
            if result["status"] == "completed":
                for filename in ("poses.jsonl.gz", "scores.jsonl.gz"):
                    if not (output / method / filename).is_file():
                        raise ValueError(f"Completed report has missing {method}/{filename}: {output}")
        print(f"[SKIP] {output.name} validated complete result", flush=True)
        return report
    output.mkdir(parents=True, exist_ok=True)
    write_gzip(output / "observations.jsonl.gz", observations)
    write_gzip(output / "ground_truth.jsonl.gz", gt)
    report = {"case_signature": case_signature, "manifest": manifest,
              "observation_sha256": fingerprint(observations), "cameras": cameras,
              "n_targets": len(gt), "methods": {}}
    scored, target_ids = {}, {row["frame_id"] for row in gt}
    for name, predictions in run_baselines(observations, cameras).items():
        rows = score_poses(gt, predictions)
        save_scores(output / name, predictions, rows)
        scored[name] = rows
        report["methods"][name] = {"status": "completed", "metrics": summarize(rows)}
    for variant in variants:
        if variant.inactive_reason:
            report["methods"][variant.name] = {"status": "not_applicable", "reason": variant.inactive_reason}
            continue
        tick = time.perf_counter()
        with (output / f"{variant.name}.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            predictions, runtime = await run_fusion(observations, variant, fps=fps)
        # Arrival-time ablation can emit states after the dataset's final target.
        runtime["n_predictions_outside_targets"] = sum(r["frame_id"] not in target_ids for r in predictions)
        predictions = [r for r in predictions if r["frame_id"] in target_ids]
        rows = score_poses(gt, predictions)
        save_scores(output / variant.name, predictions, rows)
        scored[variant.name] = rows
        report["methods"][variant.name] = {"status": "completed", "variant": asdict(variant),
                                           "effective_fusion_config": asdict(build_config(variant).fusion),
                                           "metrics": summarize(rows), "runtime": runtime}
        print(f"  {variant.name}: {time.perf_counter() - tick:.2f}s, "
              f"P95={report['methods'][variant.name]['metrics']['translation_p95_mm']} mm", flush=True)
    if "full" in scored:
        report["paired_full_minus"] = {name: paired_delta(scored["full"], rows)
                                       for name, rows in scored.items() if name != "full"}
    for name, rows in scored.items():
        report["methods"][name]["per_object"] = {
            obj: summarize([r for r in rows if r["object_name"] == obj])
            for obj in sorted({r["object_name"] for r in rows})}
    write_json(report_path, report)  # Completion marker is written last.
    return report


def aggregate(output: Path) -> dict:
    pooled: dict[tuple[str, str], MetricAccumulator] = defaultdict(MetricAccumulator)
    per_take_deltas: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_take, runtimes = [], defaultdict(list)
    for path in sorted((output / "takes").glob("*/*/*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        meta = report["manifest"]
        for method, result in report["methods"].items():
            if result["status"] != "completed":
                continue
            rows = read_gzip(path.parent / method / "scores.jsonl.gz")
            if len(rows) != report["n_targets"]:
                raise ValueError(f"Incomplete scores: {path.parent / method}")
            pooled[meta["scenario"], method].add(rows)
            per_take.append({"take": meta["take"], "scenario": meta["scenario"], "seed": meta["seed"],
                             "method": method, **result["metrics"]})
            if result.get("runtime"):
                runtimes[meta["scenario"], method].append(result["runtime"])
            delta = report.get("paired_full_minus", {}).get(method, {}).get("mean_translation_delta_mm")
            if delta is not None:
                per_take_deltas[meta["scenario"], method][meta["take"]].append(delta)
    scenarios = {}
    for (scenario, method), accumulator in pooled.items():
        item = {"metrics": accumulator.result()}
        deltas = per_take_deltas.get((scenario, method), {})
        if deltas:
            # Cluster by take: seeds and neighboring frames are not independent trials.
            values = [float(np.mean(v)) for v in deltas.values()]
            item["paired_full_minus_translation_mean_mm"] = float(np.mean(values))
            item["paired_take_bootstrap_ci95_mm"] = bootstrap_mean_ci(values)
            item["n_paired_takes"] = len(values)
        timings = runtimes.get((scenario, method), [])
        if timings:
            n_input = sum(r["n_input"] for r in timings)
            item["runtime"] = {"total_elapsed_s": sum(r["elapsed_s"] for r in timings),
                               "n_input": n_input,
                               "compute_ms_mean_per_observation": sum(
                                   (r["compute_ms_mean_per_observation"] or 0) * r["n_input"] for r in timings
                               ) / n_input if n_input else None,
                               "n_rejected": sum(r["n_rejected"] for r in timings)}
        scenarios.setdefault(scenario, {})[method] = item
    raw_annotations = all(scenario == "annotations" for scenario in scenarios)
    summary = {"scope": "Unmodified annotation-fed fusion consistency; NOT independent tracking accuracy",
               "artificial_noise": False,
               "scenarios": scenarios, "completed_take_cases": len({(r['take'], r['scenario'], r['seed']) for r in per_take}),
               "public_paper_scores_comparable": False,
               "inactive_ablations": {v.name: v.inactive_reason for v in VARIANTS.values() if v.inactive_reason}}
    write_json(output / "summary.json", summary)
    if per_take:
        fields = [k for k in per_take[0] if k != "health_counts"]
        with (output / "per_take.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(per_take)
    text = "# H2O fusion ablations\n\n"
    text += ("Labels are supplied unchanged to every method: no artificial pose noise, dropout, delay or score ranking. "
             "This annotation-fed run measures consistency, NOT independent tracking accuracy. "
             "Single cameras never fall back to other sensors; averages use exactly the same capture frame. "
             "Late observations may refine historical states; these are offline final estimates, not deadline-limited "
             "online performance. Timings exclude RGB-D inference and network/I/O.\n\n")
    for scenario, methods in scenarios.items():
        text += f"## {scenario}\n\n" + metric_table({m: item["metrics"] for m, item in methods.items()})
        text += "\nPaired differences are full minus comparator (negative favours full), on shared output frames. "
        text += "95% intervals bootstrap takes after averaging seeds, not individual frames; fewer than two takes gives no interval.\n\n"
        for name, item in methods.items():
            if "paired_take_bootstrap_ci95_mm" in item:
                text += f"- {name}: {item['paired_full_minus_translation_mean_mm']:+.3f} mm; CI {item['paired_take_bootstrap_ci95_mm']}.\n"
        text += "\n"
    text += "## Inactive components\n\n" + "\n".join(
        f"- {name}: {reason}" for name, reason in summary["inactive_ablations"].items())
    text += "\n\n" + PUBLIC_COMPARISON
    (output / "summary.md").write_text(text, encoding="utf-8")
    return summary


def code_fingerprint() -> str:
    repo = Path(__file__).resolve().parents[2]
    files = [Path(__file__), Path(__file__).with_name("evaluation.py"), Path(__file__).with_name("benchmark.py"),
             Path(__file__).with_name("dataset.py"), Path(__file__).with_name("geometry.py"),
             Path(__file__).with_name("consistency.py"), repo / "managers/fusion_manager.py",
             repo / "managers/baseline_manager.py", repo / "config.py"]
    return fingerprint({str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject-root", type=Path, default=Path("data/h2o/subject1"))
    p.add_argument("--output", type=Path, help="New results_N directory; defaults to next results_N beside subject")
    p.add_argument("--resume", action="store_true", help="Validate and continue exactly the specified output")
    p.add_argument("--seeds", default="0", help="Single compatibility run id; annotations have no random seed")
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--variants", default=DEFAULT_VARIANTS)
    p.add_argument("--leave-one-out", action="store_true", help="Additionally omit each sensor in turn")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--max-takes", type=int, default=0)
    p.add_argument("--take", help="Only one take, e.g. h1/0")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--legacy-results", type=Path, help="Removed compatibility option; artificial-noise results are disabled")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.summarize_only:
        if not args.output or not (args.output / "manifest.json").is_file():
            raise ValueError("--summarize-only requires an existing --output experiment")
        aggregate(args.output)
        print(f"Summary: {args.output / 'summary.md'}")
        return
    if args.fps <= 0 or not np.isfinite(args.fps) or min(args.max_frames, args.max_takes) < 0:
        raise ValueError("fps must be positive and frame/take limits nonnegative")
    seeds = list(dict.fromkeys(int(v) for v in args.seeds.split(",")))
    if len(seeds) != 1:
        raise ValueError("No artificial randomness remains; provide one run id, not repeated seeds")
    scenarios = list(dict.fromkeys(args.scenarios.split(",")))
    if set(scenarios) - set(SCENARIOS):
        raise ValueError(f"Artificial scenarios removed. Supported scenarios: {SCENARIOS}")
    variants = [VARIANTS[name] for name in dict.fromkeys(args.variants.split(","))]
    if "full" not in [v.name for v in variants]:
        raise ValueError("--variants must include full for paired ablations")
    if args.legacy_results:
        raise ValueError("Legacy synthetic inputs are disabled; artificial-noise experiments were removed")
    subject = args.subject_root.resolve()
    takes = discover_takes(subject)
    if args.take:
        takes = [t for t in takes if t.relative_to(subject).as_posix() == args.take.replace("\\", "/")]
        if not takes:
            raise ValueError(f"Take not found: {args.take}")
    if args.max_takes:
        takes = takes[:args.max_takes]
    manifest = {"schema_version": 2, "subject_root": str(subject), "seeds": seeds, "scenarios": scenarios,
                "variants": [asdict(v) for v in variants], "max_frames": args.max_frames,
                "fps": args.fps, "takes": [str(t.relative_to(subject)) for t in takes],
                "leave_one_out": args.leave_one_out, "code_sha256": code_fingerprint(),
                "fusion_config": asdict(build_config(VARIANTS["full"]).fusion),
                "artificial_noise": False, "artificial_dropout": False, "artificial_delay": False,
                "score_policy": "Uniform 1.0 for annotations; not estimated confidence",
                "scenario_definitions": {"annotations": "Unmodified labels, no artificial perturbations"}}
    signature = fingerprint(manifest)
    if args.dry_run:
        print(json.dumps({"takes": len(takes), "cases": len(takes) * len(seeds) * len(scenarios),
                          "active_variants": [v.name for v in variants if not v.inactive_reason],
                          "output": str(args.output) if args.output else f"next results_N under {subject.parent}",
                          "manifest": manifest}, indent=2))
        return
    if args.resume and not args.output:
        raise ValueError("--resume requires --output")
    output = args.output.resolve() if args.output else allocate_results(subject.parent)
    if args.resume:
        saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            raise ValueError("Cannot resume: code, settings, or selected takes changed; start a new results_N")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Output is not empty; use --resume or a new results_N: {output}")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "manifest.json", {"signature": signature, **manifest})
    print(f"Results: {output}", flush=True)
    for take_index, take in enumerate(takes):
        take_key = f"{subject.name}/{take.relative_to(subject).as_posix()}"
        run_name = "_".join(take.relative_to(subject).parts)
        print(f"[TAKE {take_index + 1}/{len(takes)}] {take_key}", flush=True)
        gt, clean, cameras = load_take(take, args.max_frames, args.fps)
        selected = variants + ([Variant(f"without_{cam}", f"Exclude {cam}", exclude_camera=cam)
                                for cam in cameras] if args.leave_one_out else [])
        for scenario in scenarios:
            for seed in seeds:
                observations = generate_observations(clean, take_key=take_key, seed=seed, scenario=scenario)
                meta = {"take": take_key, "scenario": scenario, "seed": seed, "suite_signature": signature}
                case_dir = output / "takes" / run_name / scenario / f"seed_{seed}"
                print(f"[CASE] {run_name}/{scenario}/seed_{seed}: {len(gt)} frames, {len(observations)} observations", flush=True)
                asyncio.run(run_case(observations, gt, cameras, selected, case_dir, fps=args.fps, manifest=meta))
    aggregate(output)
    print(f"Completed. Report: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
