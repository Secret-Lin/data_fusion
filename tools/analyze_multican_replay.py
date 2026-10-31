from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def nearest_deltas(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(reference, query)
    hi = np.clip(indices, 0, len(reference) - 1)
    lo = np.clip(indices - 1, 0, len(reference) - 1)
    return np.minimum(np.abs(query - reference[lo]), np.abs(query - reference[hi]))


def describe(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {}
    return {
        "min_ms": float(np.min(values) * 1000.0),
        "p50_ms": float(np.percentile(values, 50) * 1000.0),
        "p95_ms": float(np.percentile(values, 95) * 1000.0),
        "max_ms": float(np.max(values) * 1000.0),
        "within_15ms_fraction": float(np.mean(values <= 0.015)),
        "within_40ms_fraction": float(np.mean(values <= 0.040)),
    }


def rotation_delta_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rigid_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path (defaults to <run-dir>/analysis_report.json)",
    )
    parser.add_argument(
        "--corrected-fixed-extrinsics",
        type=Path,
        help="Apply candidate fixed-camera extrinsic corrections to recorded poses",
    )
    parser.add_argument(
        "--corrected-quest-extrinsic",
        type=Path,
        help="Apply a candidate Quest tracking-world correction to recorded poses",
    )
    args = parser.parse_args()

    pose_corrections: dict[str, np.ndarray] = {}
    if args.corrected_fixed_extrinsics is not None:
        original = json.loads(
            (args.session / "tag_cube_extrinsics.json").read_text(encoding="utf-8")
        )["T_world_cam"]
        corrected = json.loads(
            args.corrected_fixed_extrinsics.read_text(encoding="utf-8")
        )["T_world_cam"]
        for serial in sorted(set(original) & set(corrected)):
            old = np.asarray(original[serial], dtype=float).reshape(4, 4)
            new = np.asarray(corrected[serial], dtype=float).reshape(4, 4)
            pose_corrections[f"realsense/{serial}"] = new @ np.linalg.inv(old)
    if args.corrected_quest_extrinsic is not None:
        original = json.loads(
            (args.session / "tag_cube_pose.json").read_text(encoding="utf-8")
        )["T_world_tracking"]
        corrected = json.loads(
            args.corrected_quest_extrinsic.read_text(encoding="utf-8")
        )["T_world_tracking"]
        old = np.asarray(original, dtype=float).reshape(4, 4)
        new = np.asarray(corrected, dtype=float).reshape(4, 4)
        pose_corrections["quest/quest"] = new @ np.linalg.inv(old)

    timestamps: dict[str, np.ndarray] = {}
    for path in sorted((args.session / "rgbd").glob("*.h5")):
        with h5py.File(path, "r") as handle:
            timestamps[f"realsense/{path.stem}"] = np.asarray(handle["t_host"][:], dtype=float)
    for path in sorted((args.session / "quest").glob("*.h5")):
        with h5py.File(path, "r") as handle:
            timestamps[f"quest/{path.stem}"] = np.asarray(handle["t_host"][:], dtype=float)

    timing = {
        name: {
            "frames": int(len(values)),
            "start": float(values[0]),
            "end": float(values[-1]),
            "duration_s": float(values[-1] - values[0]),
            "effective_fps": float((len(values) - 1) / (values[-1] - values[0])),
        }
        for name, values in timestamps.items()
    }
    pairwise: dict[str, Any] = {}
    names = sorted(timestamps)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            pairwise[f"{left} -> {right}"] = describe(
                nearest_deltas(timestamps[right], timestamps[left])
            )

    detections = load_jsonl(args.run_dir / "tracking_detections.jsonl")
    detection_times: dict[tuple[str, str], list[float]] = defaultdict(list)
    detections_by_group: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = defaultdict(list)
    detection_lookup: dict[tuple[str, str, int], np.ndarray] = {}
    for row in detections:
        source = str(row["source"])
        sensor, object_name = source.split("/foundationpose/", 1)
        detection_times[(sensor, object_name)].append(float(row["t_host"]))
        sync_group = row.get("sync_group")
        if sync_group is not None:
            pose = np.asarray(row["pose_world"], dtype=float).reshape(4, 4)
            correction = pose_corrections.get(sensor)
            if correction is not None:
                pose = correction @ pose
            detection_lookup[(sensor, object_name, int(sync_group))] = pose
            detections_by_group[(object_name, int(sync_group))].append(
                (sensor, pose)
            )

    cross_sensor_deltas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"translation": [], "rotation": []}
    )
    pair_sensor_deltas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"translation": [], "rotation": []}
    )
    multi_sensor_groups: Counter[str] = Counter()
    gate_pass_groups: Counter[str] = Counter()
    translation_gate_pass_groups: Counter[str] = Counter()
    for (object_name, _group), observations in detections_by_group.items():
        if len(observations) < 2:
            continue
        multi_sensor_groups[object_name] += 1
        group_translation: list[float] = []
        group_rotation: list[float] = []
        for left_index in range(len(observations)):
            for right_index in range(left_index + 1, len(observations)):
                left = observations[left_index][1]
                right = observations[right_index][1]
                translation_delta = float(
                    np.linalg.norm(left[:3, 3] - right[:3, 3])
                )
                rotation_delta = rotation_delta_deg(left, right)
                group_translation.append(translation_delta)
                group_rotation.append(rotation_delta)
                left_sensor = observations[left_index][0]
                right_sensor = observations[right_index][0]
                sensor_pair = " <-> ".join(sorted((left_sensor, right_sensor)))
                pair_key = f"{object_name}: {sensor_pair}"
                pair_sensor_deltas[pair_key]["translation"].append(
                    translation_delta
                )
                pair_sensor_deltas[pair_key]["rotation"].append(rotation_delta)
        cross_sensor_deltas[object_name]["translation"].extend(group_translation)
        cross_sensor_deltas[object_name]["rotation"].extend(group_rotation)
        if max(group_translation) <= 0.15:
            translation_gate_pass_groups[object_name] += 1
        if max(group_translation) <= 0.15 and max(group_rotation) <= 45.0:
            gate_pass_groups[object_name] += 1

    fused = load_jsonl(args.run_dir / "pose_fused.jsonl")
    observation_counts: dict[str, Counter[int]] = defaultdict(Counter)
    sync_group_rows: dict[tuple[str, int], list[int]] = defaultdict(list)
    for wrapper in fused:
        row = wrapper.get("msg", wrapper)
        object_name = str(row.get("object_name"))
        count = int(row.get("observation_count", 0))
        observation_counts[object_name][count] += 1
        sync_group = row.get("sync_group")
        if sync_group is not None:
            sync_group_rows[(object_name, int(sync_group))].append(count)

    final_count_per_group: dict[str, Counter[int]] = defaultdict(Counter)
    for (object_name, _group), counts in sync_group_rows.items():
        final_count_per_group[object_name][max(counts)] += 1

    reference_sensor = "realsense/134722070227"
    extrinsic_alignment: dict[str, Any] = {}
    other_sensors = sorted({key[0] for key in detection_lookup} - {reference_sensor})
    for sensor in other_sensors:
        source_points: list[np.ndarray] = []
        target_points: list[np.ndarray] = []
        for (candidate_sensor, object_name, group), pose in detection_lookup.items():
            if candidate_sensor != sensor or group > 300:
                continue
            reference = detection_lookup.get((reference_sensor, object_name, group))
            if reference is None:
                continue
            source_points.append(pose[:3, 3])
            target_points.append(reference[:3, 3])
        if len(source_points) < 3:
            continue
        source_array = np.asarray(source_points)
        target_array = np.asarray(target_points)
        rotation, translation = rigid_align(source_array, target_array)
        before = np.linalg.norm(source_array - target_array, axis=1)
        after = np.linalg.norm(
            (rotation @ source_array.T).T + translation - target_array, axis=1
        )
        extrinsic_alignment[sensor] = {
            "reference": reference_sensor,
            "matched_points_first_300_groups": int(len(source_array)),
            "before_alignment_median_m": float(np.median(before)),
            "after_single_rigid_alignment_median_m": float(np.median(after)),
            "after_single_rigid_alignment_p95_m": float(np.percentile(after, 95)),
            "required_rotation_deg": rotation_delta_deg(np.eye(4), np.block([
                [rotation, np.zeros((3, 1))],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ])),
            "required_translation_m": translation.tolist(),
        }

    report = {
        "source_timing": timing,
        "pairwise_nearest_capture_delta": pairwise,
        "detection_counts": {
            f"{sensor}/{object_name}": len(values)
            for (sensor, object_name), values in sorted(detection_times.items())
        },
        "cross_sensor_same_sync_group": {
            name: {
                "multi_sensor_groups": int(multi_sensor_groups[name]),
                "translation_m": {
                    "p50": float(np.percentile(values["translation"], 50)),
                    "p95": float(np.percentile(values["translation"], 95)),
                    "max": float(np.max(values["translation"])),
                },
                "rotation_deg": {
                    "p50": float(np.percentile(values["rotation"], 50)),
                    "p95": float(np.percentile(values["rotation"], 95)),
                    "max": float(np.max(values["rotation"])),
                },
                "translation_gate_pass_groups": int(translation_gate_pass_groups[name]),
                "full_gate_pass_groups": int(gate_pass_groups[name]),
            }
            for name, values in sorted(cross_sensor_deltas.items())
            if values["translation"]
        },
        "pairwise_sensor_same_sync_group": {
            name: {
                "groups": len(values["translation"]),
                "translation_m": {
                    "p50": float(np.percentile(values["translation"], 50)),
                    "p95": float(np.percentile(values["translation"], 95)),
                    "max": float(np.max(values["translation"])),
                },
                "rotation_deg": {
                    "p50": float(np.percentile(values["rotation"], 50)),
                    "p95": float(np.percentile(values["rotation"], 95)),
                    "max": float(np.max(values["rotation"])),
                },
            }
            for name, values in sorted(pair_sensor_deltas.items())
            if values["translation"]
        },
        "diagnostic_single_rigid_world_alignment": extrinsic_alignment,
        "fused_emitted_row_observation_counts": {
            name: dict(sorted(counts.items())) for name, counts in observation_counts.items()
        },
        "final_observation_count_per_sync_group": {
            name: dict(sorted(counts.items())) for name, counts in final_count_per_group.items()
        },
    }
    output_path = args.output or args.run_dir / "analysis_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
