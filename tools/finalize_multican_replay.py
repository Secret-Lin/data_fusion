from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _safe(value: str) -> str:
    return "_".join(part for part in str(value).replace("\\", "/").split("/") if part)


def _tracking_timestamps_by_sensor(run_dir: Path) -> dict[str, np.ndarray]:
    grouped: dict[str, set[float]] = defaultdict(set)
    for row in _load_jsonl(run_dir / "tracking_detections.jsonl"):
        source = str(row.get("source", ""))
        sensor = source.split("/foundationpose/", 1)[0]
        if sensor:
            grouped[sensor].add(float(row.get("t_host", 0.0)))
    return {
        sensor: np.asarray(sorted(values), dtype=np.float64)
        for sensor, values in grouped.items()
        if values
    }


def _frame_index_at_time(timestamps: np.ndarray, timestamp: float) -> int:
    """Return the newest frame captured no later than ``timestamp``."""
    return int(np.searchsorted(timestamps, timestamp, side="right") - 1)


def _pose_columns(transform: Any) -> dict[str, float]:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    output = {
        "x_m": float(matrix[0, 3]),
        "y_m": float(matrix[1, 3]),
        "z_m": float(matrix[2, 3]),
    }
    output.update(
        {
            f"T_{row}{column}": float(matrix[row, column])
            for row in range(4)
            for column in range(4)
        }
    )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_pose_csvs(run_dir: Path) -> dict[str, Any]:
    tracking = _load_jsonl(run_dir / "tracking_detections.jsonl")
    fused_wrapped = _load_jsonl(run_dir / "pose_fused.jsonl")
    tracking_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in tracking:
        source = str(row.get("source", ""))
        object_name = str(row.get("object_name") or source.rsplit("/", 1)[-1])
        sensor = source.split("/foundationpose/", 1)[0]
        output = {
            "t_host": float(row.get("t_host", 0.0)),
            "sync_group": row.get("sync_group"),
            "sensor": sensor,
            "source": source,
            "object_name": object_name,
            "score": row.get("score"),
            **_pose_columns(row["pose_world"]),
        }
        tracking_groups[(sensor, object_name)].append(output)

    fused_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for wrapper in fused_wrapped:
        message = wrapper.get("msg", wrapper)
        if not isinstance(message, dict) or "T" not in message:
            continue
        object_name = str(message.get("object_name", "__default__"))
        fused_groups[object_name].append(
            {
                "t_capture": float(message.get("t_capture", 0.0)),
                "t_publish": float(message.get("t_publish", wrapper.get("t_host", 0.0))),
                "sync_group": message.get("sync_group"),
                "step": message.get("step"),
                "observation_count": message.get("observation_count"),
                "object_name": object_name,
                **_pose_columns(message["T"]),
            }
        )

    for (sensor, object_name), rows in tracking_groups.items():
        rows.sort(key=lambda row: (row["t_host"], row.get("sync_group") or -1))
        _write_csv(
            run_dir / f"pose_{_safe(sensor)}__{_safe(object_name)}.csv", rows
        )
    all_fused: list[dict[str, Any]] = []
    for object_name, rows in fused_groups.items():
        rows.sort(key=lambda row: (row["t_capture"], row.get("step") or -1))
        _write_csv(run_dir / f"pose_fused__{_safe(object_name)}.csv", rows)
        all_fused.extend(rows)
    all_fused.sort(key=lambda row: (row["t_capture"], row["object_name"]))
    _write_csv(run_dir / "pose_fused_all_objects.csv", all_fused)
    return {
        "tracking_counts": {
            f"{sensor}/{object_name}": len(rows)
            for (sensor, object_name), rows in sorted(tracking_groups.items())
        },
        "fused_counts": {
            object_name: len(rows) for object_name, rows in sorted(fused_groups.items())
        },
        "fused_rows": fused_groups,
    }


def _fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, int(round(frame.shape[1] * scale))), max(1, int(round(frame.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def make_combined_video(
    run_dir: Path, fused_groups: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    timestamps_by_sensor = _tracking_timestamps_by_sensor(run_dir)
    paths = sorted(
        path
        for path in run_dir.glob("vis_*.mp4")
        if not path.name.startswith("vis_all_sensors_fusion")
    )
    if not paths:
        raise RuntimeError(f"No per-sensor visualization MP4 files in {run_dir}")
    streams: list[dict[str, Any]] = []
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0.0 or frames <= 0:
            capture.release()
            raise RuntimeError(f"Invalid video metadata for {path}: fps={fps}, frames={frames}")
        safe_source = path.stem.removeprefix("vis_")
        matching_sources = [
            source for source in timestamps_by_sensor if _safe(source) == safe_source
        ]
        if len(matching_sources) != 1:
            capture.release()
            raise RuntimeError(
                f"Cannot uniquely map {path.name} to tracking timestamps; "
                f"matches={matching_sources}"
            )
        source = matching_sources[0]
        timestamps = timestamps_by_sensor[source]
        if len(timestamps) < frames:
            capture.release()
            raise RuntimeError(
                f"{path.name} has {frames} frames but only {len(timestamps)} "
                f"unique tracking timestamps for {source}"
            )
        # A replay can be interrupted while processing the final multi-object
        # frame. In that case the video contains the complete prefix only.
        timestamps = timestamps[:frames]
        streams.append(
            {
                "path": path,
                "source": source,
                "capture": capture,
                "fps": fps,
                "frames": frames,
                "timestamps": timestamps,
                "last_index": -1,
                "frame": None,
            }
        )
    output_fps = max(stream["fps"] for stream in streams)
    timeline_start = min(float(stream["timestamps"][0]) for stream in streams)
    timeline_end = max(float(stream["timestamps"][-1]) for stream in streams)
    duration = max(0.0, timeline_end - timeline_start)
    # Include the final captured frame even when the container FPS is rounded
    # (for example 29.970149 Hz is stored as 29.97 Hz by OpenCV).
    output_frames = max(1, int(math.ceil(duration * output_fps - 1e-9)) + 1)
    output_path = run_dir / "vis_all_sensors_fusion_timestamp_aligned.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (1920, 1080),
    )
    if not writer.isOpened():
        for stream in streams:
            stream["capture"].release()
        raise RuntimeError(f"Cannot create {output_path}")

    fused_sorted = {
        name: sorted(rows, key=lambda row: row["t_capture"])
        for name, rows in fused_groups.items()
    }
    pointers = {name: -1 for name in fused_sorted}
    colors = ((0, 255, 255), (255, 80, 255), (80, 220, 80), (255, 180, 40))
    try:
        for output_index in range(output_frames):
            current_time = timeline_start + output_index / output_fps
            cells: list[np.ndarray] = []
            for stream in streams[:3]:
                wanted = _frame_index_at_time(stream["timestamps"], current_time)
                while stream["last_index"] < wanted:
                    ok, frame = stream["capture"].read()
                    if not ok:
                        break
                    stream["frame"] = frame
                    stream["last_index"] += 1
                frame = stream["frame"]
                if frame is None:
                    frame = np.zeros((540, 960, 3), dtype=np.uint8)
                cell = _fit(frame, 960, 540)
                cv2.putText(
                    cell,
                    stream["path"].stem.replace("vis_", ""),
                    (18, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cells.append(cell)
            while len(cells) < 3:
                cells.append(np.zeros((540, 960, 3), dtype=np.uint8))

            panel = np.zeros((540, 960, 3), dtype=np.uint8)
            cv2.putText(panel, "MULTI-OBJECT FUSION (tag-cube world, Y-down)", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            for object_index, (name, rows) in enumerate(sorted(fused_sorted.items())):
                pointer = pointers[name]
                while pointer + 1 < len(rows) and rows[pointer + 1]["t_capture"] <= current_time:
                    pointer += 1
                pointers[name] = pointer
                if not rows or pointer < 0:
                    continue
                row = rows[pointer]
                color = colors[object_index % len(colors)]
                cv2.putText(panel, name, (35, 120 + object_index * 110), cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 3, cv2.LINE_AA)
                text = f"xyz=({row['x_m']:+.3f}, {row['y_m']:+.3f}, {row['z_m']:+.3f}) m"
                cv2.putText(panel, text, (250, 115 + object_index * 110), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
                cv2.putText(panel, f"sync={row.get('sync_group')} observations={row.get('observation_count')}", (250, 150 + object_index * 110), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 210, 210), 1, cv2.LINE_AA)
            cv2.putText(panel, f"timeline={output_index / output_fps:7.3f}s | fps={output_fps:.6f} | aligned by t_host", (24, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2, cv2.LINE_AA)
            canvas = np.vstack((np.hstack((cells[0], cells[1])), np.hstack((cells[2], panel))))
            writer.write(canvas)
    finally:
        writer.release()
        for stream in streams:
            stream["capture"].release()
    return {
        "path": output_path.name,
        "fps": output_fps,
        "frames": output_frames,
        "duration_s": output_frames / output_fps,
        "timeline_start_t_host": timeline_start,
        "timeline_end_t_host": timeline_end,
        "sources": {
            stream["path"].name: {
                "source": stream["source"],
                "fps": stream["fps"],
                "frames": stream["frames"],
                "first_t_host": float(stream["timestamps"][0]),
                "last_t_host": float(stream["timestamps"][-1]),
                "duration_s": float(
                    stream["timestamps"][-1] - stream["timestamps"][0]
                ),
            }
            for stream in streams
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export multi-object pose CSVs and a native-FPS replay overview"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    pose_summary = export_pose_csvs(run_dir)
    video_summary = make_combined_video(run_dir, pose_summary.pop("fused_rows"))
    summary = {
        "world_frame": "tag_cube",
        "coordinate_system": "y_down",
        "objects": sorted(pose_summary["fused_counts"]),
        **pose_summary,
        "video": video_summary,
    }
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
