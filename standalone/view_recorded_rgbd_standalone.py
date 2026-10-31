#!/usr/bin/env python3
"""
Compact HDF5 RGBD viewer for RealSense + Quest recordings.

This version reduces PowerShell output and adds:
  --compact          print short summaries/checks/stats
  --report-file      save text report to file
  --max-gaps         number of discontinuity examples to print
  --list-datasets    print full dataset list only when needed

It still supports:
  /color + /depth playback
  RealSense fields: color_frame_number, depth_frame_number, timestamps
  Quest fields: quest_frame_id, quest_timestamp_ms, fx/fy/cx/cy, pose, detections
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compact RealSense/Quest HDF5 RGBD viewer")
    p.add_argument("--input", type=str, required=True)

    p.add_argument("--summary", action="store_true")
    p.add_argument("--check-drops", action="store_true")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--play", action="store_true")

    p.add_argument("--compact", action="store_true", help="Print compact output")
    p.add_argument("--report-file", type=str, default=None, help="Save printed report to a text file")
    p.add_argument("--list-datasets", action="store_true", help="Print full dataset list")
    p.add_argument("--max-gaps", type=int, default=3, help="Max discontinuity examples to print")

    p.add_argument("--play-fps", type=float, default=30.0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--depth-vis-max-mm", type=float, default=3000.0)
    p.add_argument(
        "--depth-colormap",
        type=str,
        default="turbo",
        choices=["turbo", "jet", "bone", "gray"],
    )
    p.add_argument("--show-lowres-depth", action="store_true")

    p.add_argument("--export-frames", nargs="*", type=int, default=None)
    p.add_argument("--export-dir", type=str, default=None)
    p.add_argument("--export-rgb-png", action="store_true")
    p.add_argument("--export-depth-png", action="store_true")
    p.add_argument("--export-lowres-depth-png", action="store_true")
    p.add_argument("--export-depth-npy", action="store_true")
    p.add_argument("--export-meta-json", action="store_true")

    return p


def is_dataset(f: h5py.File, name: str) -> bool:
    return name in f and isinstance(f[name], h5py.Dataset)


def load_optional_dataset(f: h5py.File, name: str) -> np.ndarray | None:
    if is_dataset(f, name):
        return f[name][:]
    return None


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def one_dataset_value(f: h5py.File, name: str, idx: int) -> Any:
    if not is_dataset(f, name):
        return None
    return to_python_scalar(f[name][idx])


def decode_json_string(value: Any) -> Any:
    value = to_python_scalar(value)
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    if value == "":
        return []
    try:
        return json.loads(value)
    except Exception:
        return value


def parse_json_attr(attrs: h5py.AttributeManager, key: str, default: Any) -> Any:
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    try:
        return json.loads(value)
    except Exception:
        return default


def infer_kind(f: h5py.File) -> str:
    if is_dataset(f, "quest_frame_id") or is_dataset(f, "quest_timestamp_ms"):
        return "quest"
    if is_dataset(f, "color_frame_number") or "serial" in f.attrs:
        return "realsense"
    return "generic_rgbd"


def frame_count(f: h5py.File) -> int:
    if is_dataset(f, "color"):
        return int(f["color"].shape[0])
    if is_dataset(f, "depth"):
        return int(f["depth"].shape[0])
    return 0


def dataset_shape_str(f: h5py.File, name: str) -> str:
    if not is_dataset(f, name):
        return "missing"
    ds = f[name]
    return f"{tuple(ds.shape)} {ds.dtype}"


def print_summary(f: h5py.File, compact: bool, list_datasets: bool) -> None:
    kind = infer_kind(f)
    n = frame_count(f)

    print("=" * 72)
    print("HDF5 RGBD Summary")
    print("=" * 72)
    print(f"File:   {f.filename}")
    print(f"Kind:   {kind}")
    print(f"Frames: {n}")
    print(f"Color:  {dataset_shape_str(f, 'color')}")
    print(f"Depth:  {dataset_shape_str(f, 'depth')}")

    if is_dataset(f, "depth_lowres"):
        print(f"Depth lowres: {dataset_shape_str(f, 'depth_lowres')}")

    if kind == "realsense":
        print(f"Serial: {f.attrs.get('serial', 'N/A')}")
        print(f"Depth scale m: {f.attrs.get('depth_scale_m', 'N/A')}")
        print(f"FPS attr: {f.attrs.get('fps', 'N/A')}")
        k_attr = parse_json_attr(f.attrs, "K_json", None)
        if k_attr is not None:
            print(f"K: {k_attr}")

    if kind == "quest":
        fields = [
            "quest_frame_id",
            "quest_timestamp_ms",
            "fx", "fy", "cx", "cy",
            "camera_position",
            "camera_rotation_xyzw",
            "depth_mode",
            "detections_json",
            "pose_objects_json",
        ]
        present = [x for x in fields if is_dataset(f, x)]
        print(f"Quest fields: {', '.join(present) if present else 'none'}")
        if all(is_dataset(f, x) for x in ["fx", "fy", "cx", "cy"]) and n > 0:
            print(
                "K first: "
                f"fx={float(f['fx'][0]):.2f}, fy={float(f['fy'][0]):.2f}, "
                f"cx={float(f['cx'][0]):.2f}, cy={float(f['cy'][0]):.2f}"
            )

    if list_datasets or not compact:
        print("\nDatasets:")
        for k in sorted(f.keys()):
            ds = f[k]
            if isinstance(ds, h5py.Dataset):
                print(
                    f"  - {k}: shape={ds.shape}, dtype={ds.dtype}, "
                    f"chunks={ds.chunks}, compression={ds.compression}"
                )

    print("=" * 72)


def frame_number_report(frame_numbers: np.ndarray | None, name: str, compact: bool, max_gaps: int) -> dict[str, Any]:
    if frame_numbers is None or len(frame_numbers) == 0:
        out = {"name": name, "available": False}
        print(f"{name}: no data")
        return out

    frame_numbers = np.asarray(frame_numbers, dtype=np.int64)
    diffs = np.diff(frame_numbers)
    bad = np.where(diffs != 1)[0]
    total_missing = int(np.sum(np.clip(diffs - 1, 0, None)))

    out = {
        "name": name,
        "available": True,
        "count": int(len(frame_numbers)),
        "first": int(frame_numbers[0]),
        "last": int(frame_numbers[-1]),
        "discontinuities": int(len(bad)),
        "estimated_missing": int(total_missing),
        "first_gap_examples": [],
    }

    for i in bad[:max(0, max_gaps)]:
        out["first_gap_examples"].append(
            {
                "idx": int(i),
                "next_idx": int(i + 1),
                "frame": int(frame_numbers[i]),
                "next_frame": int(frame_numbers[i + 1]),
                "diff": int(diffs[i]),
            }
        )

    if compact:
        print(
            f"{name}: count={out['count']} first={out['first']} last={out['last']} "
            f"gaps={out['discontinuities']} missing={out['estimated_missing']}"
        )
    else:
        print(f"\n{name} continuity check:")
        print(f"  frames stored: {out['count']}")
        print(f"  first frame number: {out['first']}")
        print(f"  last frame number:  {out['last']}")
        print(f"  discontinuity count: {out['discontinuities']}")
        print(f"  estimated missing frames: {out['estimated_missing']}")
        if out["first_gap_examples"]:
            print("  first discontinuities:")
            for e in out["first_gap_examples"]:
                print(
                    f"    idx={e['idx']} -> {e['next_idx']}, "
                    f"frame {e['frame']} to {e['next_frame']}, diff={e['diff']}"
                )

    return out


def print_drop_checks(f: h5py.File, compact: bool, max_gaps: int) -> None:
    print("=" * 72)
    print("Frame continuity")
    print("=" * 72)

    items = [
        ("Local frame_index", "frame_index"),
        ("RealSense color frame", "color_frame_number"),
        ("RealSense depth frame", "depth_frame_number"),
        ("Quest frame ID", "quest_frame_id"),
    ]
    for label, ds_name in items:
        frame_number_report(load_optional_dataset(f, ds_name), label, compact, max_gaps)

    if is_dataset(f, "dropped_count"):
        print(f"Writer queue dropped_count: {int(f['dropped_count'][0])}")

    print("=" * 72)


def timestamp_stats(ts: np.ndarray | None) -> dict[str, Any] | None:
    if ts is None or len(ts) == 0:
        return None
    ts = np.asarray(ts, dtype=np.float64)
    d = np.diff(ts)
    out = {
        "count": int(len(ts)),
        "first": float(ts[0]),
        "last": float(ts[-1]),
    }
    if len(d) > 0:
        out.update(
            {
                "delta_mean": float(np.mean(d)),
                "delta_std": float(np.std(d)),
                "delta_min": float(np.min(d)),
                "delta_p50": float(np.percentile(d, 50)),
                "delta_p95": float(np.percentile(d, 95)),
                "delta_max": float(np.max(d)),
            }
        )
    return out


def print_one_ts(name: str, ts: np.ndarray | None, unit: str, compact: bool) -> None:
    s = timestamp_stats(ts)
    if s is None:
        if not compact:
            print(f"\n{name}: no data")
        return

    if compact:
        if "delta_mean" in s:
            print(
                f"{name}: n={s['count']} "
                f"mean={s['delta_mean']:.3f}{unit} "
                f"std={s['delta_std']:.3f}{unit} "
                f"p95={s['delta_p95']:.3f}{unit} "
                f"max={s['delta_max']:.3f}{unit}"
            )
        else:
            print(f"{name}: n={s['count']}")
    else:
        print(f"\n{name} statistics:")
        print(f"  count: {s['count']}")
        print(f"  first: {s['first']:.6f}")
        print(f"  last:  {s['last']:.6f}")
        if "delta_mean" in s:
            print(f"  delta mean: {s['delta_mean']:.6f} {unit}")
            print(f"  delta std:  {s['delta_std']:.6f} {unit}")
            print(f"  delta min:  {s['delta_min']:.6f} {unit}")
            print(f"  delta p50:  {s['delta_p50']:.6f} {unit}")
            print(f"  delta p95:  {s['delta_p95']:.6f} {unit}")
            print(f"  delta max:  {s['delta_max']:.6f} {unit}")


def print_timestamp_stats(f: h5py.File, compact: bool) -> None:
    print("=" * 72)
    print("Timestamp stats")
    print("=" * 72)

    fields = [
        ("Host t_host", "t_host", "s"),
        ("Color HW", "color_hw_timestamp_ms", "ms"),
        ("Depth HW", "depth_hw_timestamp_ms", "ms"),
        ("Color sensor", "color_sensor_ts_ms", "ms"),
        ("Depth sensor", "depth_sensor_ts_ms", "ms"),
        ("Color backend", "color_backend_ts_ms", "ms"),
        ("Depth backend", "depth_backend_ts_ms", "ms"),
        ("Quest timestamp", "quest_timestamp_ms", "ms"),
    ]

    for label, name, unit in fields:
        print_one_ts(label, load_optional_dataset(f, name), unit, compact)

    color_hw = load_optional_dataset(f, "color_hw_timestamp_ms")
    depth_hw = load_optional_dataset(f, "depth_hw_timestamp_ms")
    if color_hw is not None and depth_hw is not None and len(color_hw) == len(depth_hw):
        diff = np.asarray(color_hw, dtype=np.float64) - np.asarray(depth_hw, dtype=np.float64)
        print(
            f"ColorHW-DepthHW: mean={float(np.mean(diff)):.3f}ms "
            f"std={float(np.std(diff)):.3f}ms "
            f"min={float(np.min(diff)):.3f}ms "
            f"max={float(np.max(diff)):.3f}ms"
        )

    if all(is_dataset(f, name) for name in ["fx", "fy", "cx", "cy"]):
        vals = []
        for name in ["fx", "fy", "cx", "cy"]:
            arr = np.asarray(f[name][:], dtype=np.float64)
            vals.append(f"{name}={float(np.mean(arr)):.2f}±{float(np.std(arr)):.4f}")
        print("Quest intrinsics:", ", ".join(vals))

    print("=" * 72)


def get_depth_vis(depth_u16: np.ndarray, max_mm: float, colormap: str) -> np.ndarray:
    depth_f = depth_u16.astype(np.float32)
    valid = depth_f > 0
    norm = np.clip(depth_f / max(float(max_mm), 1.0), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)

    if colormap == "gray":
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif colormap == "jet":
        out = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    elif colormap == "bone":
        out = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
    else:
        out = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    out[~valid] = 0
    return out


def overlay_text(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    y = 28
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y += 24
    return out


def playback(
    f: h5py.File,
    start: int,
    end: int,
    step: int,
    play_fps: float,
    depth_vis_max_mm: float,
    depth_colormap: str,
    show_lowres_depth: bool,
) -> None:
    if not is_dataset(f, "color") or not is_dataset(f, "depth"):
        print("[!] File must contain /color and /depth for playback")
        return

    color_ds = f["color"]
    depth_ds = f["depth"]
    lowres_ds = f["depth_lowres"] if is_dataset(f, "depth_lowres") else None

    n = int(color_ds.shape[0])
    end = n if end < 0 else min(end, n)
    start = max(start, 0)
    step = max(step, 1)
    delay = max(int(round(1000.0 / max(play_fps, 0.1))), 1)

    print(f"Playback: start={start}, end={end}, step={step}, fps={play_fps}")
    print("Controls: q/Esc quit, space pause, s save preview PNGs")

    paused = False
    i = start
    current_i = start

    while i < end:
        if not paused:
            current_i = i
            color = color_ds[i]
            depth = depth_ds[i]
            depth_vis = get_depth_vis(depth, depth_vis_max_mm, depth_colormap)

            if depth_vis.shape[:2] != color.shape[:2]:
                depth_vis = cv2.resize(depth_vis, (int(color.shape[1]), int(color.shape[0])), interpolation=cv2.INTER_NEAREST)

            lines = [f"idx {i}/{n-1}"]
            if is_dataset(f, "frame_index"):
                lines.append(f"frame_index {int(f['frame_index'][i])}")
            if is_dataset(f, "quest_frame_id"):
                lines.append(f"quest_frame_id {int(f['quest_frame_id'][i])}")
            if is_dataset(f, "quest_timestamp_ms"):
                lines.append(f"quest_ts {int(f['quest_timestamp_ms'][i])}")

            cv2.imshow("RGB/BGR", overlay_text(color, lines))
            cv2.imshow("Depth", overlay_text(depth_vis, [f"depth max {depth_vis_max_mm:.0f}mm"]))

            if show_lowres_depth and lowres_ds is not None:
                low = lowres_ds[i]
                low_vis = get_depth_vis(low, depth_vis_max_mm, depth_colormap)
                low_vis = cv2.resize(low_vis, (int(color.shape[1]), int(color.shape[0])), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Depth Lowres", overlay_text(low_vis, ["depth_lowres resized"]))

            i += step

        k = cv2.waitKey(delay if not paused else 0) & 0xFF
        if k in (27, ord("q"), ord("Q")):
            break
        if k == ord(" "):
            paused = not paused
        if k in (ord("s"), ord("S")):
            export_preview(f, current_i, Path("preview_exports"), depth_vis_max_mm, depth_colormap)

    cv2.destroyAllWindows()


def build_meta(f: h5py.File, idx: int) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "kind": infer_kind(f),
        "frame_array_index": int(idx),
    }

    names = [
        "t_host",
        "frame_index",
        "color_hw_timestamp_ms",
        "depth_hw_timestamp_ms",
        "color_frame_number",
        "depth_frame_number",
        "color_sensor_ts_ms",
        "depth_sensor_ts_ms",
        "color_backend_ts_ms",
        "depth_backend_ts_ms",
        "quest_frame_id",
        "quest_timestamp_ms",
        "fx", "fy", "cx", "cy",
        "camera_position",
        "camera_rotation_xyzw",
        "depth_mode",
        "detections_json",
        "pose_objects_json",
    ]

    for name in names:
        meta[name] = one_dataset_value(f, name, idx)

    if meta.get("detections_json") is not None:
        meta["detections"] = decode_json_string(meta["detections_json"])
    if meta.get("pose_objects_json") is not None:
        meta["pose_objects"] = decode_json_string(meta["pose_objects_json"])
    if all(meta.get(k) is not None for k in ["fx", "fy", "cx", "cy"]):
        fx = float(meta["fx"])
        fy = float(meta["fy"])
        cx = float(meta["cx"])
        cy = float(meta["cy"])
        meta["K"] = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    attrs: dict[str, Any] = {}
    for k in sorted(f.attrs.keys()):
        attrs[k] = to_python_scalar(f.attrs[k])
    meta["attrs"] = attrs

    return meta


def export_preview(f: h5py.File, idx: int, export_dir: Path, depth_vis_max_mm: float, depth_colormap: str) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    color = f["color"][idx]
    depth = f["depth"][idx]
    depth_vis = get_depth_vis(depth, depth_vis_max_mm, depth_colormap)
    cv2.imwrite(str(export_dir / f"frame_{idx:06d}_rgb.png"), color)
    cv2.imwrite(str(export_dir / f"frame_{idx:06d}_depth_vis.png"), depth_vis)
    print(f"Saved preview frame {idx} -> {export_dir}")


def export_frames(
    f: h5py.File,
    indices: list[int],
    export_dir: Path,
    export_rgb_png: bool,
    export_depth_png: bool,
    export_lowres_depth_png: bool,
    export_depth_npy: bool,
    export_meta_json: bool,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    if not is_dataset(f, "color") or not is_dataset(f, "depth"):
        print("[!] File must contain /color and /depth for export")
        return

    color_ds = f["color"]
    depth_ds = f["depth"]
    lowres_ds = f["depth_lowres"] if is_dataset(f, "depth_lowres") else None
    n = int(color_ds.shape[0])

    if not any([export_rgb_png, export_depth_png, export_lowres_depth_png, export_depth_npy, export_meta_json]):
        export_rgb_png = True

    for idx in indices:
        if idx < 0 or idx >= n:
            print(f"Skip invalid frame index: {idx}")
            continue

        stem = export_dir / f"frame_{idx:06d}"
        color = color_ds[idx]
        depth = depth_ds[idx]

        if export_rgb_png:
            cv2.imwrite(str(stem.with_name(stem.name + "_rgb.png")), color)
        if export_depth_png:
            cv2.imwrite(str(stem.with_name(stem.name + "_depth_16u.png")), depth)
        if export_lowres_depth_png and lowres_ds is not None:
            cv2.imwrite(str(stem.with_name(stem.name + "_depth_lowres_16u.png")), lowres_ds[idx])
        if export_depth_npy:
            np.save(str(stem.with_name(stem.name + "_depth.npy")), depth)
        if export_meta_json:
            meta = build_meta(f, idx)
            (stem.with_name(stem.name + "_meta.json")).write_text(
                json.dumps(meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        print(f"Exported frame {idx}")


def run_actions(args: argparse.Namespace) -> None:
    h5_path = Path(args.input)
    if not h5_path.exists():
        print(f"[!] File not found: {h5_path}")
        return

    any_action = (
        args.summary
        or args.check_drops
        or args.stats
        or args.play
        or (args.export_frames is not None and len(args.export_frames) > 0)
    )
    if not any_action:
        args.summary = True

    with h5py.File(str(h5_path), "r") as f:
        if args.summary:
            print_summary(f, compact=args.compact, list_datasets=args.list_datasets)

        if args.check_drops:
            print_drop_checks(f, compact=args.compact, max_gaps=args.max_gaps)

        if args.stats:
            print_timestamp_stats(f, compact=args.compact)

        if args.play:
            playback(
                f,
                start=args.start,
                end=args.end,
                step=args.step,
                play_fps=args.play_fps,
                depth_vis_max_mm=args.depth_vis_max_mm,
                depth_colormap=args.depth_colormap,
                show_lowres_depth=args.show_lowres_depth,
            )

        if args.export_frames is not None and len(args.export_frames) > 0:
            export_dir = Path(args.export_dir) if args.export_dir is not None else h5_path.parent / f"{h5_path.stem}_exports"
            export_frames(
                f,
                indices=args.export_frames,
                export_dir=export_dir,
                export_rgb_png=args.export_rgb_png,
                export_depth_png=args.export_depth_png,
                export_lowres_depth_png=args.export_lowres_depth_png,
                export_depth_npy=args.export_depth_npy,
                export_meta_json=args.export_meta_json,
            )


def main() -> None:
    args = build_parser().parse_args()

    if args.report_file is not None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_actions(args)
        text = buf.getvalue()

        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")

        # Print only a short pointer in PowerShell.
        print(f"[OK] Report saved -> {report_path}")

        # Still allow playback/export messages to be visible only if play/export requested.
        # If you want playback plus report, run playback separately for cleaner output.
        return

    run_actions(args)


if __name__ == "__main__":
    main()
