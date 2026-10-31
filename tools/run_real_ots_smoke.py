"""Bounded CNOS/FoundationPose -> production fusion -> candidate OTS check.

No OTS data is loaded until inference and cached re-fusion have completed.
This is an offline integration check, not a live-latency benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from common.rgbd_tag_alignment import _tracking_camera
from config import FoundationPoseConfig
from core.event_bus import EventBus
from core.types import TimeStampedPacket
from experiments.h2o.ablation import VARIANTS, run_baselines, run_fusion
from experiments.h2o.evaluation import allocate_results
from managers.foundationpose_manager import FoundationPoseManager
from tools.analyze_dual_tag_cubes import _camera_parameters
from tools.real_ots_reference import interpolate_reference, reference_valid


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, default=serialize, indent=2), encoding="utf-8")


def reference_at(times, poses, queries):
    return interpolate_reference(times, poses, queries)


def frame_indices(times, origin, start, duration, full_sequence=False):
    """Full mode includes the final frame of every source, with no stride."""
    times = np.asarray(times)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Input timestamps must be finite, nonempty and strictly increasing")
    if full_sequence:
        return np.arange(len(times))
    return np.flatnonzero((times >= origin + start) & (times < origin + start + duration))


async def run(args):
    root = args.session.resolve()
    preflight = args.calibration_dir.resolve()
    reference_dir = (getattr(args, 'reference_dir', None) or preflight).resolve()
    skip_reference = bool(getattr(args, 'skip_reference_evaluation', False))
    reference_path = reference_dir / f'{root.name}_reference_for_native_mesh.npz'
    if not skip_reference and not reference_path.is_file():
        raise ValueError(f'Evaluation reference not ready: {reference_path}; prepare/check this session clock first')
    calibration = json.loads((preflight / "production_checks.json").read_text())[root.name]
    calibration_scope = "legacy candidate camera calibration"
    reference_scope = "legacy candidate reference"
    if (preflight / 'manifest.json').is_file():
        preparation = json.loads((preflight / 'manifest.json').read_text())
        if preparation.get('full_sequence') and root.name in preparation.get('exported_sessions', []):
            calibration_scope = f"Per-session full-sequence tag fit with held-out validation: {preflight}"
        elif preparation.get('evaluation_only'):
            calibration_scope = f"Per-session validated calibration: {preparation['camera_calibration_dir']}"
    if (reference_dir / 'manifest.json').is_file():
        preparation = json.loads((reference_dir / 'manifest.json').read_text())
        if preparation.get('evaluation_only'):
            state = preparation['sessions'][root.name]
            if not skip_reference and not state['reference_ready']:
                raise ValueError(f'{root.name}: reference preparation has not passed')
            reference_scope = state.get('mode', 'unavailable')
    # Opt-in for this isolated offline process only. Do not alter vendor code
    # or production/live defaults; warps have no cross-hypothesis interaction.
    import kornia
    from adapters.bounded_image_warp import bounded_warp
    original_warp = kornia.geometry.transform.warp_perspective
    kornia.geometry.transform.warp_perspective = bounded_warp(original_warp, 16)
    metadata = json.loads((root / "camera_metadata.json").read_text())
    output = allocate_results(args.output_parent / root.name)
    cfg = FoundationPoseConfig(objects=["TomatoSauce"], recovery_enable=True, depth_fit_quality_enable=True)
    manifest = {"session": str(root), "clip_start_s": args.start, "clip_duration_s": args.duration,
                "full_sequence": bool(getattr(args, "full_sequence", False)),
                "status": "running", "configuration": asdict(cfg),
                "calibration_scope": calibration_scope, "reference_scope": reference_scope,
                "calibration_dir": str(preflight),
                "reference_dir": str(reference_dir),
                "reference_evaluation_requested": not skip_reference,
                "mobile_drift_correction": "fixed T_world_tracking composed with recorded per-frame camera pose; no online tag correction",
                "GT_used_for_initialization_or_fusion": False,
                "native_image_warp_batch_size": 16,
                "timing_scope": "native t_host proxy, 30 Hz grouping; no live-latency claim"}
    manifest["calibration_sha256"] = hashlib.sha256((preflight / "production_checks.json").read_bytes()).hexdigest()
    write_json(output / "calibration_used.json", calibration)
    write_json(output / "manifest.json", manifest)
    bus = EventBus()
    detections = bus.subscribe(cfg.output_topic)
    statuses = bus.subscribe(cfg.status_topic)
    manager = FoundationPoseManager(bus, cfg, input_mode="replay")
    handles = {p.stem: h5py.File(p, "r") for p in sorted(root.rglob("*.h5"))}
    origin = min(float(f["t_host"][0]) for f in handles.values())
    for serial, values in calibration.items():
        if serial != "quest":
            manager._T_world_camera[f"realsense/{serial}"] = np.array(values["T_world_cam"])
    events = []
    for serial, handle in handles.items():
        times = handle["t_host"][:]
        indices = frame_indices(times, origin, args.start, args.duration, manifest["full_sequence"])
        events.extend((float(times[i]), serial, int(i)) for i in indices)
    events.sort()
    if not events:
        raise ValueError("Selected interval contains no input frames")
    manifest.update(origin_host_s=origin, input_frame_count=len(events),
                    selected_start_s=events[0][0]-origin, selected_end_s=events[-1][0]-origin)
    write_json(output / "input_timeline.json", [
        {"camera_id": serial, "source_frame_index": index, "capture_time": timestamp-origin,
         "frame_id": round((timestamp-origin)*30)} for timestamp, serial, index in events])
    write_json(output / "manifest.json", manifest)
    print(f"OUTPUT: {output.resolve()} ({len(events)} input frames)", flush=True)
    observations, counts, writers = [], {}, {}
    tick = time.perf_counter()
    try:
        with (output / "observations.jsonl").open("w", encoding="utf-8") as saved, (output / "frontend_status.jsonl").open("w", encoding="utf-8") as status_saved:
            for ordinal, (timestamp, serial, index) in enumerate(events):
                f = handles[serial]
                source = "quest/rgbd" if serial == "quest" else f"realsense/{serial}"
                k, _ = _camera_parameters(f, index, metadata, serial)
                color = f["color"][index]
                if str(f.attrs.get('color_format', '')).lower().startswith('rgb'):
                    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                payload = {"frames": {"color": color, "depth": f["depth"][index]},
                           "camera": {"K": k, "depth_scale": float(f.attrs.get('depth_scale_m', .001))}}
                if serial == "quest":
                    payload["T_world_tracking"] = calibration[serial]["T_world_tracking"]
                    payload["T_tracking_cam"] = _tracking_camera(f["camera_position"][index], f["camera_rotation_xyzw"][index], f.attrs.get("coordinate_system", "unity_cam"))
                relative = timestamp - origin
                group = round(relative * 30)
                packet = TimeStampedPacket(t_host=relative, source=source, payload=payload, sync_group=group, frame_index=index)
                await manager._handle_frame(packet)
                counts[serial] = counts.get(serial, 0) + 1
                while not detections.empty():
                    det = detections.get_nowait()
                    row = {"frame_id": group, "camera_id": serial, "source": det.source,
                           "t_host": relative, "arrival_time": relative, "capture_time": relative,
                           "original_host_time": timestamp, "source_frame_index": index,
                           "pose_world": det.pose_world.tolist(), "score": det.score,
                           "object_name": det.object_name, "quality_metrics": det.quality_metrics}
                    observations.append(row)
                    saved.write(json.dumps(row, default=serialize) + "\n")
                    saved.flush()
                preview = color.copy()
                while not statuses.empty():
                    status = statuses.get_nowait()
                    status_saved.write(json.dumps({"camera_id": serial, "source_frame_index": index,
                                                  "capture_time": relative, "status": status},
                                                 default=serialize) + "\n")
                    status_saved.flush()
                    if "pose_cam" not in status:
                        continue
                    bounds = manager.runner.model_bounds("TomatoSauce")
                    corners = np.array([[bounds[x,0],bounds[y,1],bounds[z,2]] for x in range(2) for y in range(2) for z in range(2)])
                    pose = status["pose_cam"]
                    rv, _ = cv2.Rodrigues(pose[:3,:3])
                    uv, _ = cv2.projectPoints(corners, rv, pose[:3,3], k, None)
                    uv = np.rint(np.clip(uv.reshape(-1,2), -10000, 10000)).astype(int)
                    for i in range(8):
                        for j in [i^1,i^2,i^4]:
                            if j>i:cv2.line(preview,tuple(uv[i]),tuple(uv[j]),(0,255,0),2)
                    if not (output / f"{serial}_first_pose.jpg").exists():
                        cv2.imwrite(str(output / f"{serial}_first_pose.jpg"), preview)
                cv2.putText(preview, f"{serial} t={relative:.3f}s", (15,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                if serial not in writers:
                    timestamps = f['t_host'][:]
                    fps = (len(timestamps)-1)/(timestamps[-1]-timestamps[0])
                    writer=cv2.VideoWriter(str(output/f'{serial}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(preview.shape[1],preview.shape[0]))
                    if not writer.isOpened():raise RuntimeError('MP4 writer failed')
                    writers[serial]=writer
                writers[serial].write(preview)
                if ordinal%20==0:print(f'{root.name}: {ordinal+1}/{len(events)} frames, {len(observations)} observations',flush=True)
        if not observations:
            raise RuntimeError("CNOS/FoundationPose produced no observations; inspect masks/status before full run")
        methods = run_baselines(observations, list(handles))
        fused, fusion_stats = await run_fusion(observations, VARIANTS['full'], fps=30., base_overrides={'enable_world_revision_checkpoint':True})
        methods['fusion'] = fused
        write_json(output/'predictions.json', methods)
        write_json(output/'fusion_stats.json', fusion_stats)
        if skip_reference:
            manifest.update(status='completed', evaluation_status='reference_pending',
                            frames_by_sensor=counts, n_observations=len(observations),
                            elapsed_s=time.perf_counter()-tick)
            (output/'summary.md').write_text(
                '# Tracking and fusion completed; reference evaluation pending\n\n'
                'No pose accuracy is reported because the independent OTS clock/reference is unresolved.\n'
                'Cached observations and predictions are retained for later evaluation without rerunning inference.\n',
                encoding='utf-8')
            return output
        # Evaluation-only reference is first opened after all predictions are frozen.
        refs=np.load(reference_path)
        reference_times=refs['candidate_host_time_s']
        metrics={}
        for method, rows in methods.items():
            if not rows:
                metrics[method]={'n':0};continue
            queries=origin+np.array([r['frame_id']/30 for r in rows])
            estimated=np.array([r['pose_world'] for r in rows])
            inside = (queries >= reference_times[0]) & (queries <= reference_times[-1])
            valid = reference_valid(refs, queries)
            n_outside = int((~inside).sum())
            n_invalid = int((inside & ~valid).sum())
            queries, estimated = queries[valid], estimated[valid]
            variants={}
            for key in refs.files:
                if not key.startswith('T_world_ydown_mesh_native_'):continue
                if not len(queries):continue
                reference=reference_at(reference_times,refs[key],queries)
                trans=np.linalg.norm(estimated[:,:3,3]-reference[:,:3,3],axis=1)*1000
                rot=np.rad2deg(Rotation.from_matrix(reference[:,:3,:3].transpose(0,2,1)@estimated[:,:3,:3]).magnitude())
                variants[key]={'translation_p50_p95_mm':np.percentile(trans,[50,95]).tolist(),'rotation_p50_p95_deg':np.percentile(rot,[50,95]).tolist()}
            metrics[method]={'n':len(rows),'n_evaluated':len(queries),
                             'n_outside_reference':n_outside,'calibration_variants':variants}
            metrics[method]['n_invalid_reference'] = n_invalid
        write_json(output/'candidate_reference_errors.json',metrics)
        manifest.update(status='completed',evaluation_status='candidate_reference_evaluated',frames_by_sensor=counts,n_observations=len(observations),elapsed_s=time.perf_counter()-tick,origin_host_s=origin,
                        warning='Candidate OTS reference calibration/time remain preliminary. No symmetry-aware rotation scoring.')
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc),frames_by_sensor=counts,n_observations=len(observations))
        raise
    finally:
        write_json(output/'manifest.json',manifest)
        for writer in writers.values():writer.release()
        for handle in handles.values():handle.close()
        manager._executor.shutdown(wait=True)
        kornia.geometry.transform.warp_perspective = original_warp
        print(f'OUTPUT: {output.resolve()}',flush=True)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--start',type=float)
    parser.add_argument('--duration',type=float)
    parser.add_argument('--full-sequence',action='store_true',help='Explicit opt-in: process all frames of every source')
    parser.add_argument('--calibration-dir',type=Path,default=Path('artifacts/ots_probe_analysis/results_6'))
    parser.add_argument('--reference-dir', type=Path, help='Evaluation-only reference package; defaults to calibration-dir')
    parser.add_argument('--skip-reference-evaluation', action='store_true', help='Explicitly retain predictions without reporting OTS accuracy')
    parser.add_argument('--output-parent',type=Path)
    args=parser.parse_args()
    if args.full_sequence:
        if args.start is not None or args.duration is not None:
            parser.error('--full-sequence cannot be combined with --start/--duration')
        args.start, args.duration = 0., None
    else:
        if args.start is None:parser.error('--start is required for a bounded check')
        if args.duration is None:args.duration=2.
        if not 0<args.duration<=5:parser.error('Bounded checks are limited to 5 seconds; use --full-sequence explicitly')
    if args.output_parent is None:
        args.output_parent = Path('artifacts/real_ots_full' if args.full_sequence else 'artifacts/real_ots_smoke')
    output = asyncio.run(run(args))
    if args.full_sequence:
        from tools.analyze_real_ots_smoke import analyze
        from tools.check_real_ots_outputs import check
        if not args.skip_reference_evaluation:
            analyze(output, args.reference_dir or args.calibration_dir)
        check(output)


if __name__=='__main__':main()
