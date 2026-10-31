"""Independent real-data M3T/MegaPose tracking, with one FP seed per camera.

Uses the existing camera/native-mesh conventions and cached ablation interface.
No OTS input or subsequent FP correction is used by tracking or fusion.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import h5py
import numpy as np

from adapters.foundationpose_runner import HopeModelRepository
from adapters.m3t_runner import M3TRunner, validate_pose
from adapters.megapose_runner import MegaPoseRunner
from common.rgbd_tag_alignment import _tracking_camera
from config import FoundationPoseConfig
from experiments.h2o.ablation import VARIANTS, build_config, run_fusion
from experiments.h2o.evaluation import allocate_results
from tools.analyze_dual_tag_cubes import _camera_parameters
from tools.analyze_real_ots_smoke import analyze
from tools.run_real_ots_ablation import equal_camera_baselines
from tools.run_real_ots_smoke import frame_indices, serialize, write_json


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_seeds(seed_run: Path, session: Path, calibration_file: Path,
               start: float = 0., duration: float | None = None) -> tuple[dict, dict]:
    manifest = json.loads((seed_run / 'manifest.json').read_text())
    if manifest['status'] != 'completed' or Path(manifest['session']).resolve() != session.resolve():
        raise ValueError('Seed must be a completed FoundationPose run of the same session')
    if manifest.get('GT_used_for_initialization_or_fusion') is not False:
        raise ValueError('Seed must explicitly declare no GT-assisted initialization/fusion')
    if manifest['calibration_sha256'] != sha(calibration_file):
        raise ValueError('Seed and current camera calibrations differ')
    seeds = {}
    # Select by native capture time, not 30-Hz group or OTS quality/error.
    with (seed_run / 'observations.jsonl').open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in sorted(rows, key=lambda r: (r['capture_time'], r['source_frame_index'])):
        if row['capture_time'] < start or (duration is not None and row['capture_time'] >= start + duration):
            continue
        if row['object_name'] != 'TomatoSauce' or 'foundationpose' not in row['source'].lower():
            raise ValueError('Expected TomatoSauce FoundationPose seed observations')
        camera = row['camera_id']
        if camera not in seeds:
            validate_pose(row['pose_world'])
            seeds[camera] = row
    return seeds, manifest


def camera_world(handle, index: int, camera: str, calibration: dict) -> np.ndarray:
    if camera == 'quest':
        tracking = _tracking_camera(handle['camera_position'][index], handle['camera_rotation_xyzw'][index],
                                    handle.attrs.get('coordinate_system', 'unity_cam'))
        return validate_pose(np.asarray(calibration[camera]['T_world_tracking']) @ tracking)
    return validate_pose(calibration[camera]['T_world_cam'])


def seed_camera_pose(row: dict, index: int, relative: float, world_camera: np.ndarray) -> np.ndarray:
    if index != row['source_frame_index'] or not np.isclose(relative, row['capture_time'], atol=1e-6, rtol=0):
        raise ValueError('Seed does not match this exact source frame/capture time')
    return validate_pose(np.linalg.inv(world_camera) @ validate_pose(row['pose_world']))


def export_mesh(cache: Path) -> tuple[Path, np.ndarray, dict]:
    config = FoundationPoseConfig()
    model, mesh = HopeModelRepository(config.hope_models_root, config.hope_scale_to_m).load_mesh('TomatoSauce')
    files = sorted(p for p in model.mesh_path.parent.iterdir() if p.is_file())
    hashes = {p.name: sha(p) for p in files}
    export_policy = 'native_metre_mesh_windows_safe_material_v2'
    key = hashlib.sha256(json.dumps([hashes, model.scale_to_m, export_policy], sort_keys=True).encode()).hexdigest()[:16]
    # Panda3D renderer subprocesses resolve relative assets against their model
    # search path, not necessarily the caller's current working directory.
    directory = cache.resolve() / ('TomatoSauce_' + key)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'TomatoSauce.obj'
    if not path.exists():
        # Original HOPE material names contain ':'; exporting that name creates
        # an NTFS alternate stream instead of an ordinary texture file.
        if mesh.visual.material.image is None:
            raise ValueError('HOPE texture image is missing')
        mesh.visual.material.name = 'TomatoSauce'
        mesh.export(path)  # Metres; no centering/rotation; export UVs and texture too.
    texture = directory / 'TomatoSauce.png'
    if not texture.is_file() or texture.stat().st_size == 0:
        raise ValueError(f'Exported texture missing: {texture}')
    return path, mesh.bounds, {'source': str(model.mesh_path), 'scale_to_m': model.scale_to_m,
                               'source_sha256': hashes, 'exported_mesh_sha256': sha(path),
                               'export_policy': export_policy, 'exported_texture_sha256': sha(texture),
                               'coordinate_policy': 'native HOPE mesh basis and origin retained'}


async def run(args) -> Path:
    root = args.session.resolve()
    calibration_file = args.calibration_dir.resolve() / 'production_checks.json'
    calibration = json.loads(calibration_file.read_text())[root.name]
    reference_path = args.reference_dir / f'{root.name}_reference_for_native_mesh.npz'
    if not reference_path.is_file():
        raise ValueError('Validated evaluation reference is required for this comparison')
    seeds, seed_manifest = load_seeds(args.seed_run_dir, root, calibration_file, args.start, args.duration)
    metadata = json.loads((root / 'camera_metadata.json').read_text())
    paths = sorted(root.rglob('*.h5'))
    cameras = [p.stem for p in paths]
    if not cameras or len(set(cameras)) != len(cameras) or set(seeds) != set(cameras):
        raise ValueError(f'Missing/duplicate cameras or first-pose seeds: cameras={cameras}, seeds={list(seeds)}')
    if args.check_only:
        for path in paths:
            with h5py.File(path, 'r') as handle:
                row = seeds[path.stem]
                i = int(row['source_frame_index'])
                relative = float(handle['t_host'][i]) - seed_manifest['origin_host_s']
                seed_camera_pose(row, i, relative, camera_world(handle, i, path.stem, calibration))
        print(json.dumps({'session': root.name, 'frontend': args.frontend,
                          'seed_frames': {c: r['source_frame_index'] for c, r in seeds.items()}, 'status': 'input_checks_passed'}))
        return args.seed_run_dir

    output = allocate_results(args.output_parent / root.name)
    manifest = {'status': 'running', 'session': str(root), 'frontend': args.frontend,
                'full_sequence': args.full_sequence, 'clip_start_s': args.start, 'clip_duration_s': args.duration,
                'calibration_dir': str(args.calibration_dir.resolve()), 'calibration_sha256': sha(calibration_file),
                'reference_dir': str(args.reference_dir.resolve()), 'GT_used_for_initialization_or_fusion': False,
                'initialization': 'one saved FP observation per camera; no later FP correction or automatic redetection',
                'seed_run': str(args.seed_run_dir.resolve()), 'seed_observations_sha256': sha(args.seed_run_dir / 'observations.jsonl'),
                'seed_frames': {c: r['source_frame_index'] for c, r in seeds.items()},
                'input_policy': 'all native frames retained; frames before each camera seed produce no observation',
                'mobile_drift_correction': 'fixed T_world_tracking composed with recorded camera pose; no online tag correction',
                'timing_scope': 'capture-paced timestamps, arrival_time=capture_time; no live latency claim',
                'video_policy': 'native dimensions; CFR at each source mean recorded FPS; exact capture timestamps retained separately',
                'frontend_settings': {'m3t_threads': args.threads, 'megapose_iterations': args.iterations, 'megapose_render_workers': 1},
                'fusion_configuration': asdict(build_config(VARIANTS['full']).fusion)}
    manifest['code_sha256'] = {str(p): sha(p) for p in [Path(__file__), Path('adapters/m3t_runner.py'), Path('adapters/megapose_runner.py')]}
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'calibration_used.json', calibration)
    write_json(output / 'initialization.json', seeds)
    counts, observations, runner = {}, [], None
    started = time.perf_counter()
    try:
        mesh_path, bounds, manifest['mesh'] = export_mesh(args.mesh_cache)
        runner = (M3TRunner(mesh_path, args.model_cache, threads=args.threads) if args.frontend == 'm3t'
                  else MegaPoseRunner(mesh_path, 'TomatoSauce', iterations=args.iterations, render_workers=1))
        with ExitStack() as stack:
            if hasattr(runner, 'close'):
                stack.callback(runner.close)
            handles = {p.stem: stack.enter_context(h5py.File(p, 'r')) for p in paths}
            origin = min(float(h['t_host'][0]) for h in handles.values())
            if not np.isclose(origin, seed_manifest['origin_host_s'], atol=1e-6, rtol=0):
                raise ValueError('Recording origin differs from seed run')
            events = []
            for c, h in handles.items():
                times = h['t_host'][:]
                indices = frame_indices(times, origin, args.start, args.duration, args.full_sequence)
                events.extend((float(times[i]), c, int(i)) for i in indices)
            events.sort()
            if not events:
                raise ValueError('No frames selected')
            manifest.update(origin_host_s=origin, input_frame_count=len(events),
                            selected_start_s=events[0][0] - origin, selected_end_s=events[-1][0] - origin)
            timeline = [{'camera_id': c, 'source_frame_index': i, 'capture_time': t - origin,
                         'frame_id': round((t - origin) * 30)} for t, c, i in events]
            write_json(output / 'input_timeline.json', timeline)
            write_json(output / 'manifest.json', manifest)
            writers = {}
            saved = stack.enter_context((output / 'observations.jsonl').open('w', encoding='utf-8'))
            status_file = stack.enter_context((output / 'frontend_status.jsonl').open('w', encoding='utf-8'))
            corners = np.array([[bounds[x, 0], bounds[y, 1], bounds[z, 2]] for x in range(2) for y in range(2) for z in range(2)])
            for ordinal, (timestamp, camera, index) in enumerate(events):
                h = handles[camera]
                relative = timestamp - origin
                k, _ = _camera_parameters(h, index, metadata, camera)
                color = h['color'][index]
                if str(h.attrs.get('color_format', '')).lower().startswith('rgb'):
                    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                preview = color.copy()
                initial_index = seeds[camera]['source_frame_index']
                status = {'camera_id': camera, 'source_frame_index': index, 'capture_time': relative,
                          'mode': 'waiting_for_initial_pose'}
                if index >= initial_index:
                    world_camera = camera_world(h, index, camera, calibration)
                    initial = seed_camera_pose(seeds[camera], index, relative, world_camera) if index == initial_index else None
                    depth = h['depth'][index].astype(np.float32) * float(h.attrs.get('depth_scale_m', .001))
                    depth[~np.isfinite(depth) | (depth < 0)] = 0
                    tick = time.perf_counter()
                    pose, quality = runner.process(camera, color, depth, k, initial)
                    row = {'frame_id': round(relative * 30), 'camera_id': camera, 'source_frame_index': index,
                           'source': f'{"quest/rgbd" if camera == "quest" else "realsense/" + camera}/{args.frontend}/TomatoSauce',
                           'detector': args.frontend, 'object_name': 'TomatoSauce', 't_host': relative,
                           'capture_time': relative, 'arrival_time': relative, 'original_host_time': timestamp,
                           'score': 1., 'score_kind': 'constant_not_calibrated', 'quality_metrics': quality,
                           'pose_cam': pose, 'T_world_camera': world_camera, 'pose_world': world_camera @ pose,
                           'mode': 'initialize' if initial is not None else 'track',
                           'tracking_and_quality_ms': (time.perf_counter() - tick) * 1000}
                    observations.append(row)
                    saved.write(json.dumps(row, default=serialize) + '\n')
                    saved.flush()
                    status.update(mode=row['mode'], pose_cam=pose, quality=quality)
                    if np.all((corners @ pose[:3, :3].T + pose[:3, 3])[:, 2] > .001):
                        rv, _ = cv2.Rodrigues(pose[:3, :3])
                        uv, _ = cv2.projectPoints(corners, rv, pose[:3, 3], k, None)
                        uv = np.rint(np.clip(uv.reshape(-1, 2), -10000, 10000)).astype(int)
                        for i in range(8):
                            for j in (i ^ 1, i ^ 2, i ^ 4):
                                if j > i:
                                    cv2.line(preview, tuple(uv[i]), tuple(uv[j]), (0, 255, 0), 2)
                    if initial is not None:
                        cv2.imwrite(str(output / f'{camera}_first_pose.jpg'), preview)
                status_file.write(json.dumps(status, default=serialize) + '\n')
                status_file.flush()
                cv2.putText(preview, f'{args.frontend} {camera} t={relative:.3f}s {status["mode"]}',
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                if camera not in writers:
                    times = h['t_host'][:]
                    fps = (len(times) - 1) / (times[-1] - times[0])
                    writer = cv2.VideoWriter(str(output / f'{camera}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (color.shape[1], color.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('MP4 writer failed')
                    writers[camera] = writer
                    stack.callback(writer.release)
                writers[camera].write(preview)
                counts[camera] = counts.get(camera, 0) + 1
                if ordinal % 20 == 0:
                    print(f'{args.frontend}/{root.name}: {ordinal+1}/{len(events)} frames, {len(observations)} poses', flush=True)
        if not observations:
            raise ValueError('No frontend outputs')
        methods = equal_camera_baselines(observations, cameras)
        methods['fusion'], runtime = await run_fusion(observations, VARIANTS['full'], fps=30.)
        write_json(output / 'predictions.json', methods)
        write_json(output / 'fusion_stats.json', runtime)
        manifest.update(status='completed', frames_by_sensor=counts, n_observations=len(observations),
                        elapsed_s=time.perf_counter() - started, evaluation_status='candidate_reference_evaluated')
        write_json(output / 'manifest.json', manifest)
        analyze(output, args.reference_dir)  # Evaluation reference first used after predictions freeze.
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=repr(exc), frames_by_sensor=counts, n_observations=len(observations))
        raise
    finally:
        write_json(output / 'manifest.json', manifest)
        print(f'OUTPUT: {output.resolve()}', flush=True)
    return output


def main() -> None:
    if os.environ.get('CONDA_DEFAULT_ENV') != 'mr_nav' or Path(sys.prefix).name.lower() != 'mr_nav':
        raise RuntimeError('Activate conda mr_nav')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend', choices=['m3t', 'megapose'], required=True)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--seed-run-dir', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, default=Path('artifacts/new_real_calibration/results_1'))
    parser.add_argument('--reference-dir', type=Path, default=Path('artifacts/new_real_reference/results_1'))
    parser.add_argument('--output-parent', type=Path)
    parser.add_argument('--full-sequence', action='store_true')
    parser.add_argument('--start', type=float)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--mesh-cache', type=Path, default=Path('artifacts/real_frontend_models'))
    parser.add_argument('--model-cache', type=Path, default=Path('artifacts/m3t_models'))
    args = parser.parse_args()
    if args.full_sequence:
        if args.start is not None or args.duration is not None:
            parser.error('Full sequence cannot also select a clip')
        args.start, args.duration = 0., None
    elif args.start is None or args.start < 0 or args.duration is None or not 0 < args.duration <= 5:
        parser.error('Select a <=5-second clip or explicitly --full-sequence')
    if args.threads < 1 or args.iterations < 1:
        parser.error('Threads and iterations must be positive')
    if args.output_parent is None:
        args.output_parent = Path(f'artifacts/new_real_{args.frontend}' if args.full_sequence else f'artifacts/real_frontend_smoke/{args.frontend}')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
