"""Bounded, lossless wall-paced RGB-D -> FoundationPose -> fusion -> UDP test.

The loopback sender repeats the latest fused result at a requested rate. Distinct
updates, capture groups and result age are measured separately from packet rate.
This is NOT physical-camera acquisition, Unity rendering or a LAN benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import socket
import threading
import time

import numpy as np


def statistics(values):
    data = np.asarray(values, dtype=float)
    return ({"n": len(data), "mean": float(data.mean()),
             "p50": float(np.percentile(data, 50)),
             "p95": float(np.percentile(data, 95)), "max": float(data.max())}
            if len(data) else {"n": 0, "mean": None, "p50": None, "p95": None, "max": None})


def receiver_metrics(rows, start, end, hz):
    selected = [r for r in rows if start <= r['received'] < end]
    span = end - start
    updates = {r['update_id'] for r in selected}
    groups = {(r['fused']['object_name'], r['fused']['sync_group']) for r in selected}
    ages = [(r['received'] - r['capture_due']) * 1000 for r in selected]
    return {
        'window_s': span, 'packets': len(selected),
        'packet_hz': len(selected) / span, 'distinct_update_hz': len(updates) / span,
        'distinct_capture_group_hz': len(groups) / span,
        'repeat_fraction': 1 - len(updates) / len(selected) if selected else None,
        'result_age_ms': statistics(ages),
        'fraction_age_within_one_receiver_period':
            sum(a <= 1000 / hz for a in ages) / len(ages) if ages else None,
        'udp_loopback_ms': statistics([(r['received'] - r['sent']) * 1000 for r in selected]),
        'receive_interval_ms': statistics(np.diff([r['received'] for r in selected]) * 1000),
    }


class LoopbackReceiver:
    """Independent OS threads: GPU/event-loop stalls cannot fake fresh output."""

    def __init__(self, hz):
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError('Receiver rate must be positive and finite')
        self.hz = hz
        self.rows = []
        self.errors = []
        self.sent = 0
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind(('127.0.0.1', 0))
        self._socket.settimeout(.1)
        self._address = self._socket.getsockname()
        self._threads = []

    def update(self, fused, capture_due, published, update_id):
        with self._lock:
            self._latest = {'fused': fused, 'capture_due': capture_due,
                            'published': published, 'update_id': update_id}

    def start(self):
        self._threads = [threading.Thread(target=self._send, daemon=True),
                         threading.Thread(target=self._receive, daemon=True)]
        for thread in self._threads:
            thread.start()

    def _send(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                deadline = time.perf_counter()
                while not self._stop.is_set():
                    if self._stop.wait(max(0., deadline - time.perf_counter())):
                        break
                    with self._lock:
                        latest = self._latest
                    if latest is not None:
                        message = {**latest, 'packet_id': self.sent, 'sent': time.perf_counter()}
                        raw = json.dumps(message, allow_nan=False).encode()
                        if len(raw) > 65507:
                            raise ValueError('Fused datagram exceeds UDP payload limit')
                        sender.sendto(raw, self._address)
                        self.sent += 1
                    deadline += 1 / self.hz
                    # Do not issue catch-up bursts and count them as 30 Hz.
                    if deadline < time.perf_counter():
                        deadline = time.perf_counter() + 1 / self.hz
        except Exception as exc:
            self.errors.append(repr(exc))
            self._stop.set()

    def _receive(self):
        try:
            while not self._stop.is_set():
                try:
                    raw, _ = self._socket.recvfrom(65535)
                except socket.timeout:
                    continue
                received = time.perf_counter()
                self.rows.append({**json.loads(raw), 'received': received})
        except Exception as exc:
            if not self._stop.is_set():
                self.errors.append(repr(exc))

    def close(self):
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2)
        self._socket.close()


def load_clip(args):
    import cv2
    import h5py
    from common.rgbd_tag_alignment import _tracking_camera
    from core.types import TimeStampedPacket
    from tools.analyze_dual_tag_cubes import _camera_parameters

    metadata = json.loads((args.session / 'camera_metadata.json').read_text())
    calibration = json.loads((args.calibration_dir / 'production_checks.json').read_text())[args.session.name]
    paths = sorted(args.session.rglob('*.h5'))
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError('Missing or duplicate sensor H5 files')
    with_handles = {}
    try:
        with_handles = {p.stem: h5py.File(p, 'r') for p in paths}
        origin = min(float(f['t_host'][0]) for f in with_handles.values())
        packets, byte_count = [], 0
        for serial, handle in with_handles.items():
            if serial not in calibration or not calibration[serial].get('ready', False):
                raise ValueError(f'Validated extrinsics missing: {serial}')
            times = handle['t_host'][:]
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f'Invalid capture timestamps: {serial}')
            indices = np.flatnonzero((times >= origin + args.start) &
                                    (times < origin + args.start + args.warmup + args.duration))
            if not len(indices):
                raise ValueError(f'No frames in requested interval: {serial}')
            for i in indices:
                color, depth = handle['color'][i], handle['depth'][i]
                if str(handle.attrs.get('color_format', '')).lower().startswith('rgb'):
                    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                byte_count += color.nbytes + depth.nbytes
                if byte_count > args.max_preload_gib * 1024 ** 3:
                    raise ValueError('Preload memory cap reached; use a shorter bounded clip')
                k, _ = _camera_parameters(handle, i, metadata, serial)
                payload = {'frames': {'color': color, 'depth': depth},
                           'camera': {'K': k, 'depth_scale': float(handle.attrs.get('depth_scale_m', .001))}}
                if serial == 'quest':
                    payload['T_world_tracking'] = calibration[serial]['T_world_tracking']
                    payload['T_tracking_cam'] = _tracking_camera(
                        handle['camera_position'][i], handle['camera_rotation_xyzw'][i],
                        handle.attrs.get('coordinate_system', 'unity_cam'))
                capture = float(times[i]) - origin
                source = 'quest/rgbd' if serial == 'quest' else f'realsense/{serial}'
                packets.append(TimeStampedPacket(capture, source, payload,
                                                frame_index=int(i), sync_group=round(capture * 30)))
        return sorted(packets, key=lambda p: (p.t_host, p.source)), calibration, byte_count
    finally:
        for handle in with_handles.values():
            handle.close()


async def run(args):
    import kornia
    import torch
    from adapters.bounded_image_warp import bounded_warp
    from config import FoundationPoseConfig
    from core.event_bus import EventBus
    from experiments.h2o.ablation import build_config, VARIANTS
    from experiments.h2o.evaluation import allocate_results
    from managers.foundationpose_manager import FoundationPoseManager
    from managers.fusion_manager import FusionManager

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this actual FoundationPose benchmark')
    output = allocate_results(args.output_parent / args.session.name)
    print(f'OUTPUT: {output.resolve()}', flush=True)
    manifest = {'status': 'running', 'session': str(args.session.resolve()),
                'start_s': args.start, 'warmup_capture_s': args.warmup,
                'measured_capture_s': args.duration, 'receiver_target_hz': args.receiver_hz,
                'scope': 'preloaded RGB-D release through real inference/fusion to local UDP receipt',
                'excludes': ['physical acquisition', 'H5 loading', 'model loading', 'LAN', 'Unity display'],
                'queue_policy': 'lossless FIFO; no frame drops, throttling or cached poses',
                'initialization': 'CNOS masks + FoundationPose register; no GT',
                'gpu': torch.cuda.get_device_name(), 'platform': platform.platform(),
                'code_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                    __file__, 'managers/foundationpose_manager.py', 'managers/fusion_manager.py',
                    'core/pose_consensus.py', 'config.py', 'adapters/foundationpose_runner.py')},
                'calibration_sha256': hashlib.sha256((args.calibration_dir / 'production_checks.json').read_bytes()).hexdigest()}
    def save_manifest():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save_manifest()
    receiver = LoopbackReceiver(args.receiver_hz)
    tasks, frontend = [], None
    frame_rows, detection_rows, fused_rows, backlog, stage_rows = [], [], [], [], []
    try:
        packets, calibration, byte_count = load_clip(args)
        boundary = args.start + args.warmup
        warm = [p for p in packets if p.t_host < boundary]
        measured = [p for p in packets if p.t_host >= boundary]
        if not measured:
            raise ValueError('No measured frames')
        sources = sorted({p.source for p in measured})
        manifest.update(preload_bytes=byte_count, input_frames=len(measured), sources=sources)
        manifest['input_frames_by_source'] = {source: sum(p.source == source for p in measured) for source in sources}
        shutil.copyfile(__file__, output / 'benchmark_source.py')
        cfg = FoundationPoseConfig(objects=['TomatoSauce'], recovery_enable=True, depth_fit_quality_enable=True)
        fusion_cfg = build_config(VARIANTS['full'], {'enable_singleton_history_guard': True})
        manifest.update(frontend_config=asdict(cfg), fusion_config=asdict(fusion_cfg.fusion))
        bus = EventBus()
        detections = bus.subscribe(cfg.output_topic)
        statuses = bus.subscribe(cfg.status_topic)
        fused_queue = bus.subscribe(fusion_cfg.fusion.topic_fused)
        frontend = FoundationPoseManager(bus, cfg, input_mode='replay')
        for serial, values in calibration.items():
            if serial != 'quest':
                frontend._T_world_camera[f'realsense/{serial}'] = np.asarray(values['T_world_cam'])
        fusion = FusionManager(bus, asyncio.get_running_loop(), fusion_cfg)
        fusion.configure_runtime(offline_replay=True)
        # Match the existing physical runner; do not change vendor/live defaults.
        kornia.geometry.transform.warp_perspective = bounded_warp(kornia.geometry.transform.warp_perspective, 16)
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        await loop.run_in_executor(frontend._executor, frontend.runner.prepare, cfg.objects)
        await loop.run_in_executor(frontend._executor, frontend.recovery_runner.prepare, cfg.objects)
        manifest['model_prepare_wall_s'] = time.perf_counter() - started
        context = {'measured': False, 'capture': None, 'source': None}

        def instrument(owner, name, stage):
            original = getattr(owner, name)
            def timed(*values, **keywords):
                t0 = time.perf_counter()
                result = original(*values, **keywords)
                torch.cuda.synchronize()
                if context['measured']:
                    stage_rows.append({'stage': stage, 'capture': context['capture'],
                                       'source': context['source'],
                                       'mode': getattr(result, 'mode', None),
                                       'ms': (time.perf_counter() - t0) * 1000})
                return result
            setattr(owner, name, timed)

        instrument(frontend.runner, 'process', 'foundationpose')
        instrument(frontend.runner, 'measure_depth_fit', 'depth_fit')
        instrument(frontend.recovery_runner, 'detect', 'cnos')

        async def fuse(det, measured_phase, epoch=0):
            t0 = time.perf_counter()
            result = await fusion._on_pose_message(det)
            if result is not None:
                await fusion._update_and_publish(*result)
                if fused_queue.empty():
                    raise RuntimeError('Fusion accepted observation but did not publish')
            done = time.perf_counter()
            if measured_phase:
                detection_rows.append({'source': det.source, 'capture': det.capture_time,
                                       'fusion_ms': (done - t0) * 1000, 'accepted': result is not None})
            while not fused_queue.empty():
                message = fused_queue.get_nowait()
                if measured_phase:
                    row = {'fused': message, 'published': done,
                           'capture_due': epoch + message['t_capture'] - boundary,
                           'update_id': len(fused_rows)}
                    fused_rows.append(row)
                    receiver.update(**row)

        def drain_status():
            modes = []
            while not statuses.empty():
                message = statuses.get_nowait()
                modes.append(message.get('mode', message.get('event', 'status')))
            return modes

        started = time.perf_counter()
        for packet in warm:
            await frontend._handle_frame(packet)
            while not detections.empty():
                await fuse(detections.get_nowait(), False)
            drain_status()
        manifest.update(warmup_wall_s=time.perf_counter() - started,
                        warmup_frames=len(warm), active_sources_after_warmup=sorted({s for s, _ in frontend._active}))
        save_manifest()
        print(f'Prepared; active sources: {manifest["active_sources_after_warmup"]}. Releasing {len(measured)} frames at capture cadence.', flush=True)
        pending = asyncio.Queue()
        producer_finished = False
        epoch = time.perf_counter() + .1
        receiver.start()

        async def producer():
            nonlocal producer_finished
            for packet in measured:
                due = epoch + packet.t_host - boundary
                await asyncio.sleep(max(0., due - time.perf_counter()))
                released = time.perf_counter()
                pending.put_nowait((packet, due, released))
                backlog.append({'at': released, 'pending': pending.qsize(), 'phase': 'release'})
            producer_finished = True
            pending.put_nowait(None)

        async def tracker():
            while True:
                item = await pending.get()
                if item is None:
                    detections.put_nowait(None)
                    return
                packet, due, released = item
                context.update(measured=True, capture=packet.t_host, source=packet.source)
                t0 = time.perf_counter()
                await frontend._handle_frame(packet)
                # Explicitly include any outstanding GPU work in stage duration.
                await loop.run_in_executor(frontend._executor, torch.cuda.synchronize)
                done = time.perf_counter()
                frame_rows.append({'source': packet.source, 'frame_index': packet.frame_index,
                                   'capture': packet.t_host, 'due': due, 'released': released,
                                   'started': t0, 'done': done, 'frontend_ms': (done - t0) * 1000,
                                   'queue_ms': (t0 - released) * 1000,
                                   'release_lag_ms': (released - due) * 1000, 'modes': drain_status()})
                backlog.append({'at': done, 'pending': max(0, pending.qsize() - int(producer_finished)), 'phase': 'complete'})
                if len(frame_rows) % 30 == 0:
                    print(f'{len(frame_rows)}/{len(measured)} processed; age={done-due:.2f}s', flush=True)

        async def fuse_worker():
            while True:
                det = await detections.get()
                if det is None:
                    return
                await fuse(det, True, epoch)

        tasks = [asyncio.create_task(fn()) for fn in (producer, tracker, fuse_worker)]
        await asyncio.gather(*tasks)
        completed = time.perf_counter()
        await asyncio.sleep(max(.1, epoch + args.duration - completed))
        receiver.close()
        if receiver.errors:
            raise RuntimeError(f'Receiver errors: {receiver.errors}')
        summary = {
            'release_window': receiver_metrics(receiver.rows, epoch, epoch + args.duration, args.receiver_hz),
            'including_drain': receiver_metrics(receiver.rows, epoch, max(completed, epoch + args.duration) + .1, args.receiver_hz),
            'input_frames': len(measured), 'processed_frames': len(frame_rows),
            'processed_within_release_window': sum(r['done'] <= epoch + args.duration for r in frame_rows),
            'drain_after_release_s': max(0., completed - epoch - args.duration),
            'frontend_ms': statistics([r['frontend_ms'] for r in frame_rows]),
            'frontend_queue_ms': statistics([r['queue_ms'] for r in frame_rows]),
            'release_lag_ms': statistics([r['release_lag_ms'] for r in frame_rows]),
            'fusion_ms': statistics([r['fusion_ms'] for r in detection_rows]),
            'fresh_fused_outputs': len(fused_rows),
            'first_fused_after_release_s': fused_rows[0]['published'] - epoch if fused_rows else None,
            'release_to_fused_ms': statistics([(r['published'] - r['capture_due']) * 1000 for r in fused_rows]),
            'max_pending_frames': max((r['pending'] for r in backlog), default=0),
            'stages_ms': {stage: statistics([r['ms'] for r in stage_rows if r['stage'] == stage])
                          for stage in ('foundationpose', 'cnos', 'depth_fit')},
            'foundationpose_modes_ms': {mode: statistics([r['ms'] for r in stage_rows
                if r['stage'] == 'foundationpose' and r['mode'] == mode]) for mode in ('track', 'register')},
            'udp_sent': receiver.sent, 'udp_received': len(receiver.rows),
            'note': '30 Hz repeats latest output; only distinct updates/groups measure freshness. No accuracy/physical-live claim.'}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        metric = summary['release_window']
        report = ('# Recorded RGB-D paced timing check\n\n'
                  f'- Frames: {len(frame_rows)}/{len(measured)}; lossless, real FoundationPose.\n'
                  f'- Receiver target: {args.receiver_hz} Hz; release window: {args.duration} s.\n'
                  f'- Packet / distinct update / distinct group Hz during release: {metric["packet_hz"]:.2f} / {metric["distinct_update_hz"]:.2f} / {metric["distinct_capture_group_hz"]:.2f}.\n'
                  f'- Receiver result age (ms): {metric["result_age_ms"]}.\n'
                  f'- Frontend frame time (ms): {summary["frontend_ms"]}.\n'
                  f'- Fusion per-observation time (ms): {summary["fusion_ms"]}.\n'
                  f'- Drain beyond release: {summary["drain_after_release_s"]:.3f} s; peak pending: {summary["max_pending_frames"]}.\n\n'
                  'Preloaded images; excludes acquisition, disk load, model preparation, LAN and rendering. '
                  'Warmup is sequential and separately timed. Repeated packets are not fresh poses. '
                  'One bounded replay is not long-run or physical end-to-end validation.\n')
        (output / 'summary.md').write_text(report, encoding='utf-8')
        manifest['status'] = 'completed' if fused_rows else 'no_fused_output'
        print(report, flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        receiver.close()
        if frontend is not None:
            frontend._executor.shutdown(wait=True, cancel_futures=True)
        for name, rows in [('frames', frame_rows), ('detections', detection_rows),
                           ('fused', fused_rows), ('receiver', receiver.rows), ('backlog', backlog), ('stages', stage_rows)]:
            with (output / f'{name}.jsonl').open('w', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row) + '\n')
        save_manifest()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=Path('data/laboratory/Test-0'))
    parser.add_argument('--calibration-dir', type=Path, default=Path('artifacts/new_real_calibration/results_1'))
    parser.add_argument('--output-parent', type=Path, default=Path('artifacts/recorded_realtime'))
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--warmup', type=float, default=1.)
    parser.add_argument('--duration', type=float, default=3.)
    parser.add_argument('--receiver-hz', type=float, default=30.)
    parser.add_argument('--max-preload-gib', type=float, default=2.)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v < 0 for v in (args.start, args.warmup)):
        parser.error('start/warmup must be finite and nonnegative')
    if any(not math.isfinite(v) or v <= 0 for v in (args.duration, args.receiver_hz, args.max_preload_gib)):
        parser.error('duration/rate/memory cap must be positive and finite')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
