"""Prepare evaluation-only references; independently diagnose each session clock.

Low-rate RGB colour-centroid motion is not a tracking frontend and never enters
fusion. Clock candidates failing cross-view / held-out checks are not exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation

from experiments.h2o.evaluation import allocate_results
from tools.real_ots_reference import read_motive


SESSIONS = ["stable-0", "stable-1", "stable-2", *[f"Test-{i}" for i in range(6)]]


def save_json(path, data):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(data, default=convert, indent=2), encoding="utf-8")


def load_geometry(root):
    cube_path = root / "results_1/evaluation.json"
    body_path = root / "results_4/candidate_reference_manifest.json"
    cubes = json.loads(cube_path.read_text())["cube"]
    bodies = json.loads(body_path.read_text())["calibration_candidates"]
    geometry = {}
    basis = np.diag([1., -1., -1., 1.])
    for name in ("tagcube0", "tagcube1"):
        ots_cube = np.array(cubes[name]["candidate_T_OTS_from_cube_y_up_translation_mm"])
        ots_cube[:3, 3] /= 1000
        geometry[name] = basis @ np.linalg.inv(ots_cube)
    for name in ("Tomato2", "Tomato3"):
        geometry[name] = np.array(bodies[name]["T_body_from_original_mesh_m"])
    provenance = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (cube_path, body_path)}
    return geometry, provenance


def motion_speed(times, points, valid):
    good = valid & np.isfinite(points).all(axis=1)
    if good.sum() < 6:
        return np.full(len(times), np.nan)
    fill = np.column_stack([np.interp(times, times[good], points[good, k])
                            for k in range(points.shape[1])])
    smooth = gaussian_filter1d(median_filter(fill, size=(3, 1)), 1., axis=0)
    speed = np.linalg.norm(np.gradient(smooth, times, axis=0), axis=1)
    # Interpolation is for smoothing only; do not use gaps or their neighbours.
    usable = np.convolve(good.astype(int), np.ones(5, dtype=int), mode="same") == 5
    speed[~usable] = np.nan
    return speed


def rgb_proxy(path, output):
    points, previews, flows = [], [], []
    last = None
    previous_gray, previous_mask = None, None
    with h5py.File(path, "r") as handle:
        ids = np.arange(0, len(handle["t_host"]), 5)
        times = handle["t_host"][ids]
        for ordinal, index in enumerate(ids):
            image = cv2.resize(handle["color"][index], (640, 360), interpolation=cv2.INTER_AREA)
            if str(handle.attrs.get("color_format", "")).lower().startswith("rgb"):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = (((hsv[:, :, 0] < 6) | (hsv[:, :, 0] > 174)) &
                    (hsv[:, :, 1] > 130) & (hsv[:, :, 2] > 45)).astype(np.uint8)
            # Shared table ROI for these fixed camera views, not adjusted per take.
            mask[:140] = 0
            mask[340:] = 0
            mask[:, :60] = 0
            mask[:, 610:] = 0
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            flow = np.nan
            if previous_gray is not None:
                features = cv2.goodFeaturesToTrack(previous_gray, 100, .01, 3, mask=previous_mask)
                if features is not None and len(features) >= 4:
                    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, features, None,
                                                             winSize=(31, 31), maxLevel=3)
                    back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, moved, None,
                                                                 winSize=(31, 31), maxLevel=3)
                    good = (status.ravel()>0) & (back_status.ravel()>0) & (np.linalg.norm(back-features, axis=2).ravel()<1.)
                    if good.sum() >= 4:
                        distances = np.linalg.norm(moved-features, axis=2).ravel()[good]
                        flow = float(np.median(distances)/(times[ordinal]-times[ordinal-1]))
            flows.append(flow)
            previous_gray = gray
            previous_mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))*255
            joined = cv2.dilate(mask, np.ones((9, 9), np.uint8))
            count, labels, _, _ = cv2.connectedComponentsWithStats(joined)
            candidates = []
            for component in range(1, count):
                yy, xx = np.where((labels == component) & (mask > 0))
                if not 8 <= len(xx) <= 2500:
                    continue
                center = np.array([xx.mean(), yy.mean()])
                score = -len(xx) if last is None else np.linalg.norm(center - last) - .01 * len(xx)
                candidates.append((score, center))
            point = min(candidates, key=lambda item: item[0])[1] if candidates else np.full(2, np.nan)
            if np.isfinite(point).all():
                last = point
            points.append(point)
            if ordinal % max(1, len(ids)//12) == 0:
                if np.isfinite(point).all():
                    cv2.circle(image, tuple(point.astype(int)), 12, (0, 255, 255), 2)
                cv2.putText(image, f'{times[ordinal]-times[0]:.2f}s', (10, 25), 0, .7, (0, 255, 255), 2)
                previews.append(image)
        backend = handle["color_backend_ts_ms"][:]
        full_host = handle["t_host"][:]
    points = np.array(points)
    speed = motion_speed(times, points, np.isfinite(points).all(axis=1))
    data = {"times": times, "points": points, "speed": speed, "flow_speed": np.array(flows),
            "backend_minus_host_s": np.median(backend/1000-full_host)}
    np.savez_compressed(output / f'{path.stem}_rgb_motion.npz', **data)
    canvas = np.zeros(((len(previews)+3)//4*360, 4*640, 3), dtype=np.uint8)
    for i, image in enumerate(previews):
        canvas[i//4*360:(i//4+1)*360, i%4*640:(i%4+1)*640] = image
    cv2.imwrite(str(output / f'{path.stem}_rgb_motion.jpg'), canvas)
    return data


def correlation(times, signal, ots_times, ots_signal, offset):
    shifted = np.interp(times + offset, ots_times, ots_signal, left=np.nan, right=np.nan)
    valid = np.isfinite(signal) & np.isfinite(shifted)
    if valid.sum() < 20 or valid.sum() < .8*np.isfinite(signal).sum():
        return -1.
    if np.std(signal[valid]) < 1e-3 or np.std(shifted[valid]) < 1e-3:
        return -1.
    return float(np.corrcoef(signal[valid], shifted[valid])[0, 1])


def fit_clock(times, signal, ots_times, ots_signal, coarse_offset=None):
    """First 60% fits the clock, final 40% independently checks it."""
    cut = times[0] + .6*(times[-1]-times[0])
    train, held = times < cut, times >= cut
    offsets = (np.arange(-10., max(10., ots_times[-1]-times[-1]+10.), .025) if coarse_offset is None
               else np.arange(coarse_offset-2., coarse_offset+2.001, .025))
    scores = np.array([correlation(times[train], signal[train], ots_times, ots_signal, shift)
                       for shift in offsets])
    best = int(np.argmax(scores))
    offset = float(offsets[best])
    held_scores = np.array([correlation(times[held], signal[held], ots_times, ots_signal, shift)
                            for shift in offsets])
    alternatives = scores[np.abs(offsets-offset) >= 1.]
    margin = float(scores[best]-np.max(alternatives)) if len(alternatives) else 0.
    return {"offset_s": offset, "train_correlation": float(scores[best]),
            "held_correlation": correlation(times[held], signal[held], ots_times, ots_signal, offset),
            "held_best_offset_s": float(offsets[int(np.argmax(held_scores))]),
            "alternative_peak_margin": margin,
            "valid_proxy_fraction": float(np.isfinite(signal).mean()),
            "grid_resolution_s": .025,
                    "profile_offsets": offsets, "profile_correlations": scores}


def motive_epoch(value):
    text = value.strip()
    # Motive 3 CSV uses localized AM/PM following a 12-hour dot-separated time.
    parsed = datetime.strptime(text[:23], '%Y-%m-%d %I.%M.%S.%f')
    if text.endswith(('下午', 'PM')):
        parsed = parsed.replace(hour=parsed.hour+12)
    elif not text.endswith(('上午', 'AM')):
        raise ValueError('Unknown Motive start-time suffix; cannot infer wall clock')
    return parsed.timestamp()


def static_reference(poses):
    center = np.eye(4)
    center[:3, 3] = np.median(poses[:, :3, 3], axis=0)
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    center[:3, :3] = rotations.mean().as_matrix()
    translation = np.linalg.norm(poses[:, :3, 3]-center[:3, 3], axis=1)*1000
    angle = (Rotation.from_matrix(center[:3, :3]).inv()*rotations).magnitude()*180/np.pi
    return center, {"translation_p95_max_mm": np.percentile(translation, [95, 100]),
                    "rotation_p95_max_deg": np.percentile(angle, [95, 100])}


def prepare(args):
    output = allocate_results(args.output_parent)
    geometry, provenance = load_geometry(args.geometry_dir)
    calibrations = json.loads((args.calibration_dir / "production_checks.json").read_text())
    manifest = {"status": "running", "evaluation_only": True, "geometry_sources": provenance,
                "camera_calibration_dir": str(args.calibration_dir.resolve()),
                "clock_scope": "per-session host-time mapping, rate fixed to 1; not exposure synchronization",
                "quality_policy": "finite normalized body quaternion, >=3 measured markers, non-collinear (second singular value >2mm), fitted-marker RMS <=2mm; engineering screen not absolute GT proof",
                "sessions": {}}
    ready_calibrations = {}
    save_json(output / "manifest.json", manifest)
    for name in args.sessions:
        print(f'{name}: preparing OTS and independent clock diagnostics', flush=True)
        session_out = output / name
        session_out.mkdir()
        try:
            root = args.root / name
            csvs = list(root.glob('*.csv'))
            if len(csvs) != 1:
                raise ValueError("Expected exactly one OTS CSV per session")
            data = read_motive(csvs[0])
            paths = sorted(root.rglob('*.h5'))
            timeline = {}
            for path in paths:
                with h5py.File(path) as handle:
                    times = handle['t_host'][:]
                    timeline[path.stem] = [float(times[0]), float(times[-1]), len(times)]
            if set(timeline) != set(calibrations[name]):
                raise ValueError("Camera set does not match validated calibration")
            origin = min(v[0] for v in timeline.values())
            end = max(v[1] for v in timeline.values())
            entry = {"reference_ready": False, "timeline": timeline, "origin_host_s": origin,
                     "ots_csv": str(csvs[0]), "ots_sha256": hashlib.sha256(csvs[0].read_bytes()).hexdigest(),
                     "ots_frames": len(data['times']), "finite_fraction": float(data['finite'].mean()),
                     "quality_valid_fraction": float(data['valid'].mean()),
                     "marker_residual_p50_p95_mm": np.nanpercentile(data['marker_residual_mm'], [50, 95]),
                     "marker_count_distribution": {str(n): int((data['marker_count']==n).sum()) for n in np.unique(data['marker_count'])}}
            np.savez_compressed(session_out/'ots_quality.npz', **{k: data[k] for k in
                                ('times', 'finite', 'valid', 'marker_count', 'marker_residual_mm')})
            payload = {}
            if name.startswith('stable-'):
                center, stability = static_reference(data['poses'][data['valid']])
                entry.update(mode='static_constant_reference_no_clock_fit', stability=stability)
                if (data['valid'].mean() < .95 or stability['translation_p95_max_mm'][1] > 1.
                        or stability['rotation_p95_max_deg'][1] > 1.):
                    raise ValueError('Static recording fails 1mm/1deg maximum variation or 95% validity gate')
                # This is explicitly a constant reference, not an invented clock map.
                body_poses = np.repeat(center[None], 2, axis=0)
                payload = {'candidate_host_time_s': np.array([origin-.05, end+.05]),
                           'reference_valid': np.ones(2, dtype=bool),
                           'static_reference': np.array(True)}
                entry['reference_ready'] = True
            else:
                entry.update(mode='motion_clock_candidate', clocks={})
                for path in paths:
                    if path.stem == 'quest':
                        continue
                    proxy = rgb_proxy(path, session_out)
                    # Motion energy includes rotation-in-place. It is only a clock
                    # proxy, not an object-pose measurement or a fusion observation.
                    ids = np.arange(0, len(data['times']), 20)
                    ot = data['times'][ids]
                    positions = data['poses'][ids, :3, 3]*1000
                    os = motion_speed(ot, positions, data['finite'][ids])
                    if not data['finite'][ids].all():
                        raise ValueError('Clock diagnostic requires finite OTS poses; do not bridge missing poses')
                    rotations = Rotation.from_matrix(data['poses'][ids, :3, :3])
                    angular = np.r_[0., (rotations[:-1].inv()*rotations[1:]).magnitude()/np.diff(ot)]
                    os += 35*gaussian_filter1d(angular, 1.)
                    coarse = origin+float(proxy['backend_minus_host_s'])-motive_epoch(data['metadata']['Capture Start Time'])
                    fit = fit_clock(proxy['times']-origin, proxy['flow_speed'], ot, os, coarse)
                    fit['wall_clock_coarse_offset_s'] = coarse
                    fit['wall_clock_assumption'] = 'Local wall clocks within 2 seconds; independently checked by motion, never accepted alone'
                    np.savez_compressed(session_out/f'{path.stem}_clock_profile.npz',
                                        offsets=fit.pop('profile_offsets'), correlations=fit.pop('profile_correlations'),
                                        ots_times=ot, ots_speed=os)
                    entry['clocks'][path.stem] = fit
                clocks = list(entry['clocks'].values())
                offsets = [v['offset_s'] for v in clocks]
                ready = (len(clocks) >= 2 and np.ptp(offsets) <= .25 and all(
                    v['train_correlation'] >= .5 and v['held_correlation'] >= .4 and
                    v['alternative_peak_margin'] >= .03 and abs(v['offset_s']-v['held_best_offset_s']) <= .3
                    for v in clocks))
                entry['reference_ready'] = bool(ready)
                if ready:
                    offset = float(np.median(offsets))
                    entry['clock'] = {'ots_at_host_origin_s': offset, 'rate': 1.,
                                      'cross_view_spread_s': float(np.ptp(offsets)),
                                      'status': 'motion_checked_candidate_not_hardware_synchronized'}
                    body_poses = data['poses']
                    payload = {'candidate_host_time_s': data['times']-offset+origin,
                               'ots_time_s': data['times'], 'reference_valid': data['valid']}
                else:
                    entry['reason'] = 'Independent clock checks failed; no dynamic reference exported'
            if entry['reference_ready']:
                for cube in ('tagcube0', 'tagcube1'):
                    for body in ('Tomato2', 'Tomato3'):
                        payload[f'T_world_ydown_mesh_native_{cube}_{body}'] = geometry[cube] @ body_poses @ geometry[body]
                np.savez_compressed(output/f'{name}_reference_for_native_mesh.npz', **payload)
                ready_calibrations[name] = calibrations[name]
            manifest['sessions'][name] = entry
            print(name, 'ready', entry['reference_ready'], 'mode', entry['mode'], flush=True)
        except Exception as exc:
            manifest['sessions'][name] = {'reference_ready': False, 'error': repr(exc)}
            print(name, repr(exc), flush=True)
        save_json(output/'manifest.json', manifest)
    save_json(output/'production_checks.json', ready_calibrations)
    manifest.update(status='completed', ready_sessions=list(ready_calibrations),
                    pending_sessions=[n for n in args.sessions if n not in ready_calibrations])
    save_json(output/'manifest.json', manifest)
    print('OUTPUT:', output.resolve(), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/laboratory'))
    parser.add_argument('--sessions', nargs='+', default=SESSIONS)
    parser.add_argument('--calibration-dir', type=Path, default=Path('artifacts/new_real_calibration/results_1'))
    parser.add_argument('--geometry-dir', type=Path, default=Path('artifacts/ots_probe_analysis'))
    parser.add_argument('--output-parent', type=Path, default=Path('artifacts/new_real_reference'))
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
