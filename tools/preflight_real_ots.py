"""Read-only full-sequence input checks; no tracker or OTS-based fitting."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from tools.run_real_ots_smoke import frame_indices


def check(session, calibration_dir, reference_dir=None, allow_missing_reference=False):
    calibration = json.loads((calibration_dir/'production_checks.json').read_text())[session.name]
    json.loads((session/'camera_metadata.json').read_text())
    reference_dir = reference_dir or calibration_dir
    reference_path = reference_dir/f'{session.name}_reference_for_native_mesh.npz'
    if not allow_missing_reference and not reference_path.is_file():
        raise FileNotFoundError(f'Missing candidate reference for {session.name}')
    if reference_path.is_file():
        with np.load(reference_path) as refs:
            times = refs['candidate_host_time_s']
            if len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
                raise ValueError('Invalid reference timeline')
            mask = refs['reference_valid'] if 'reference_valid' in refs else np.ones(len(times), dtype=bool)
            if mask.shape != times.shape or not np.any(mask):
                raise ValueError('No valid OTS reference')
            keys = [k for k in refs.files if k.startswith('T_world_ydown_mesh_native_')]
            if not keys:
                raise ValueError('No native-mesh reference variants')
            for key in keys:
                poses = refs[key]
                if poses.shape != (len(times),4,4) or not np.isfinite(poses[mask]).all():
                    raise ValueError(f'Invalid reference poses: {key}')
    sources = {}
    for path in sorted(session.rglob('*.h5')):
        if path.stem in sources:
            raise ValueError(f'Duplicate sensor name: {path}')
        source = path.stem
        entry = calibration[source]
        transform = np.array(entry['T_world_tracking' if source=='quest' else 'T_world_cam'])
        if transform.shape != (4,4) or not np.isfinite(transform).all():
            raise ValueError(f'Invalid transform: {source}')
        if not np.allclose(transform[3], [0,0,0,1]) or not np.allclose(transform[:3,:3].T@transform[:3,:3], np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(transform[:3,:3]),1):
            raise ValueError(f'Non-rigid or reflected transform: {source}')
        with h5py.File(path, 'r') as f:
            times=f['t_host'][:]
            indices=frame_indices(times,0,0,None,True)
            keys=['color','depth'] + (['camera_position','camera_rotation_xyzw'] if source=='quest' else [])
            for key in keys:
                if len(f[key])!=len(times):raise ValueError(f'{path}: inconsistent {key} count')
            sources[source]={'frames':len(indices),'duration_s':float(times[-1]-times[0]),
                             'first_host':float(times[0]),'last_host':float(times[-1])}
    if not sources:raise ValueError(f'No H5 inputs in {session}')
    print(json.dumps({'session':str(session),'sources':sources,'total_frames':sum(s['frames'] for s in sources.values())},indent=2))
    return sources


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--calibration-dir',type=Path,default=Path('artifacts/ots_probe_analysis/results_6'))
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--allow-missing-reference', action='store_true')
    args=parser.parse_args()
    check(args.session,args.calibration_dir,args.reference_dir,args.allow_missing_reference)
