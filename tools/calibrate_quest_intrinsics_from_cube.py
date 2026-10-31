from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from standalone.tag_cube_localizer_replay_with_cube_model import (
    CubeModelLocalizer,
    load_cube_model,
    make_aruco_parameters,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--cube-model", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fix-distortion",
        action="store_true",
        help="Keep k1, k2, k3 and tangential distortion fixed at zero",
    )
    args = parser.parse_args()
    model = load_cube_model(args.cube_model.resolve())
    solver = CubeModelLocalizer(model, 100, 100, 3, min_observations=1)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5),
        make_aruco_parameters(),
    )
    objects, images = [], []
    session = args.session.resolve()
    h5_candidates = (session / "quest" / "quest.h5", session / "quest.h5")
    h5_path = next((path for path in h5_candidates if path.is_file()), None)
    if h5_path is None:
        searched = ", ".join(str(path) for path in h5_candidates)
        raise FileNotFoundError(f"Quest HDF5 not found; searched: {searched}")
    with h5py.File(h5_path, "r") as stream:
        height, width = stream["color"].shape[1:3]
        initial = np.array(
            [[stream["fx"][0], 0, stream["cx"][0]],
             [0, stream["fy"][0], stream["cy"][0]], [0, 0, 1]],
            dtype=np.float64,
        )
        for index in range(0, len(stream["color"]), max(1, args.stride)):
            gray = cv2.cvtColor(np.asarray(stream["color"][index]), cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue
            mapping = {
                int(marker_id[0]): np.asarray(corner).reshape(4, 2)
                for corner, marker_id in zip(corners, ids)
                if int(marker_id[0]) in model.T_world_tag
            }
            object_points, image_points, tag_ids = solver._frame_points(mapping)
            if len(tag_ids) >= 2:
                objects.append(object_points.astype(np.float32))
                images.append(image_points.astype(np.float32))
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K3
    )
    if args.fix_distortion:
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    (
        rms,
        calibrated,
        distortion,
        _,
        _,
        intrinsic_std,
        _,
        per_view_errors,
    ) = cv2.calibrateCameraExtended(
        objects, images, (width, height), initial.copy(), np.zeros(5), flags=flags
    )
    per_view_errors = per_view_errors.ravel()
    report = {
        "frames": len(objects), "image_size": [width, height],
        "distortion_model": "fixed_zero" if args.fix_distortion else "fitted_k1_k2",
        "initial_K": initial.tolist(), "calibrated_K": calibrated.tolist(),
        "distortion_k1_k2_p1_p2_k3": distortion.ravel().tolist(),
        "calibration_rms_px": float(rms),
        "intrinsic_std_fx_fy_cx_cy": intrinsic_std.ravel()[:4].tolist(),
        "per_view_rms_px": {
            "median": float(np.median(per_view_errors)),
            "p95": float(np.percentile(per_view_errors, 95)),
            "max": float(np.max(per_view_errors)),
        },
        "relative_change": {
            "fx": float((calibrated[0, 0] - initial[0, 0]) / initial[0, 0]),
            "fy": float((calibrated[1, 1] - initial[1, 1]) / initial[1, 1]),
            "cx_px": float(calibrated[0, 2] - initial[0, 2]),
            "cy_px": float(calibrated[1, 2] - initial[1, 2]),
        },
    }
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
