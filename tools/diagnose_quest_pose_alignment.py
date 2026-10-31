from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def invert(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def dispersion(transforms: list[np.ndarray]) -> tuple[float, float, float, float]:
    xyz = np.array([item[:3, 3] for item in transforms])
    translation = np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1)
    rotations = Rotation.from_matrix(np.array([item[:3, :3] for item in transforms]))
    rotation = np.degrees((rotations.mean().inv() * rotations).magnitude())
    return (
        float(np.median(translation)), float(np.percentile(translation, 95)),
        float(np.median(rotation)), float(np.percentile(rotation, 95)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--max-lag", type=int, default=8)
    args = parser.parse_args()
    root = args.session.resolve()
    table = pd.read_csv(root / "quest_extrinsic_evaluation" / "per_frame.csv")
    indices = table.frame_index.astype(int).to_numpy()
    columns = [f"T_world_tracking_{row}_{column}" for row in range(4) for column in range(4)]
    external = np.array([row.reshape(4, 4) for row in table[columns].to_numpy()])
    with h5py.File(root / "quest" / "quest.h5", "r") as stream:
        position = np.asarray(stream["camera_position"])
        quaternion = np.asarray(stream["camera_rotation_xyzw"])
        times = np.asarray(stream["t_host"])
    tracking_camera = np.repeat(np.eye(4)[None], len(position), axis=0)
    tracking_camera[:, :3, :3] = Rotation.from_quat(quaternion).as_matrix()
    tracking_camera[:, :3, 3] = position
    tag_world_camera = np.array(
        [external[row] @ tracking_camera[index] for row, index in enumerate(indices)]
    )
    frame_period_ms = float(np.median(np.diff(times)) * 1000.0)
    print("lag_frames lag_ms translation_median_m translation_p95_m rotation_median_deg rotation_p95_deg")
    for lag in range(-args.max_lag, args.max_lag + 1):
        values = [
            tag_world_camera[row] @ invert(tracking_camera[index + lag])
            for row, index in enumerate(indices)
            if 0 <= index + lag < len(tracking_camera)
        ]
        print(lag, round(lag * frame_period_ms, 3), *[round(item, 6) for item in dispersion(values)])

    translation_motion = np.linalg.norm(
        np.diff(position, axis=0, prepend=position[:1]), axis=1
    )
    rotation_motion = np.r_[
        0.0,
        np.degrees(
            (
                Rotation.from_quat(quaternion[:-1]).inv()
                * Rotation.from_quat(quaternion[1:])
            ).magnitude()
        ),
    ]
    center = np.eye(4)
    center[:3, :3] = Rotation.from_matrix(external[:, :3, :3]).mean().as_matrix()
    center[:3, 3] = np.median(external[:, :3, 3], axis=0)
    errors = [dispersion([center, item]) for item in external]
    translation_error = np.asarray([item[0] * 2.0 for item in errors])
    rotation_error = np.asarray([item[2] * 2.0 for item in errors])
    print("translation_error_vs_motion_correlation", float(np.corrcoef(translation_error, translation_motion[indices])[0, 1]))
    print("rotation_error_vs_motion_correlation", float(np.corrcoef(rotation_error, rotation_motion[indices])[0, 1]))

    selected = table.pnp_rms_px.to_numpy() <= np.percentile(
        table.pnp_rms_px.to_numpy(), 75
    )
    reported_poses = tracking_camera[indices[selected]]
    optical_poses = tag_world_camera[selected]
    target_camera = np.array([invert(item) for item in optical_poses])
    rotation_camera_to_reported, translation_camera_to_reported = cv2.calibrateHandEye(
        [item[:3, :3] for item in reported_poses],
        [item[:3, 3] for item in reported_poses],
        [item[:3, :3] for item in target_camera],
        [item[:3, 3] for item in target_camera],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    reported_optical = np.eye(4)
    reported_optical[:3, :3] = rotation_camera_to_reported
    reported_optical[:3, 3] = np.asarray(translation_camera_to_reported).reshape(3)
    corrected_external = [
        tag_world_camera[row]
        @ invert(tracking_camera[index] @ reported_optical)
        for row, index in enumerate(indices)
    ]
    print("hand_eye_frames", int(np.count_nonzero(selected)))
    print("T_reported_optical", reported_optical.tolist())
    print("reported_to_optical_translation_m", float(np.linalg.norm(reported_optical[:3, 3])))
    print("before_hand_eye", dispersion(list(external)))
    print("after_hand_eye", dispersion(corrected_external))


if __name__ == "__main__":
    main()
