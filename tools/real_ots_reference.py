"""Evaluation-only OTS parsing and reference validity. Never used by fusion."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def read_motive(path: Path, body: str = "TomatoSouce") -> dict:
    """Read measured markers separately from Motive's fitted rigid-body markers."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = [next(reader) for _ in range(7)]
        meta = dict(zip(header[0][::2], header[0][1::2]))
        if meta.get("Rotation Type") != "Quaternion" or meta.get("Coordinate Space") != "Global":
            raise ValueError("Expected global quaternion Motive export")
        units = {"Millimeters": .001, "Meters": 1.}
        if meta.get("Length Units") not in units:
            raise ValueError("Unsupported Motive length units")
        scale = units[meta["Length Units"]]

        def columns(kind, name, part, axes):
            result = []
            for axis in axes:
                hits = [i for i in range(2, len(header[6])) if
                        (header[2][i], header[3][i], header[5][i], header[6][i]) ==
                        (kind, name, part, axis)]
                if len(hits) != 1:
                    raise ValueError(f"Missing/duplicate {kind}/{name}/{part}/{axis}")
                result.append(hits[0])
            return result

        pos_cols = columns("Rigid Body", body, "Position", "XYZ")
        rot_cols = columns("Rigid Body", body, "Rotation", "XYZW")
        names = sorted({header[3][i] for i in range(2, len(header[6]))
                        if header[2][i] == "Marker" and header[3][i].startswith(body + ":")})
        measured_cols = [columns("Marker", n, "Position", "XYZ") for n in names]
        fitted_cols = [columns("Rigid Body Marker", n, "Position", "XYZ") for n in names]
        selected = [1] + pos_cols + rot_cols + sum(measured_cols, []) + sum(fitted_cols, [])
        data = np.array([[float(row[i]) if row[i] else np.nan for i in selected]
                         for row in reader if row])
    times, pos, quat = data[:, 0], data[:, 1:4] * scale, data[:, 4:8]
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid OTS timeline")
    norms = np.linalg.norm(quat, axis=1)
    finite = np.isfinite(np.c_[pos, quat]).all(axis=1) & (np.abs(norms - 1) < .01)
    poses = np.full((len(times), 4, 4), np.nan)
    poses[finite] = np.eye(4)
    poses[finite, :3, :3] = Rotation.from_quat(quat[finite]).as_matrix()
    poses[finite, :3, 3] = pos[finite]
    count = np.zeros(len(times), dtype=int)
    residual = np.full(len(times), np.nan)
    noncollinear = np.zeros(len(times), dtype=bool)
    if names:
        measured = data[:, 8:8 + 3 * len(names)].reshape(-1, len(names), 3) * scale
        fitted = data[:, 8 + 3 * len(names):].reshape(-1, len(names), 3) * scale
        visible = np.isfinite(measured).all(axis=2) & np.isfinite(fitted).all(axis=2)
        count = visible.sum(axis=1)
        for i in np.flatnonzero(count >= 3):
            points = measured[i, visible[i]]
            singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
            noncollinear[i] = singular[1] > .002
            residual[i] = np.sqrt(np.mean(np.sum((points - fitted[i, visible[i]]) ** 2, axis=1))) * 1000
    valid = finite & (count >= 3) & noncollinear & (residual <= 2.)
    return {"times": times, "poses": poses, "finite": finite, "valid": valid,
            "marker_count": count, "marker_residual_mm": residual,
            "metadata": meta, "marker_names": names}


def reference_valid(refs, queries: np.ndarray) -> np.ndarray:
    """Reject outside range and either endpoint invalid; never bridge missing OTS."""
    times = refs["candidate_host_time_s"]
    queries = np.asarray(queries)
    right = np.searchsorted(times, queries, side="left").clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    exact = np.isclose(times[right], queries, rtol=0, atol=1e-8)
    left = np.where(exact, right, left)
    valid = (queries >= times[0]) & (queries <= times[-1])
    if "reference_valid" in refs:
        mask = refs["reference_valid"].astype(bool)
        valid &= mask[left] & mask[right]
        if not bool(refs.get("static_reference", False)):
            valid &= (times[right] - times[left]) <= .05
    return valid


def interpolate_reference(times, poses, queries):
    """Interpolate only selected endpoints, so unrelated NaN gaps are harmless."""
    times, queries = np.asarray(times), np.asarray(queries)
    if np.any(queries < times[0]) or np.any(queries > times[-1]):
        raise ValueError("Cannot extrapolate OTS reference")
    if not len(queries):
        return np.empty((0, 4, 4))
    right = np.searchsorted(times, queries, side="left").clip(1, len(times) - 1)
    left = right - 1
    exact = np.isclose(times[right], queries, rtol=0, atol=1e-8)
    left = np.where(exact, right, left)
    left = np.where(np.isclose(times[0], queries, rtol=0, atol=1e-8), 0, left)
    right = np.where(np.isclose(times[0], queries, rtol=0, atol=1e-8), 0, right)
    if not np.isfinite(poses[np.r_[left, right]]).all():
        raise ValueError("Cannot interpolate an invalid OTS endpoint")
    used = np.unique(np.r_[left, right])
    if len(used) == 1:
        return np.repeat(poses[used], len(queries), axis=0)
    out = np.repeat(np.eye(4)[None], len(queries), axis=0)
    out[:, :3, :3] = Slerp(times[used], Rotation.from_matrix(poses[used, :3, :3]))(queries).as_matrix()
    out[:, :3, 3] = np.column_stack([np.interp(queries, times[used], poses[used, k, 3]) for k in range(3)])
    return out
