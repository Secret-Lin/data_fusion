"""Estimate detector/sensor pose noise from a static validation recording.

The input is JSONL (or JSONL.GZ) with one row per observation.  Each row must
contain ``pose_world`` and ``ground_truth`` (4x4 matrices), plus either
``sensor_id``/``source`` or ``detector``.  The generated JSON can be supplied
as ``FusionConfig.covariance_calibration_path``; it is optional and does not
change the existing record/replay schema.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

import gtsam
import numpy as np


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _pose(value: Any) -> gtsam.Pose3:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("pose must be a 4x4 matrix")
    return gtsam.Pose3(gtsam.Rot3(matrix[:3, :3]), gtsam.Point3(matrix[:3, 3]))


def _robust_sigma(values: np.ndarray, floor: float) -> float:
    if values.size == 0:
        return floor
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    sigma = 1.4826 * mad
    if not np.isfinite(sigma).all() or float(np.max(sigma)) <= floor:
        sigma = np.std(values, axis=0, ddof=1 if len(values) > 1 else 0)
    sigma = np.asarray(sigma, dtype=np.float64)
    return max(float(np.sqrt(np.mean(np.square(sigma)))), floor)


def calibrate(input_path: Path, output_path: Path) -> dict[str, Any]:
    grouped: dict[str, list[np.ndarray]] = {}
    skipped = 0
    for row in _rows(input_path):
        try:
            observed = _pose(row.get("pose_world", row.get("T")))
            truth = _pose(row.get("ground_truth", row.get("gt_pose_world")))
            residual = np.asarray(gtsam.Pose3.Logmap(truth.between(observed)))
            if residual.shape != (6,) or not np.isfinite(residual).all():
                raise ValueError("invalid residual")
            key = str(row.get("sensor_id", row.get("source", row.get("detector", "unknown"))))
            grouped.setdefault(key, []).append(residual)
        except Exception:
            skipped += 1

    sensors: dict[str, Any] = {}
    for key, values in grouped.items():
        residuals = np.asarray(values, dtype=np.float64)
        sensors[key] = {
            "rot_sigma": _robust_sigma(residuals[:, :3], 1e-4),
            "trans_sigma": _robust_sigma(residuals[:, 3:], 1e-4),
            "n": int(len(values)),
            "residual_mean": np.mean(residuals, axis=0).tolist(),
            "residual_covariance": np.cov(residuals, rowvar=False).tolist()
            if len(values) > 1
            else np.zeros((6, 6), dtype=np.float64).tolist(),
        }
    result = {
        "schema_version": 1,
        "source": str(input_path),
        "n_rows": int(sum(len(v) for v in grouped.values())),
        "n_skipped": int(skipped),
        "sensors": sensors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL/JSONL.GZ residual rows")
    parser.add_argument("--output", type=Path, required=True, help="calibration JSON")
    args = parser.parse_args()
    result = calibrate(args.input, args.output)
    print(json.dumps({"output": str(args.output), "sensors": len(result["sensors"]), "n_rows": result["n_rows"]}))


if __name__ == "__main__":
    main()
