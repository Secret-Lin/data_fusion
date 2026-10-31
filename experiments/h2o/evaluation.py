"""Paired pose evaluation shared by the H2O analysis and ablation tools."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.h2o.benchmark import _rotation_error_deg


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def attach_keypoints(ground_truth: list[dict[str, Any]], sequence_dir: Path, camera: str) -> None:
    """Add official 21-point boxes in object coordinates for scoring only."""
    for row in ground_truth:
        stem = f"{row['frame_id']:06d}.txt"
        point_path = sequence_dir / camera / "obj_pose" / stem
        rt_dir = sequence_dir / camera / "obj_pose_rt"
        if not rt_dir.is_dir():
            rt_dir = sequence_dir / camera / "obj_pose_RT"
        if not point_path.is_file():
            continue
        values = np.loadtxt(point_path).reshape(-1)
        rt_values = np.loadtxt(rt_dir / stem).reshape(-1)
        if values.size != 64 or rt_values.size != 17 or int(values[0]) != int(rt_values[0]):
            raise ValueError(f"Invalid H2O object points or identity: {point_path}")
        T = rt_values[1:].reshape(4, 4)
        points = values[1:].reshape(21, 3)
        row["object_points"] = ((points - T[:3, 3]) @ T[:3, :3]).tolist()


def allocate_results(parent: Path) -> Path:
    """Atomically reserve results_N, never reusing an existing experiment."""
    parent.mkdir(parents=True, exist_ok=True)
    indices = [int(p.name[8:]) for p in parent.glob("results_*") if p.name[8:].isdigit()]
    index = max(indices, default=-1) + 1
    while True:
        path = parent / f"results_{index}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            index += 1


def score_poses(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    mesh_points: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Score every target frame, explicitly retaining missing estimates.

    Each input is a single take, which has one annotated object per frame.
    No alignment to the test ground truth is fitted.
    """
    truth = {int(row["frame_id"]): row for row in ground_truth}
    if len(truth) != len(ground_truth):
        raise ValueError("Duplicate ground-truth frame ids")
    estimates = {int(row["frame_id"]): row for row in predictions}
    if len(estimates) != len(predictions):
        raise ValueError("Duplicate prediction frame ids")
    if set(estimates) - set(truth):
        raise ValueError("Predictions contain frames outside the evaluation targets")
    rows = []
    for frame_id, gt in truth.items():
        row = {"frame_id": frame_id, "object_name": gt["object_name"], "present": False,
               "success_5cm_10deg": False}
        pred = estimates.get(frame_id)
        if pred is not None and pred.get("pose_world") is not None:
            if pred.get("object_name", gt["object_name"]) != gt["object_name"]:
                raise ValueError(f"Object identity mismatch at frame {frame_id}")
            T = np.asarray(pred["pose_world"], dtype=float)
            ref = np.asarray(gt["pose_world"], dtype=float)
            if T.shape != (4, 4) or not np.isfinite(T).all():
                raise ValueError(f"Invalid prediction at frame {frame_id}")
            trans = float(np.linalg.norm(T[:3, 3] - ref[:3, 3]))
            rot = _rotation_error_deg(T, ref)
            row.update(present=True, translation_error_m=trans, rotation_error_deg=rot,
                       success_5cm_10deg=trans <= 0.05 and rot <= 10.0)
            if gt.get("object_points") is not None:
                points = np.asarray(gt["object_points"], dtype=float)
                a = points @ T[:3, :3].T + T[:3, 3]
                b = points @ ref[:3, :3].T + ref[:3, 3]
                row["keypoint_mepe_mm"] = float(np.linalg.norm(a - b, axis=1).mean() * 1000)
            if mesh_points is not None:
                points = np.asarray(mesh_points, dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                    raise ValueError("mesh_points must be a finite Nx3 array")
                a = points @ T[:3, :3].T + T[:3, 3]
                b = points @ ref[:3, :3].T + ref[:3, 3]
                # Mean vertex error is ADD. This is not ADD-S; symmetric
                # objects require a nearest-neighbour variant.
                row["mesh_add_mm"] = float(np.linalg.norm(a - b, axis=1).mean() * 1000)
            raw = pred.get("raw", pred)
            if "integrity" in raw:
                row["health"] = raw["integrity"].get("health")
            if "observation_count" in raw:
                row["observation_count"] = int(raw["observation_count"])
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = [r for r in rows if r["present"]]
    n = len(rows)
    result: dict[str, Any] = {
        "n_expected": n, "n": len(present), "n_missing": n - len(present),
        "coverage": len(present) / n if n else None,
        "failure_rate_including_missing": 1 - sum(r["success_5cm_10deg"] for r in rows) / n if n else None,
        "failure_rate_on_outputs": 1 - sum(r["success_5cm_10deg"] for r in present) / len(present) if present else None,
    }
    for field, prefix, scale in [("translation_error_m", "translation", 1000),
                                  ("rotation_error_deg", "rotation", 1)]:
        values = np.array([r[field] for r in present]) * scale
        unit = "mm" if prefix == "translation" else "deg"
        for name, operation in [("mean", np.mean), ("p50", lambda x: np.percentile(x, 50)),
                                ("p95", lambda x: np.percentile(x, 95))]:
            result[f"{prefix}_{name}_{unit}"] = float(operation(values)) if len(values) else None
    keypoints = [r["keypoint_mepe_mm"] for r in present if "keypoint_mepe_mm" in r]
    result["keypoint_mepe_mm"] = float(np.mean(keypoints)) if keypoints else None
    mesh_add = [r["mesh_add_mm"] for r in present if "mesh_add_mm" in r]
    result["mesh_add_mean_mm"] = float(np.mean(mesh_add)) if mesh_add else None
    result["mesh_add_p50_mm"] = float(np.percentile(mesh_add, 50)) if mesh_add else None
    result["mesh_add_p95_mm"] = float(np.percentile(mesh_add, 95)) if mesh_add else None
    counts = [r["observation_count"] for r in present if "observation_count" in r]
    result["observations_per_output_mean"] = float(np.mean(counts)) if counts else None
    result["health_counts"] = {h: sum(r.get("health") == h for r in present)
                               for h in ("Reliable", "Degraded", "Unsafe")}
    return result


def paired_delta(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> dict[str, Any]:
    """Positive differences mean the first method has larger error."""
    b = {r["frame_id"]: r for r in second}
    pairs = [(a, b[a["frame_id"]]) for a in first
             if a["present"] and a["frame_id"] in b and b[a["frame_id"]]["present"]]
    return {
        "n_common": len(pairs),
        "mean_translation_delta_mm": float(np.mean([
            (a["translation_error_m"] - z["translation_error_m"]) * 1000 for a, z in pairs
        ])) if pairs else None,
        "mean_rotation_delta_deg": float(np.mean([
            a["rotation_error_deg"] - z["rotation_error_deg"] for a, z in pairs
        ])) if pairs else None,
    }


def bootstrap_mean_ci(values: Iterable[float], *, seed: int = 811) -> list[float] | None:
    values = np.asarray(list(values), dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


@dataclass
class MetricAccumulator:
    """Pool exact quantiles with numeric arrays, without millions of Python dicts."""

    n_expected: int = 0
    n: int = 0
    n_success: int = 0
    chunks: dict[str, list[np.ndarray]] = field(default_factory=dict)
    health: dict[str, int] = field(default_factory=lambda: {h: 0 for h in ("Reliable", "Degraded", "Unsafe")})

    def add(self, rows: list[dict[str, Any]]) -> None:
        valid = [r for r in rows if r["present"]]
        self.n_expected += len(rows)
        self.n += len(valid)
        self.n_success += sum(r["success_5cm_10deg"] for r in valid)
        for key in (
            "translation_error_m",
            "rotation_error_deg",
            "keypoint_mepe_mm",
            "mesh_add_mm",
            "observation_count",
        ):
            values = np.asarray([r[key] for r in valid if key in r], dtype=np.float64)
            if len(values):
                self.chunks.setdefault(key, []).append(values)
        for h in self.health:
            self.health[h] += sum(r.get("health") == h for r in valid)

    def result(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "n_expected": self.n_expected, "n": self.n, "n_missing": self.n_expected - self.n,
            "coverage": self.n / self.n_expected if self.n_expected else None,
            "failure_rate_including_missing": 1 - self.n_success / self.n_expected if self.n_expected else None,
            "failure_rate_on_outputs": 1 - self.n_success / self.n if self.n else None,
            "health_counts": dict(self.health),
        }
        for key, prefix, scale, unit in [("translation_error_m", "translation", 1000, "mm"),
                                         ("rotation_error_deg", "rotation", 1, "deg")]:
            values = np.concatenate(self.chunks[key]) * scale if key in self.chunks else np.array([])
            result[f"{prefix}_mean_{unit}"] = float(np.mean(values)) if len(values) else None
            for percentile in (50, 95):
                result[f"{prefix}_p{percentile}_{unit}"] = float(np.percentile(values, percentile)) if len(values) else None
        for key, output in [("keypoint_mepe_mm", "keypoint_mepe_mm"),
                            ("mesh_add_mm", "mesh_add_mean_mm"),
                            ("observation_count", "observations_per_output_mean")]:
            chunks = self.chunks.get(key, [])
            if output == "mesh_add_mean_mm":
                values = np.concatenate(chunks) if chunks else np.array([])
                result[output] = float(np.mean(values)) if len(values) else None
                result["mesh_add_p50_mm"] = float(np.percentile(values, 50)) if len(values) else None
                result["mesh_add_p95_mm"] = float(np.percentile(values, 95)) if len(values) else None
            else:
                result[output] = float(sum(x.sum() for x in chunks) / sum(x.size for x in chunks)) if chunks else None
        return result


def metric_table(metrics: dict[str, dict[str, Any]]) -> str:
    lines = ["| Method | Frames | Coverage % | Mean t mm | P95 t mm | P95 R deg | Failure %* |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def number(value: float | None, scale: float = 1) -> str:
        return "N/A" if value is None else f"{value * scale:.2f}"
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['n']}/{m['n_expected']} | {number(m['coverage'], 100)} | "
                     f"{number(m['translation_mean_mm'])} | {number(m['translation_p95_mm'])} | "
                     f"{number(m['rotation_p95_deg'])} | {number(m['failure_rate_including_missing'], 100)} |")
    return "\n".join(lines) + "\n\n*Failure: translation > 50 mm OR rotation > 10 degrees OR missing.\n"
