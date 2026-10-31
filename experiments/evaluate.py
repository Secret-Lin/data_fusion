from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _rot_err_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R = R_est @ R_gt.T
    c = float((np.trace(R) - 1.0) * 0.5)
    c = min(1.0, max(-1.0, c))
    return float(np.degrees(np.arccos(c)))


def _extract_T(rec: dict[str, Any]) -> np.ndarray | None:
    # supported patterns
    if "T" in rec:
        try:
            return np.array(rec["T"], dtype=np.float64).reshape(4, 4)
        except Exception:
            return None
    if "pose_world" in rec:
        try:
            return np.array(rec["pose_world"], dtype=np.float64).reshape(4, 4)
        except Exception:
            return None
    if "msg" in rec and isinstance(rec["msg"], dict):
        return _extract_T(rec["msg"])
    return None


def _extract_t_host(rec: dict[str, Any]) -> float | None:
    for k in ("t_host", "stamp", "timestamp"):
        if k in rec:
            try:
                return float(rec[k])
            except Exception:
                pass
    if "msg" in rec and isinstance(rec["msg"], dict):
        return _extract_t_host(rec["msg"])
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _read_gt(path: Path) -> list[tuple[float, np.ndarray]]:
    if path.suffix.lower() == ".csv":
        out: list[tuple[float, np.ndarray]] = []
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                t = float(row["t_host"])
                if "T00" in row:
                    T = np.array(
                        [[float(row[f"T{i}{j}"]) for j in range(4)] for i in range(4)],
                        dtype=np.float64,
                    )
                else:
                    tx, ty, tz = float(row["tx"]), float(row["ty"]), float(row["tz"])
                    qw, qx, qy, qz = (
                        float(row["qw"]),
                        float(row["qx"]),
                        float(row["qy"]),
                        float(row["qz"]),
                    )
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = _quat_to_R(qw, qx, qy, qz)
                    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
                out.append((t, T))
        return out

    # json/jsonl
    recs = _read_jsonl(path)
    out: list[tuple[float, np.ndarray]] = []
    for r in recs:
        t = _extract_t_host(r)
        T = _extract_T(r)
        if t is None or T is None:
            continue
        out.append((float(t), T))
    return out


def _nearest_gt(t: float, gt_times: np.ndarray) -> int:
    return int(np.argmin(np.abs(gt_times - t)))


def evaluate(
    *,
    est_path: Path,
    gt_path: Path,
    fail_trans_m: float,
    fail_rot_deg: float,
    degrade_trans_m: float,
    degrade_rot_deg: float,
    max_match_dt_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    est_recs = _read_jsonl(est_path)
    gt = _read_gt(gt_path)
    if not gt:
        raise RuntimeError(f"No GT loaded from {gt_path}")

    gt_times = np.array([t for t, _ in gt], dtype=np.float64)
    gt_T = [T for _, T in gt]

    per: list[dict[str, Any]] = []
    trans_errs: list[float] = []
    rot_errs: list[float] = []
    est_ts: list[float] = []

    # Integrity labels
    y_true_deg: list[int] = []
    y_pred_deg: list[int] = []

    for rec in est_recs:
        t = _extract_t_host(rec)
        T_est = _extract_T(rec)
        if t is None or T_est is None:
            continue

        j = _nearest_gt(float(t), gt_times)
        dt = abs(float(t) - float(gt_times[j]))
        if dt > max_match_dt_s:
            continue

        T_gt = gt_T[j]
        te = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
        re = _rot_err_deg(T_est[:3, :3], T_gt[:3, :3])

        trans_errs.append(te)
        rot_errs.append(re)
        est_ts.append(float(t))

        degraded_true = int((te > degrade_trans_m) or (re > degrade_rot_deg))
        y_true_deg.append(degraded_true)

        integrity = None
        if isinstance(rec.get("msg"), dict):
            integrity = rec["msg"].get("integrity")
        elif isinstance(rec.get("integrity"), dict):
            integrity = rec.get("integrity")
        degraded_pred = 0
        if isinstance(integrity, dict):
            health = str(integrity.get("health", "")).lower()
            degraded_pred = int(health in {"degraded", "unsafe"})
        y_pred_deg.append(degraded_pred)

        per.append(
            {
                "t_host": float(t),
                "dt_match": float(dt),
                "trans_err_m": te,
                "rot_err_deg": re,
                "degraded_true": degraded_true,
                "degraded_pred": degraded_pred,
            }
        )

    if not trans_errs:
        raise RuntimeError("No matched estimate/GT pairs after timestamp association")

    trans_np = np.array(trans_errs, dtype=np.float64)
    rot_np = np.array(rot_errs, dtype=np.float64)
    fail = (trans_np > fail_trans_m) | (rot_np > fail_rot_deg)

    # runtime/update frequency from estimate timestamps
    est_ts_np = np.array(est_ts, dtype=np.float64)
    est_ts_np = np.sort(est_ts_np)
    dts = np.diff(est_ts_np)
    mean_dt = float(np.mean(dts)) if dts.size > 0 else float("nan")
    update_hz = (
        float(1.0 / mean_dt) if dts.size > 0 and mean_dt > 1e-9 else float("nan")
    )

    # integrity precision/recall/FAR
    y_true = np.array(y_true_deg, dtype=np.int32)
    y_pred = np.array(y_pred_deg, dtype=np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    far = float(fp / max(fp + tn, 1))

    summary = {
        "n_matched": int(trans_np.size),
        "mean_translation_error_m": float(np.mean(trans_np)),
        "mean_rotation_error_deg": float(np.mean(rot_np)),
        "translation_rmse_m": float(np.sqrt(np.mean(trans_np**2))),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rot_np**2))),
        "failure_rate": float(np.mean(fail.astype(np.float64))),
        "runtime_per_frame_s": mean_dt,
        "update_frequency_hz": update_hz,
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": far,
        "thresholds": {
            "fail_trans_m": float(fail_trans_m),
            "fail_rot_deg": float(fail_rot_deg),
            "degrade_trans_m": float(degrade_trans_m),
            "degrade_rot_deg": float(degrade_rot_deg),
            "max_match_dt_s": float(max_match_dt_s),
        },
    }
    return summary, per


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Unified evaluation for fusion/baselines")
    ap.add_argument("--est", required=True, help="Estimate JSONL path")
    ap.add_argument("--gt", required=True, help="Ground-truth JSONL/CSV path")
    ap.add_argument("--out-json", required=True, help="Summary output JSON")
    ap.add_argument("--out-csv", default="", help="Optional per-frame CSV")
    ap.add_argument("--fail-trans-m", type=float, default=0.05)
    ap.add_argument("--fail-rot-deg", type=float, default=10.0)
    ap.add_argument("--degrade-trans-m", type=float, default=0.03)
    ap.add_argument("--degrade-rot-deg", type=float, default=5.0)
    ap.add_argument("--max-match-dt-s", type=float, default=0.05)
    args = ap.parse_args(argv)

    summary, per = evaluate(
        est_path=Path(args.est),
        gt_path=Path(args.gt),
        fail_trans_m=float(args.fail_trans_m),
        fail_rot_deg=float(args.fail_rot_deg),
        degrade_trans_m=float(args.degrade_trans_m),
        degrade_rot_deg=float(args.degrade_rot_deg),
        max_match_dt_s=float(args.max_match_dt_s),
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            if per:
                w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
                w.writeheader()
                w.writerows(per)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
