from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.evaluate import evaluate


def _latest_subdir(path: Path) -> Path | None:
    if not path.exists():
        return None
    subs = [p for p in path.iterdir() if p.is_dir()]
    if not subs:
        return None
    subs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subs[0]


def _run_backend(
    *,
    repo_root: Path,
    args: list[str],
    duration_s: float,
) -> int:
    cmd = [sys.executable, "backend_main.py", *args]
    proc = subprocess.Popen(cmd, cwd=str(repo_root))
    try:
        proc.wait(timeout=float(duration_s))
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
    return int(proc.returncode or 0)


def _load_cfg(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Single-run experiment orchestrator")
    ap.add_argument("--config", required=True, help="Experiment config JSON")
    args = ap.parse_args(argv)

    cfg = _load_cfg(Path(args.config))
    repo_root = Path(__file__).resolve().parent.parent

    exp_name = str(cfg.get("experiment_name", "exp"))
    duration_s = float(cfg.get("duration_s", 60.0))
    gt_path = Path(str(cfg.get("gt_path", ""))).resolve()
    if not gt_path.exists():
        raise RuntimeError(f"Ground truth path not found: {gt_path}")

    common_args = list(cfg.get("common_args", []))
    methods: dict[str, Any] = dict(cfg.get("methods", {}))
    if not methods:
        raise RuntimeError("No methods defined in config")

    results_root = repo_root / "results" / exp_name
    results_root.mkdir(parents=True, exist_ok=True)

    all_summary: dict[str, Any] = {
        "experiment_name": exp_name,
        "duration_s": duration_s,
        "gt_path": str(gt_path),
        "methods": {},
    }

    for method_name, mcfg_any in methods.items():
        mcfg = dict(mcfg_any)
        method_dir = results_root / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        backend_args = list(common_args) + list(mcfg.get("backend_args", []))
        backend_args += ["--record", "--record-out-dir", str(method_dir)]

        print(f"[run] method={method_name}")
        t0 = time.time()
        rc = _run_backend(repo_root=repo_root, args=backend_args, duration_s=duration_s)
        elapsed = time.time() - t0

        run_dir = _latest_subdir(method_dir)
        if run_dir is None:
            all_summary["methods"][method_name] = {
                "status": "failed",
                "reason": "no_record_output",
                "return_code": rc,
                "elapsed_s": elapsed,
            }
            continue

        est_file = run_dir / str(mcfg.get("est_file", "pose_fused.jsonl"))
        if not est_file.exists():
            all_summary["methods"][method_name] = {
                "status": "failed",
                "reason": f"estimate_file_missing:{est_file.name}",
                "record_dir": str(run_dir),
                "return_code": rc,
                "elapsed_s": elapsed,
            }
            continue

        fail_trans_m = float(mcfg.get("fail_trans_m", cfg.get("fail_trans_m", 0.05)))
        fail_rot_deg = float(mcfg.get("fail_rot_deg", cfg.get("fail_rot_deg", 10.0)))
        degrade_trans_m = float(
            mcfg.get("degrade_trans_m", cfg.get("degrade_trans_m", 0.03))
        )
        degrade_rot_deg = float(
            mcfg.get("degrade_rot_deg", cfg.get("degrade_rot_deg", 5.0))
        )
        max_match_dt_s = float(
            mcfg.get("max_match_dt_s", cfg.get("max_match_dt_s", 0.05))
        )

        summary, per = evaluate(
            est_path=est_file,
            gt_path=gt_path,
            fail_trans_m=fail_trans_m,
            fail_rot_deg=fail_rot_deg,
            degrade_trans_m=degrade_trans_m,
            degrade_rot_deg=degrade_rot_deg,
            max_match_dt_s=max_match_dt_s,
        )

        (method_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        # lightweight per-frame JSONL for debugging
        with (method_dir / "per_frame.jsonl").open("w", encoding="utf-8") as f:
            for row in per:
                f.write(json.dumps(row) + "\n")

        all_summary["methods"][method_name] = {
            "status": "ok",
            "record_dir": str(run_dir),
            "estimate_file": str(est_file),
            "return_code": rc,
            "elapsed_s": elapsed,
            "metrics": summary,
        }

    (results_root / "summary_all.json").write_text(
        json.dumps(all_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
