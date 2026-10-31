#!/usr/bin/env python3
"""Generate paper-ready plots from latest run logs.

Reads JSONL files from the latest run directory and outputs:
- score_vs_time.png
- reproj_error_vs_time.png

Usage:
  python utils/analyze_run.py --runs-root "records" --out-dir "results/figures"
  python utils/analyze_run.py --run-dir "records\20260130_164028"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, cast

import matplotlib.pyplot as plt
import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_latest_run_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"runs root not found: {root}")

    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no run directories under: {root}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def plot_series(
    df: pd.DataFrame,
    y_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
    sources: Iterable[str],
) -> None:
    plt.figure(figsize=(10, 4.5))
    for src in sources:
        s = df[df["source"] == src]
        if s.empty:
            continue
        plt.plot(s["t_host"], s[y_col], marker=".", markersize=3, linewidth=1, label=src, alpha=0.8)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot errors vs timestamp from latest run.")
    parser.add_argument("--runs-root", type=Path, default=Path("records"), help="Root dir containing run folders")
    parser.add_argument("--run-dir", type=Path, default=None, help="Specific run directory (overrides runs-root)")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"), help="Output directory for plots")
    parser.add_argument("--filter-source", type=str, default=None, help="Filter to a single source")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir(args.runs_root)
    run_dir = run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tracking = load_jsonl(run_dir / "tracking_detections.jsonl")
    if not tracking:
        raise FileNotFoundError(f"no tracking_detections.jsonl under {run_dir}")

    df = cast(pd.DataFrame, pd.DataFrame(tracking))
    if "t_host" not in df.columns:
        raise ValueError("tracking_detections.jsonl missing t_host")
    df = df.sort_values("t_host")

    if args.filter_source:
        df = df[df["source"] == args.filter_source]

    source_series = cast(pd.Series, df["source"])
    sources = source_series.dropna().unique().tolist()
    if not sources:
        raise ValueError("no source data found for plotting")

    # Score vs time
    if "score" in df.columns:
        plot_series(
            cast(pd.DataFrame, df),
            "score",
            out_dir / "score_vs_time.png",
            "DOPE Score vs Time",
            "Score",
            sources,
        )

    # Reprojection error vs time
    if "reproj_error" in df.columns:
        reproj_series = cast(pd.Series, df["reproj_error"])
        df_err = cast(pd.DataFrame, df[reproj_series.notna()].copy())
        if not df_err.empty:
            plot_series(
                cast(pd.DataFrame, df_err),
                "reproj_error",
                out_dir / "reproj_error_vs_time.png",
                "Reprojection Error vs Time",
                "Reprojection Error (px)",
                sources,
            )


if __name__ == "__main__":
    main()
