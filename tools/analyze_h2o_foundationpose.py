"""Recompute pooled metrics from an existing H2O FoundationPose run."""
import argparse
from pathlib import Path

from experiments.h2o.evaluation import allocate_results
from experiments.h2o.foundationpose_results import analyze_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    output = allocate_results(args.output_parent.resolve() if args.output_parent else root.parent)
    report = analyze_results(root, output)
    print(f"Analyzed {report['n_takes']} takes. Report: {output / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
