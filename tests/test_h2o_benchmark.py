from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from experiments.h2o.benchmark import _run, build_parser
from experiments.h2o.dataset import H2OSequence
from experiments.h2o.smoke_fixture import create_smoke_sequence


@pytest.mark.parametrize(
    "cameras",
    [
        ["cam0"],
        ["cam0", "cam4"],
        ["cam0", "cam2", "cam4"],
        ["cam0", "cam1", "cam2", "cam3", "cam4"],
    ],
)
def test_h2o_benchmark_runs_arbitrary_sensor_counts(
    tmp_path: Path, cameras: list[str]
) -> None:
    sequence_dir = create_smoke_sequence(tmp_path / "dataset", frames=8)
    sequence = H2OSequence(sequence_dir, cameras)
    report = asyncio.run(
        _run(
            sequence=sequence,
            output_dir=tmp_path / "output",
            max_frames=8,
            fps=30.0,
            seed=7,
            synthetic_noise=False,
        )
    )

    assert report["sensor_count"] == len(cameras)
    assert report["frames_with_object"] == 8
    assert report["metrics"]["fusion_full"]["n"] == 8
    assert (tmp_path / "output" / "report.json").is_file()
    assert report["synthetic_noise"] is False
    for camera in cameras:
        assert report["metrics"][f"sensor_{camera}"]["n"] == 8
        assert report["metrics"][f"sensor_{camera}"]["translation_p95_m"] < 1e-9


def test_benchmark_noise_flag_defaults_off_and_cannot_enable(tmp_path: Path) -> None:
    args = build_parser().parse_args(["--sequence", "unused", "--output", "unused"])
    assert args.synthetic_noise is False
    sequence = H2OSequence(create_smoke_sequence(tmp_path / "data", frames=2), ["cam0"])
    with pytest.raises(ValueError, match="removed"):
        asyncio.run(_run(sequence=sequence, output_dir=tmp_path / "out", max_frames=2,
                         fps=30.0, seed=7, synthetic_noise=True))
    assert not (tmp_path / "out").exists()


def test_benchmark_preserves_frame_time_gaps(tmp_path: Path) -> None:
    take = create_smoke_sequence(tmp_path / "data", frames=4)
    for camera in ("cam0", "cam4"):
        (take / camera / "obj_pose_RT" / "000001.txt").unlink()
    output = tmp_path / "out"
    asyncio.run(_run(sequence=H2OSequence(take, ["cam0", "cam4"]), output_dir=output,
                     max_frames=0, fps=30.0, seed=0))
    for name in ("ground_truth", "fusion_full", "baseline_avg", "sensor_cam4"):
        rows = [json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()]
        assert [r["frame_id"] for r in rows] == [0, 2, 3]
        assert [r["t_host"] for r in rows] == pytest.approx([0, 2 / 30, 3 / 30])
