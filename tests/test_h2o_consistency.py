from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.h2o.consistency import analyze_take, compare_poses, main
from experiments.h2o.geometry import nearest_rotation, rotation_distance_deg
from experiments.h2o.smoke_fixture import create_smoke_sequence


def test_rotation_distance_removes_rounding_self_error() -> None:
    rounded = np.diag([0.9999998, 1.0000001, 0.9999997])
    assert rotation_distance_deg(rounded, rounded) < 1e-10
    assert rotation_distance_deg(rounded, np.eye(3)) < 1e-10
    angle = np.deg2rad(27.0)
    rotated = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    assert rotation_distance_deg(rotated @ rounded, rounded) == pytest.approx(27.0)
    assert rotation_distance_deg(np.diag([-1, -1, 1]), np.eye(3)) == pytest.approx(180.0)
    with pytest.raises(ValueError, match="reflected"):
        nearest_rotation(np.diag([-1, 1, 1]))


def test_pairwise_translation_not_temporal_motion(tmp_path: Path) -> None:
    take = create_smoke_sequence(tmp_path / "subject" / "h1" / "0", frames=8)
    report, pairs, poses = analyze_take(take, fps=30)
    assert report["n_annotated_frames"] == 8
    assert report["n_world_poses"] == 40
    assert len(pairs) == 80
    assert report["all_pairs"]["translation_mm_max"] < 1e-8
    assert report["all_pairs"]["rotation_deg_max"] < 1e-8
    assert report["identity_mismatches"] == []
    first, last = np.array(poses[0]["pose_world"]), np.array(poses[-1]["pose_world"])
    assert np.linalg.norm(first[:3, 3] - last[:3, 3]) > 1e-4


def test_comparison_retains_real_disagreement() -> None:
    a, b = np.eye(4), np.eye(4)
    b[:3, 3] = [0.003, 0.004, 0]
    assert compare_poses(a, b)["translation_mm"] == pytest.approx(5)


def test_missing_camera_frame_does_not_discard_other_views(tmp_path: Path) -> None:
    take = create_smoke_sequence(tmp_path / "dataset", frames=3)
    # Delete only a test-owned fixture file, not any user data.
    (take / "cam0" / "obj_pose_RT" / "000001.txt").unlink()
    report, pairs, poses = analyze_take(take, fps=30)
    assert report["n_annotated_frames"] == 3
    assert report["n_world_poses"] == 14
    assert len(pairs) == 26
    assert report["per_camera"]["cam0"]["n_missing_in_selected_frames"] == 1


def test_consistency_cli_outputs_and_refuses_overwrite(tmp_path: Path) -> None:
    take = create_smoke_sequence(tmp_path / "subject" / "h1" / "0", frames=3)
    output = tmp_path / "results_0"
    argv = ["--subject-root", str(take.parents[1]), "--output", str(output)]
    main(argv)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_takes"] == 1
    assert summary["all_pairs"]["n"] == 30
    assert summary["artificial_noise"] is False
    assert (output / "takes" / "h1_0" / "world_annotations.jsonl.gz").is_file()
    with pytest.raises(ValueError, match="empty"):
        main(argv)
