import asyncio
from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import trimesh

from adapters.h2o_foundationpose import H2OModelRepository
from experiments.h2o.ablation import VARIANTS, run_fusion
from experiments.h2o.evaluation import read_jsonl
from experiments.h2o.foundationpose_results import load_result_take
from tools.run_h2o_foundationpose_ablation import _run, compare_pose_outputs


def test_temporal_prediction_preserves_rigid_rotations_with_rounded_inputs() -> None:
    observations = []
    for frame in range(100):
        angle = 0.01 * frame
        pose = np.eye(4)
        pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        pose[:3, :3] *= 1 + 1e-6  # text/float32 roundoff, not experiment noise
        pose[:3, 3] = [0.003 * frame, 0.1, 0.6]
        for camera in ("cam0", "cam1"):
            observations.append({"frame_id": frame, "t_host": frame / 30,
                                 "arrival_time": frame / 30, "camera_id": camera,
                                 "source": f"h2o/{camera}/foundationpose/test",
                                 "object_name": "test", "score": 0.9, "pose_world": pose.tolist()})
    predictions, runtime = asyncio.run(run_fusion(observations, VARIANTS["full"], fps=30))
    assert len(predictions) == 100
    assert runtime["n_rejected"] == 0
    for row in predictions:
        pose = np.array(row["pose_world"])
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-9)
        assert np.linalg.norm(pose[:3, 3] - [0.003 * row["frame_id"], 0.1, 0.6]) < 0.01


def test_official_cocoa_material_alias_loads_texture_without_changing_assets(tmp_path: Path) -> None:
    folder = tmp_path / "object" / "cocoa"
    folder.mkdir(parents=True)
    obj = folder / "cocoa.obj"
    obj.write_text("mtllib ovomaltine.mtl\nv 0 0 0\nv 0.1 0 0\nv 0 0.1 0\n"
                   "vt 0 0\nvt 1 0\nvt 0 1\nusemtl material\nf 1/1 2/2 3/3\n")
    mtl = folder / "ovomaltine.mtl"
    original = "newmtl material\nKd 1 1 1\nmap_Kd cocoa.png\n"
    mtl.write_text(original)
    Image.new("RGB", (4, 4), (120, 20, 50)).save(folder / "ovomaltine.png")
    _, mesh = H2OModelRepository(tmp_path).load_mesh("cocoa")
    assert mesh.visual.material.image.size == (4, 4)
    assert mtl.read_text() == original
    assert not (folder / "cocoa.png").exists()


def _result_take(path: Path, empty: bool) -> None:
    path.mkdir(parents=True)
    gt = [{"frame_id": f, "t_host": f / 30, "object_name": "book", "pose_world": np.eye(4).tolist()} for f in range(3)]
    raw = [{**row, "camera_id": camera, "source": f"h2o/{camera}/foundationpose/book", "score": 0.9}
           for row in gt for camera in ("cam0", "cam1")] if not empty else []
    for name, rows in (("ground_truth", gt), ("tracking_before_fusion", raw),
                        ("tracking_after_fusion", gt if not empty else [])):
        (path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "summary.json").write_text(json.dumps({"cameras": ["cam0", "cam1"], "object_name": "book", "init_mode": "oracle_gt"}))


def test_batch_refusion_counts_empty_takes_and_keeps_take_boundaries(tmp_path: Path) -> None:
    root = tmp_path / "input"
    _result_take(root / "takes" / "h1_0", False)
    _result_take(root / "takes" / "h1_6", True)
    args = Namespace(run_dir=root, output=None, output_parent=tmp_path / "outputs", takes="",
                     max_frames=2, max_takes=0, fusion_config=None, variants="full,no_temporal,no_online_calibration",
                     leave_one_out=False, fps=30)
    report = asyncio.run(_run(args))
    assert args.output.name == "results_0"
    assert not report["failed_takes"]
    assert report["n_empty_takes"] == 1
    assert report["methods"]["full"]["metrics"]["n_expected"] == 4
    assert report["methods"]["full"]["metrics"]["coverage"] == 0.5
    assert report["methods"]["sensor_cam1"]["metrics"]["coverage"] == 0.5
    assert report["methods"]["no_online_calibration"]["status"] == "not_applicable"
    assert (args.output / "takes" / "h1_0" / "full" / "poses.jsonl.gz").is_file()
    assert len(read_jsonl(root / "takes" / "h1_0" / "tracking_before_fusion.jsonl")) == 6
    assert report["code_changed_during_run"] is False
    assert "output_diagnostics" in report["per_take"][0]


def test_paired_output_check_detects_order_pose_and_covariance_changes() -> None:
    from copy import deepcopy

    rows = [{"frame_id": 0, "object_name": "book", "pose_world": np.eye(4).tolist(),
             "covariance": np.eye(6).tolist(), "active_sensors": ["cam0"]}]
    assert compare_pose_outputs(rows, deepcopy(rows))["pose_and_covariance_identical"]
    changed = deepcopy(rows)
    changed[0]["covariance"][0][0] += 0.1
    assert not compare_pose_outputs(rows, changed)["pose_and_covariance_identical"]
    changed[0]["pose_world"][0][3] = 0.01
    assert compare_pose_outputs(rows, changed)["max_pose_matrix_delta"] == pytest.approx(0.01)
    changed[0]["frame_id"] = 1
    assert not compare_pose_outputs(rows, changed)["same_ordered_keys"]
    assert not compare_pose_outputs([], [])["pose_and_covariance_identical"]


@pytest.mark.parametrize("failed_method", ["full", "no_huber"])
def test_failed_variant_retains_prefix_and_does_not_drop_takes(tmp_path, monkeypatch, failed_method):
    import tools.run_h2o_foundationpose_ablation as module
    from experiments.h2o.ablation import FusionRunError

    root = tmp_path / "input"
    for name in ("take1", "take2"):
        _result_take(root / "takes" / name, False)
    original = module.run_fusion

    async def fail_one(observations, variant, **kwargs):
        if variant.name == failed_method:
            raise FusionRunError("injected solver failure", [observations[0]],
                                 {"frame_id": 1, "policy": "stop_variant"})
        return await original(observations, variant, **kwargs)

    monkeypatch.setattr(module, "run_fusion", fail_one)
    args = Namespace(run_dir=root, output=None, output_parent=tmp_path / "outputs", takes="",
                     max_frames=0, max_takes=0, fusion_config=None,
                     variants="full,no_huber,no_history,no_online_calibration", leave_one_out=False, fps=30)
    report = asyncio.run(module._run(args))
    assert not report["failed_takes"]
    assert len(report["failed_methods"]) == 2
    assert len(report["per_take"]) == 2
    metrics = report["methods"][failed_method]["metrics"]
    assert metrics["n_expected"] == 6
    assert metrics["n_missing"] == 4
    assert metrics["failure_rate_including_missing"] == pytest.approx(2 / 3)
    assert report["methods"][failed_method]["status"] == "completed_with_failures"
    assert report["methods"]["no_history"]["metrics"]["coverage"] == 1
    assert report["methods"]["no_online_calibration"]["status"] == "not_applicable"
    assert "completed_with_failures" in (args.output / "summary.md").read_text(encoding="utf-8")


def test_missing_solver_output_discards_incomplete_frame(monkeypatch):
    from experiments.h2o.ablation import FusionRunError
    from managers.fusion_manager import FusionManager

    original = FusionManager._update_and_publish
    calls = 0

    async def missing(self, *args):
        nonlocal calls
        calls += 1
        if calls == 4:  # second view of frame 1, after its provisional output
            return
        return await original(self, *args)

    monkeypatch.setattr(FusionManager, "_update_and_publish", missing)
    observations = [{"frame_id": f, "t_host": f / 30, "arrival_time": f / 30,
                     "object_name": "book", "camera_id": camera,
                     "source": f"h2o/{camera}/megapose/book", "detector": "megapose",
                     "score": 1, "pose_world": np.eye(4).tolist()}
                    for f in range(3) for camera in ("cam0", "cam1")]
    with pytest.raises(FusionRunError) as error:
        asyncio.run(run_fusion(observations, VARIANTS["full"], fps=30))
    assert [p["frame_id"] for p in error.value.predictions] == [0]
    assert error.value.failure["frame_id"] == 1
    assert calls == 4


def test_unexpected_variant_exception_is_visible_and_counts_all_targets(tmp_path, monkeypatch):
    import tools.run_h2o_foundationpose_ablation as module

    root = tmp_path / "take"
    _result_take(root, False)
    original = module.run_fusion

    async def fail(observations, variant, **kwargs):
        if variant.name == "no_huber":
            raise ValueError("injected unexpected error")
        return await original(observations, variant, **kwargs)

    monkeypatch.setattr(module, "run_fusion", fail)
    args = Namespace(run_dir=root, output=None, output_parent=tmp_path / "outputs", takes="",
                     max_frames=0, max_takes=0, fusion_config=None,
                     variants="full,no_huber,no_history", leave_one_out=False, fps=30)
    report = asyncio.run(module._run(args))
    assert report["methods"]["no_huber"]["metrics"]["n_missing"] == 3
    assert report["methods"]["no_huber"]["metrics"]["failure_rate_including_missing"] == 1
    assert report["methods"]["no_history"]["metrics"]["coverage"] == 1
    assert report["failed_methods"][0]["exception_type"] == "ValueError"
    assert "injected unexpected error" in (args.output / "no_huber.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("report,exit_code", [({}, 0), ({"failed_methods": [{"method": "full"}]}, 1),
                                              ({"failed_takes": [{"take": "test"}]}, 1)])
def test_cli_returns_nonzero_for_variant_or_take_failure(tmp_path, monkeypatch, report, exit_code):
    import tools.run_h2o_foundationpose_ablation as module

    args = Namespace(fps=30, max_frames=0, max_takes=0, variants="full", output=tmp_path)

    async def fake_run(args):
        return report

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "_run", fake_run)
    assert module.main() == exit_code


def test_quality_sidecar_is_attached_without_changing_cached_pose(tmp_path: Path) -> None:
    root = tmp_path / "input"
    _result_take(root / "takes" / "h1_0", False)
    sidecar = tmp_path / "scores.jsonl"
    sidecar.write_text(json.dumps({
        "frame_id": 0, "camera_id": "cam0", "object_name": "book",
        "depth_fit": {"status": "ok", "depth_inlier_fraction": 0.2},
    }) + "\n")
    args = Namespace(run_dir=root / "takes" / "h1_0", output=None,
                     output_parent=tmp_path / "outputs", takes="", max_frames=2,
                     max_takes=0, fusion_config=None, variants="full",
                     leave_one_out=False, fps=30, quality_sidecar=sidecar)
    report = asyncio.run(_run(args))
    assert report["quality_sidecar"] == str(sidecar.resolve())
    assert report["methods"]["full"]["status"] == "completed"


def test_gpu_observations_survive_fusion_failure(tmp_path: Path, monkeypatch) -> None:
    from tools import run_h2o_foundationpose as runner_module

    class Sequence:
        def __init__(self, *args):
            pass

        @staticmethod
        def discover_cameras(path):
            return ["cam0"]

        def common_frame_ids(self):
            return [0]

        def load(self, *args):
            return SimpleNamespace(object_id=1, object_name="book", T_world_object=np.eye(4),
                                   T_camera_object=np.eye(4), T_world_camera=np.eye(4))

        def load_rgbd(self, *args):
            return SimpleNamespace(K=np.eye(3), color_bgr=np.zeros((10, 10, 3), np.uint8),
                                   depth_m=np.ones((10, 10), np.float32))

    class Runner:
        def reset(self):
            pass

        def prepare(self, names):
            pass

        def process(self, **kwargs):
            return SimpleNamespace(object_name="book", mode="register", score=1.0, pose_cam=np.eye(4))

    class Repository:
        def load_mesh(self, name):
            return SimpleNamespace(mesh_path=tmp_path / "book.obj", scale_to_m=1), trimesh.creation.box()

    async def failed_fusion(*args, **kwargs):
        raise RuntimeError("test optimizer failure")

    monkeypatch.setattr(runner_module, "H2OSequence", Sequence)
    monkeypatch.setattr(runner_module, "_project_mesh_mask", lambda *args: np.ones((10, 10), np.uint8))
    monkeypatch.setattr(runner_module, "_run_fusion", failed_fusion)
    args = Namespace(cameras="all", max_frames=1, init_mode="oracle_gt", fps=30)
    output = tmp_path / "results_0"
    with pytest.raises(RuntimeError, match="test optimizer failure"):
        runner_module._run_take(args, tmp_path, tmp_path / "h1" / "0", output, Repository(), Runner())
    gt, observations, cameras, _, metadata = load_result_take(output)
    assert len(gt) == len(observations) == 1
    assert cameras == ["cam0"]
    assert metadata["status"] == "inference_saved_fusion_pending"
