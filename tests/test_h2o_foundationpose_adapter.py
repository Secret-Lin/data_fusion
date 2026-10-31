from __future__ import annotations

from pathlib import Path
import zipfile

import cv2
import numpy as np
import trimesh

from adapters.h2o_foundationpose import H2OModelRepository
from experiments.h2o.dataset import H2OSequence


def _write_pose(path: Path, matrix: np.ndarray, prefix: int | None = None) -> None:
    values = matrix.reshape(-1)
    if prefix is not None:
        values = np.concatenate(([prefix], values))
    np.savetxt(path, values)


def test_h2o_mesh_zip_and_metric_scale(tmp_path: Path) -> None:
    source = tmp_path / "object.zip"
    mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
    obj_dir = tmp_path / "unpacked" / "object" / "book"
    obj_dir.mkdir(parents=True)
    mesh.export(obj_dir / "book.obj")
    with zipfile.ZipFile(source, "w") as archive:
        archive.write(obj_dir / "book.obj", "object/book/book.obj")

    repository = H2OModelRepository(source, extraction_root=tmp_path / "cache")
    assert repository.names() == ["book"]
    model, loaded = repository.load_mesh("BOOK")
    assert model.scale_to_m == 1.0
    np.testing.assert_allclose(loaded.extents, [0.1, 0.2, 0.3], atol=1e-6)


def test_h2o_rgbd_adapter_preserves_native_shape_and_depth_units(tmp_path: Path) -> None:
    sequence = tmp_path / "subject1" / "h1" / "0"
    camera = sequence / "cam0"
    for name in ("cam_pose", "obj_pose_rt", "rgb", "depth"):
        (camera / name).mkdir(parents=True)
    K = np.eye(4)
    np.savetxt(camera / "cam_pose" / "000000.txt", K.reshape(-1))
    np.savetxt(camera / "obj_pose_rt" / "000000.txt", np.r_[1, K.reshape(-1)])
    np.savetxt(camera / "cam_intrinsics.txt", [639, 640, 636, 367, 1280, 720])
    cv2.imwrite(str(camera / "rgb" / "000000.png"), np.zeros((720, 1280, 3), np.uint8))
    cv2.imwrite(str(camera / "depth" / "000000.png"), np.full((720, 1280), 1500, np.uint16))

    reader = H2OSequence(sequence, ["cam0"])
    frame = reader.load_rgbd("cam0", 0)
    assert frame.color_bgr.shape == (720, 1280, 3)
    assert frame.depth_m.dtype == np.float32
    np.testing.assert_allclose(frame.depth_m[0, 0], 1.5, atol=1e-6)
    np.testing.assert_allclose(frame.K, [[639, 0, 636], [0, 640, 367], [0, 0, 1]])
    frame.K[0, 0] = 0
    assert reader.camera_intrinsics("cam0")[0, 0] == 639
