"""Test the vendor numeric helper without importing its CUDA/model stack."""
import ast
from pathlib import Path

import numpy as np
import pytest


def diameter_function():
    path = Path(__file__).resolve().parents[1] / "thirdparty/FoundationPose/Utils.py"
    if not path.is_file():
        pytest.skip("Optional FoundationPose source is not installed")
    source = ast.parse(path.read_text(encoding="utf-8"))
    definition = next(node for node in source.body
                      if isinstance(node, ast.FunctionDef) and node.name == "compute_mesh_diameter")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["compute_mesh_diameter"]


@pytest.mark.parametrize("count,samples", [(1, None), (256, None), (257, None), (777, 513)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_blocked_diameter_is_exact_and_preserves_random_sampling(count, samples, dtype):
    points = np.random.default_rng(100).normal(size=(count, 3)).astype(dtype)
    fn = diameter_function()
    np.random.seed(17)
    subset = points if samples is None else points[np.random.choice(count, min(samples, count), replace=False)]
    expected = np.linalg.norm(subset[None] - subset[:, None], axis=-1).max()
    expected_next = np.random.random()
    np.random.seed(17)
    assert fn(model_pts=points, n_sample=samples) == expected
    assert np.random.random() == expected_next
