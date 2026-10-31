import pytest
import torch
from kornia.geometry.transform import warp_perspective

from adapters.bounded_image_warp import bounded_warp


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_expanded_image_warps_keep_values_and_order(mode, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(3)
    image = torch.rand(1, 3, 40, 50, device=device).expand(19, -1, -1, -1)
    matrices = torch.eye(3, device=device)[None].repeat(19, 1, 1)
    matrices[:, 0, 2] = torch.arange(19, device=device) * .2
    options = dict(dsize=(20, 24), mode=mode, align_corners=False)
    expected = warp_perspective(image, matrices, **options)
    actual = bounded_warp(warp_perspective, 4)(image, matrices, **options)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
