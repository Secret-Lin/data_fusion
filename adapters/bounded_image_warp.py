"""Bound independent image warps without changing resolution or pose candidates."""
from __future__ import annotations

from functools import wraps


def bounded_warp(warp, batch_size: int = 16):
    """Wrap a BCHW warp; preserve order, transforms and all warp options.

    Expanded native-resolution images can otherwise become a very large
    contiguous allocation inside grid_sample. This bounds that temporary only.
    Intended for inference, not training or cross-sample operations.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    @wraps(warp)
    def call(src, matrix, *args, **kwargs):
        import torch

        count = src.shape[0]
        if count <= batch_size:
            return warp(src, matrix, *args, **kwargs)
        if matrix.shape[0] not in (1, count):
            raise ValueError("Warp transform batch does not match images")
        return torch.cat([
            warp(src[start:start + batch_size],
                 matrix if matrix.shape[0] == 1 else matrix[start:start + batch_size],
                 *args, **kwargs)
            for start in range(0, count, batch_size)
        ], dim=0)

    return call
