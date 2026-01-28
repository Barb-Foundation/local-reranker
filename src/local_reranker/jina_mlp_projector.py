# -*- coding: utf-8 -*-
"""Jina MLP projector module for fixed-architecture projector weights."""

import logging
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor, nn


logger = logging.getLogger(__name__)


def _load_projector(projector_path: Path) -> dict[str, torch.Tensor]:
    """Load projector weights from safetensors file with shape validation.

    Args:
        projector_path: Path to projector.safetensors file.

    Returns:
        Dictionary of weight tensors.

    Raises:
        FileNotFoundError: If projector file doesn't exist.
        ValueError: If weight shapes don't match expected dimensions.
    """
    if not projector_path.exists():
        msg = f"Projector file not found: {projector_path}"
        raise FileNotFoundError(msg)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(projector_path, framework="pt") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)

    expected_shapes = {
        "linear1.weight": torch.Size([512, 1024]),
        "linear2.weight": torch.Size([512, 512]),
    }

    for name, expected_shape in expected_shapes.items():
        if name not in tensors:
            msg = f"Missing weight tensor: {name}"
            raise ValueError(msg)
        actual_shape = tensors[name].shape
        if actual_shape != expected_shape:
            msg = (
                f"Invalid shape for {name}: expected {tuple(expected_shape)}, "
                f"got {tuple(actual_shape)}"
            )
            raise ValueError(msg)

    logger.debug("Loaded projector weights from %s", projector_path)
    return tensors


class JinaMLPProjector(nn.Module):
    """Fixed-architecture MLP projector for Jina projector weights.

    Architecture: 1024 → 512 → 512 with ReLU activation and no bias terms.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 512, bias=False)
        self.activation = nn.ReLU()

    def __call__(self, x: Tensor) -> Tensor:
        return self.linear2(self.activation(self.linear1(x)))
