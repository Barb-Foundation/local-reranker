# -*- coding: utf-8 -*-
"""Jina MLP projector module for fixed-architecture projector weights."""

from torch import Tensor, nn


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
