# -*- coding: utf-8 -*-
"""Jina MLX reranker implementation using MLP projector."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import mlx.nn as nn
from mlx_lm import load

from .jina_mlp_projector import JinaMLPProjector, _load_projector

logger = logging.getLogger(__name__)


class JinaMLXReranker:
    """MLX reranker using Jina MLP projector for embedding transformation.

    This class loads an MLX language model and a projector that transforms
    hidden states into embeddings for reranking tasks.

    Attributes:
        model: The loaded MLX language model.
        tokenizer: The tokenizer for the model.
        projector: The MLP projector for embedding transformation.
        doc_embed_token_id: Token ID for document embedding marker.
        query_embed_token_id: Token ID for query embedding marker.
    """

    doc_embed_token_id: int = 151670
    query_embed_token_id: int = 151671

    def __init__(
        self,
        model_path: str,
        projector_path: str,
    ) -> None:
        """Initialize the Jina MLX reranker.

        Args:
            model_path: Path to the MLX model directory.
            projector_path: Path to the projector.safetensors file.

        Raises:
            RuntimeError: If model or projector loading fails.
        """
        self.model_path = model_path
        self.projector_path = Path(projector_path)

        try:
            load_result = load(model_path, return_config=True)
            model, tokenizer, config = cast(
                tuple[nn.Module, Any, dict[str, object]],
                load_result,
            )
            self.model = model
            self.tokenizer: Any = tokenizer
            logger.info("Successfully loaded MLX model from %s", model_path)
        except Exception as e:
            msg = f"Failed to load MLX model from {model_path}: {e}"
            raise RuntimeError(msg) from e

        try:
            projector_weights = _load_projector(self.projector_path)
            self.projector = JinaMLPProjector()
            self.projector.load_state_dict(projector_weights)
            logger.info("Successfully loaded projector from %s", projector_path)
        except Exception as e:
            msg = f"Failed to load projector from {projector_path}: {e}"
            raise RuntimeError(msg) from e
