# -*- coding: utf-8 -*-
"""Jina MLX reranker implementation using MLP projector."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import torch
from mlx_lm import load

from .jina_mlp_projector import JinaMLPProjector, _load_projector
from .jina_prompt_formatter import _format_jina_prompt

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

    def _compute_single_batch(
        self, query: str, documents: list[str]
    ) -> tuple[mx.array, mx.array, list[float]]:
        """Compute embeddings and cosine similarity for a single batch.

        Args:
            query: The search query.
            documents: List of documents to rerank.

        Returns:
            Tuple of (query_embeds, doc_embeds, scores) where:
            - query_embeds: Projected query embedding tensor
            - doc_embeds: Projected document embedding tensors
            - scores: Cosine similarity scores between query and each document

        Raises:
            ValueError: If special tokens are not found in tokenized prompt.
        """
        prompt = _format_jina_prompt(query, documents)

        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens_array = mx.array([tokens], dtype=mx.int32)

        hidden_states = self.model(tokens_array)
        mx.eval(hidden_states)

        hidden_states = hidden_states[0]

        doc_positions = [
            i for i, t in enumerate(tokens) if t == self.doc_embed_token_id
        ]
        query_position = None
        for i, t in enumerate(tokens):
            if t == self.query_embed_token_id:
                query_position = i
                break

        if query_position is None:
            msg = "Query embed token (151671) not found in tokenized prompt"
            raise ValueError(msg)

        if len(doc_positions) != len(documents):
            msg = f"Expected {len(documents)} document embed tokens (151670), found {len(doc_positions)}"
            raise ValueError(msg)

        query_embedding = hidden_states[query_position]
        query_embeds_pt = self.projector(torch.from_numpy(query_embedding))

        doc_embeds_list_pt = []
        for pos in doc_positions:
            doc_embedding = hidden_states[pos]
            projected = self.projector(torch.from_numpy(doc_embedding))
            doc_embeds_list_pt.append(projected)

        query_embeds = mx.array(query_embeds_pt.detach().numpy())
        doc_embeds_list = [mx.array(p.detach().numpy()) for p in doc_embeds_list_pt]
        doc_embeds = mx.stack(doc_embeds_list)

        query_norm = mx.sqrt(mx.sum(query_embeds * query_embeds) + 1e-12)
        normalized_query = query_embeds / query_norm

        doc_norms = mx.sqrt(mx.sum(doc_embeds * doc_embeds, axis=1) + 1e-12)
        normalized_docs = doc_embeds / doc_norms[:, None]

        cos_sims = mx.sum(normalized_docs * normalized_query, axis=1)
        mx.eval(cos_sims)

        scores = list(cast(list[float], cos_sims.tolist()))

        return query_embeds, doc_embeds, scores
