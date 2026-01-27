"""Internal MLX cross-encoder reranker used as a fallback implementation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from safetensors import safe_open


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarkerSpan:
    """Represents the token span of a special marker within the prompt."""

    start: int
    length: int


@dataclass(frozen=True)
class PromptLayout:
    """Holds the prompt tokens and the spans needed for embedding extraction."""

    tokens: List[int]
    query_span: MarkerSpan
    document_spans: List[MarkerSpan]


@dataclass(frozen=True)
class MarkerStrings:
    """String templates used to delimit query and document regions."""

    query_start: str = "<|query|>"
    query_end: str = "<|/query|>"
    document_start: str = "<|doc|>"
    document_end: str = "<|/doc|>"
    separator: str = "\n\n"


class SimpleProjector(nn.Module):
    """Two-layer projector that maps hidden states into embedding space."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def __call__(
        self, x: mx.array
    ) -> mx.array:  # pragma: no cover - exercised indirectly
        return self.linear_out(self.activation(self.linear_in(x)))

    def load_from_tensors(self, tensors: Dict[str, mx.array]) -> bool:
        """Load projector weights from a tensor dictionary."""

        loaded_any = False
        for prefix, layer in (
            ("linear_in", self.linear_in),
            ("linear_out", self.linear_out),
        ):
            weight_key = f"{prefix}.weight"
            bias_key = f"{prefix}.bias"

            if weight_key in tensors:
                tensor = tensors[weight_key]
                if tensor.shape == layer.weight.shape:
                    layer.weight = tensor
                    loaded_any = True
                else:
                    logger.warning(
                        "[MLX] Projector weight shape mismatch for %s: expected %s, got %s",
                        weight_key,
                        layer.weight.shape,
                        tensor.shape,
                    )

            if bias_key in tensors:
                tensor = tensors[bias_key]
                if tensor.shape == layer.bias.shape:
                    layer.bias = tensor
                    loaded_any = True
                else:
                    logger.warning(
                        "[MLX] Projector bias shape mismatch for %s: expected %s, got %s",
                        bias_key,
                        layer.bias.shape,
                        tensor.shape,
                    )

        return loaded_any


class MLXCrossEncoderReranker:
    """Generic MLX reranker that converts base LMs into cross-encoders."""

    def __init__(
        self,
        model_path: str,
        projector_path: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        markers: Optional[MarkerStrings] = None,
    ) -> None:
        self.model_path = model_path
        self.projector_path = Path(projector_path) if projector_path else None
        self.marker_strings = markers or MarkerStrings()

        load_result = load(model_path, return_config=True)
        model, tokenizer, config = cast(
            Tuple[nn.Module, Any, Dict[str, object]],
            load_result,
        )
        self.model = model
        self.tokenizer: Any = tokenizer
        self.model_config: Dict[str, object] = config
        self.hidden_size = self._infer_hidden_size(config)

        self.embedding_dim = self._resolve_embedding_dim(embedding_dim)
        self.projector: Optional[SimpleProjector] = self._initialize_projector()

        # Marker token cache for quick prompt construction
        self._marker_tokens: Dict[str, List[int]] = {
            "query_start": self._encode_text(self.marker_strings.query_start),
            "query_end": self._encode_text(self.marker_strings.query_end),
            "document_start": self._encode_text(self.marker_strings.document_start),
            "document_end": self._encode_text(self.marker_strings.document_end),
            "separator": self._encode_text(self.marker_strings.separator),
        }

        # Prefer accessing the pre-logit model when available
        base_model = getattr(self.model, "model", None)
        if callable(base_model):
            self._hidden_provider: Callable[[mx.array], mx.array] = base_model  # type: ignore[assignment]
        else:
            self._hidden_provider = self.model  # type: ignore[assignment]

        logger.info(
            "[MLX] Initialized internal cross-encoder reranker (hidden=%d, embedding=%d)",
            self.hidden_size,
            self.embedding_dim,
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> List[Dict[str, object]]:
        if not documents:
            logger.warning("[MLX] No documents provided to internal reranker")
            return []

        layout = self._build_prompt_tokens(query, documents)
        hidden_states = self._compute_hidden_states(layout.tokens)

        query_vector = self._pool_span(hidden_states, layout.query_span)
        query_embedding = self._prepare_embedding(query_vector)

        doc_embeddings = [
            self._prepare_embedding(self._pool_span(hidden_states, span))
            for span in layout.document_spans
        ]

        scores = self._compute_scores(query_embedding, doc_embeddings)
        ordering = sorted(
            range(len(documents)), key=lambda idx: scores[idx], reverse=True
        )

        if top_n is not None:
            ordering = ordering[: max(0, top_n)]

        results: List[Dict[str, object]] = []
        for idx in ordering:
            result: Dict[str, object] = {
                "index": idx,
                "relevance_score": float(scores[idx]),
            }

            if return_embeddings:
                result["document"] = documents[idx]
                result["embedding"] = doc_embeddings[idx].tolist()

            results.append(result)

        return results

    def _build_prompt_tokens(
        self, query: str, documents: Sequence[str]
    ) -> PromptLayout:
        tokens: List[int] = []
        document_spans: List[MarkerSpan] = []

        def extend(sequence: List[int]) -> MarkerSpan:
            start = len(tokens)
            tokens.extend(sequence)
            return MarkerSpan(start=start, length=len(sequence))

        def extend_text(text: str) -> None:
            if not text:
                return
            tokens.extend(self._encode_text(text))

        # Query section
        extend(self._marker_tokens["query_start"])
        extend_text(query)
        query_span = extend(self._marker_tokens["query_end"])
        extend(self._marker_tokens["separator"])

        # Documents
        for doc in documents:
            extend(self._marker_tokens["document_start"])
            extend_text(doc)
            doc_span = extend(self._marker_tokens["document_end"])
            document_spans.append(doc_span)
            extend(self._marker_tokens["separator"])

        return PromptLayout(
            tokens=tokens, query_span=query_span, document_spans=document_spans
        )

    def _compute_hidden_states(self, token_ids: List[int]) -> mx.array:
        token_array = mx.array([token_ids], dtype=mx.int32)
        hidden = self._hidden_provider(token_array)

        # Some providers return logits; fall back to pre-logit model if needed
        if (
            hidden.shape[-1] != self.hidden_size
            and hasattr(self.model, "model")
            and callable(getattr(self.model, "model"))
        ):
            logger.debug(
                "[MLX] Hidden provider yielded %s, retrying with base model",
                hidden.shape,
            )
            base_model = getattr(self.model, "model")
            hidden = base_model(token_array)

        mx.eval(hidden)
        return hidden[0]

    def _pool_span(self, hidden_states: mx.array, span: MarkerSpan) -> mx.array:
        start = span.start
        length = max(1, span.length)
        slice_ = hidden_states[start : start + length]
        if length == 1:
            return slice_[0]
        return mx.mean(slice_, axis=0)

    def _prepare_embedding(self, vector: mx.array) -> mx.array:
        projected = self.projector(vector) if self.projector else vector
        norm = mx.sqrt(mx.sum(projected * projected) + 1e-12)
        return projected / norm

    def _compute_scores(
        self, query_embedding: mx.array, doc_embeddings: List[mx.array]
    ) -> List[float]:
        if not doc_embeddings:
            return []
        stacked = mx.stack(doc_embeddings)
        query = mx.expand_dims(query_embedding, axis=1)
        sims = mx.matmul(stacked, query).reshape((-1,))
        mx.eval(sims)
        return cast(List[float], sims.tolist())

    def _encode_text(self, text: str) -> List[int]:
        if not text:
            return []
        return self.tokenizer.encode(str(text), add_special_tokens=False)

    def _infer_hidden_size(self, config: Dict[str, object]) -> int:
        for key in ("hidden_size", "n_embd", "n_embed", "d_model", "model_dim"):
            value = config.get(key)
            if isinstance(value, int):
                return value
        raise ValueError("Unable to determine hidden size from MLX config")

    def _resolve_embedding_dim(self, requested: Optional[int]) -> int:
        if requested is not None:
            return requested
        env_value = os.getenv("MLX_PROJECTOR_DIM")
        if env_value:
            try:
                return int(env_value)
            except ValueError:
                logger.warning("[MLX] Invalid MLX_PROJECTOR_DIM value '%s'", env_value)
        config_value = self.model_config.get("mlx_projector_dim")
        if isinstance(config_value, int) and config_value > 0:
            return config_value
        return 512

    def _initialize_projector(self) -> Optional[SimpleProjector]:
        if self.projector_path and self.projector_path.exists():
            projector = SimpleProjector(
                self.hidden_size, self.hidden_size, self.embedding_dim
            )
            tensors = self._read_projector_weights()
            if tensors and projector.load_from_tensors(tensors):
                logger.info(
                    "[MLX] Loaded projector weights from %s", self.projector_path
                )
            else:
                logger.warning(
                    "[MLX] Projector weights missing or incompatible at %s; using random init",
                    self.projector_path,
                )
            return projector

        logger.info(
            "[MLX] projector.safetensors not found for %s; using raw hidden states as embeddings",
            self.model_path,
        )
        self.embedding_dim = self.hidden_size
        return None

    def _read_projector_weights(self) -> Optional[Dict[str, mx.array]]:
        if not self.projector_path:
            return None
        try:
            tensors: Dict[str, mx.array] = {}
            with safe_open(self.projector_path, framework="np") as handle:
                for key in handle.keys():
                    tensors[key] = mx.array(handle.get_tensor(key))
            return tensors
        except FileNotFoundError:
            logger.debug("[MLX] No projector file at %s", self.projector_path)
        except Exception as exc:
            logger.warning("[MLX] Failed to read projector weights: %s", exc)
        return None
