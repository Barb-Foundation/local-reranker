"""Unit tests for the internal MLX cross-encoder reranker."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import mlx.core as mx
import pytest

from local_reranker.mlx_cross_encoder import MLXCrossEncoderReranker


class FakeTokenizer:
    """Minimal tokenizer that assigns one token per character."""

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}
        self._next_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens: list[int] = []
        for char in text:
            if char not in self._vocab:
                self._vocab[char] = self._next_id
                self._next_id += 1
            tokens.append(self._vocab[char])
        return tokens


class FakeLanguageModel:
    """Produces cumulative hidden states to mimic contextualization."""

    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None) -> mx.array:
        tokens = inputs.astype(mx.float32)
        cumulative = mx.cumsum(tokens, axis=1)
        expanded = mx.expand_dims(cumulative, axis=-1)
        scales = mx.reshape(
            mx.arange(1, self.hidden_size + 1, dtype=mx.float32),
            (1, 1, self.hidden_size),
        )
        return expanded * scales


class FakeModel:
    """Wraps the language model to match mlx_lm.Model expectations."""

    def __init__(self, hidden_size: int) -> None:
        self.model = FakeLanguageModel(hidden_size)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None) -> mx.array:
        return self.model(inputs, cache, input_embeddings)


@pytest.fixture(name="patched_loader")
def fixture_patched_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> Tuple[FakeModel, FakeTokenizer]:
    """Patch mlx_lm.load to return lightweight fakes."""

    fake_model = FakeModel(hidden_size=4)
    fake_tokenizer = FakeTokenizer()
    config: Dict[str, Any] = {"hidden_size": 4}

    def _mock_load(path: str, return_config: bool = False, **_) -> Any:
        if return_config:
            return fake_model, fake_tokenizer, config
        return fake_model, fake_tokenizer

    monkeypatch.setattr("local_reranker.mlx_cross_encoder.load", _mock_load)
    return fake_model, fake_tokenizer


def test_prompt_layout_contains_markers(patched_loader) -> None:
    reranker = MLXCrossEncoderReranker(model_path="mock-model")
    layout = reranker._build_prompt_tokens("query", ["doc a", "doc b"])  # type: ignore[attr-defined]

    assert layout.query_span.length > 0
    assert len(layout.document_spans) == 2
    assert layout.tokens  # ensure tokens were generated


def test_rerank_returns_embeddings_and_sorted_scores(patched_loader) -> None:
    reranker = MLXCrossEncoderReranker(model_path="mock-model")
    documents = ["alpha document", "beta document", "gamma"]

    results = reranker.rerank(
        query="test query",
        documents=documents,
        top_n=2,
        return_embeddings=True,
    )

    assert len(results) == 2
    assert all("embedding" in item for item in results)
    assert all(len(item["embedding"]) == reranker.embedding_dim for item in results)
    assert all("document" in item and item["document"] in documents for item in results)

    scores = [item["relevance_score"] for item in results]
    assert scores == sorted(scores, reverse=True)
