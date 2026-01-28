# -*- coding: utf-8 -*-
"""Tests for MLX reranker implementation."""

import pytest
from unittest.mock import Mock, patch

from local_reranker.reranker import Reranker as RerankerProtocol
from local_reranker.models import RerankRequest

# Skip MLX tests if MLX is not installed
mlx_available = True
try:
    import mlx.core
    import mlx.nn
    import mlx_lm
    import numpy
    import safetensors
except ImportError:
    mlx_available = False

if mlx_available:
    from local_reranker.reranker_mlx import Reranker as MLXReranker
else:  # pragma: no cover
    MLXReranker = Mock  # type: ignore[assignment]


@pytest.mark.skipif(not mlx_available, reason="MLX dependencies not installed")
class TestMLXReranker:
    """Test MLX implementation of reranker protocol."""

    def test_mlx_reranker_implements_protocol(self):
        """Test that MLX reranker implements the protocol."""
        # This is a simple protocol compliance test without complex mocking
        assert hasattr(MLXReranker, "__annotations__")
        assert hasattr(MLXReranker, "rerank")
        assert hasattr(MLXReranker, "__init__")

    @patch("local_reranker.reranker_mlx.JinaMLXReranker")
    def test_initialization_runtime_error(self, mock_jina_reranker):
        """Test handling of runtime errors during initialization."""
        mock_jina_reranker.side_effect = RuntimeError("Failed to load model")

        with pytest.raises(RuntimeError, match="Failed to load MLX model"):
            MLXReranker()

    def test_load_mlx_reranker_uses_jina_mlx_reranker(self):
        """Test that _load_mlx_reranker directly instantiates JinaMLXReranker."""
        reranker = MLXReranker.__new__(MLXReranker)
        jina_reranker_instance = Mock()

        with patch(
            "local_reranker.reranker_mlx.JinaMLXReranker",
            return_value=jina_reranker_instance,
        ) as mock_jina:
            loaded = reranker._load_mlx_reranker("/tmp/model")

        mock_jina.assert_called_once_with(
            model_path="/tmp/model",
            projector_path="/tmp/model/projector.safetensors",
        )
        assert loaded is jina_reranker_instance

    def test_load_mlx_reranker_raises_runtime_error_on_failure(self):
        """Test that _load_mlx_reranker raises RuntimeError on failure."""
        reranker = MLXReranker.__new__(MLXReranker)

        with patch(
            "local_reranker.reranker_mlx.JinaMLXReranker",
            side_effect=RuntimeError("Failed to load model"),
        ):
            with pytest.raises(RuntimeError):
                reranker._load_mlx_reranker("/tmp/model")
