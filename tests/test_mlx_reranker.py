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

    @patch("huggingface_hub.snapshot_download")
    def test_initialization_runtime_error(self, mock_snapshot_download):
        """Test handling of runtime errors during initialization."""
        mock_snapshot_download.side_effect = Exception("Download failed")

        with pytest.raises(RuntimeError, match="Failed to load MLX model"):
            MLXReranker()

    def test_load_mlx_reranker_uses_fallback_when_file_missing(self):
        reranker = MLXReranker.__new__(MLXReranker)
        fallback_instance = Mock()

        with (
            patch("os.path.exists", return_value=False),
            patch(
                "local_reranker.reranker_mlx.MLXCrossEncoderReranker",
                return_value=fallback_instance,
            ) as mock_fallback,
        ):
            loaded = reranker._load_mlx_reranker("/tmp/model")

        mock_fallback.assert_called_once()
        assert loaded is fallback_instance

    def test_load_mlx_reranker_falls_back_on_repo_error(self):
        reranker = MLXReranker.__new__(MLXReranker)
        fallback_instance = Mock()

        with (
            patch("os.path.exists", return_value=True),
            patch.object(
                MLXReranker,
                "_load_repo_reranker",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "local_reranker.reranker_mlx.MLXCrossEncoderReranker",
                return_value=fallback_instance,
            ) as mock_fallback,
        ):
            loaded = reranker._load_mlx_reranker("/tmp/model")

        mock_fallback.assert_called_once()
        assert loaded is fallback_instance

    def test_load_mlx_reranker_prefers_repo_impl(self):
        reranker = MLXReranker.__new__(MLXReranker)
        repo_instance = Mock()

        with (
            patch("os.path.exists", return_value=True),
            patch.object(
                MLXReranker, "_load_repo_reranker", return_value=repo_instance
            ) as mock_loader,
            patch(
                "local_reranker.reranker_mlx.MLXCrossEncoderReranker",
            ) as mock_fallback,
        ):
            loaded = reranker._load_mlx_reranker("/tmp/model")

        mock_loader.assert_called_once()
        mock_fallback.assert_not_called()
        assert loaded is repo_instance
