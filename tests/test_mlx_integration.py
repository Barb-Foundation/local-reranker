# -*- coding: utf-8 -*-
"""Integration tests for MLX reranker API."""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from local_reranker.api import app
from local_reranker.models import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankDocument,
)
from local_reranker.config import Settings, get_effective_model_name


class TestMLXAPIIntegration:
    """Test API integration with MLX backend."""

    @patch("local_reranker.api.MLXReranker")
    @patch("local_reranker.api.PyTorchReranker")
    @patch("local_reranker.api.settings")
    @patch("local_reranker.api.get_effective_model_name")
    def test_api_startup_with_mlx_backend(
        self, mock_get_model_name, mock_settings, mock_pytorch, mock_mlx
    ):
        """Test API startup with MLX backend."""
        # Configure mock settings to use MLX backend
        mock_settings.backend_type = "mlx"
        mock_get_model_name.return_value = "jinaai/jina-reranker-v3-mlx"

        mock_mlx_instance = Mock()
        mock_mlx.return_value = mock_mlx_instance

        # Mock the lifespan function to test MLX loading
        with TestClient(app) as client:
            # Test that MLX reranker was instantiated
            mock_mlx.assert_called_once_with(
                model_name="jinaai/jina-reranker-v3-mlx",
                disable_batching=mock_settings.disable_batching,
            )
            mock_pytorch.assert_not_called()

    @patch("local_reranker.api.MLXReranker")
    @patch("local_reranker.api.PyTorchReranker")
    @patch("local_reranker.api.settings")
    @patch("local_reranker.api.get_effective_model_name")
    def test_api_rerank_endpoint_with_mlx(
        self, mock_get_model_name, mock_settings, mock_pytorch, mock_mlx
    ):
        """Test rerank endpoint with MLX backend."""
        # Configure mock settings to use MLX backend
        mock_settings.backend_type = "mlx"
        mock_get_model_name.return_value = "jinaai/jina-reranker-v3-mlx"

        mock_mlx_instance = Mock()
        mock_mlx.return_value = mock_mlx_instance

        # Mock MLX reranker results
        mock_mlx_results = [
            RerankResult(
                document=RerankDocument(text="doc1"), relevance_score=0.9, index=0
            ),
            RerankResult(
                document=RerankDocument(text="doc2"), relevance_score=0.7, index=1
            ),
        ]

        mock_mlx_instance.rerank.return_value = mock_mlx_results

        with TestClient(app) as client:
            response = client.post(
                "/v1/rerank",
                json={
                    "model": "jinaai/jina-reranker-v3-mlx",
                    "query": "test query",
                    "documents": ["doc1", "doc2"],
                    "top_n": 2,
                    "return_documents": True,
                },
            )

            assert response.status_code == 200

            response_data = response.json()
            assert "results" in response_data
            assert len(response_data["results"]) == 2

            # Check first result
            result1 = response_data["results"][0]
            assert result1["index"] == 0
            assert result1["relevance_score"] == 0.9
            assert result1["document"]["text"] == "doc1"

            # Check second result
            result2 = response_data["results"][1]
            assert result2["index"] == 1
            assert result2["relevance_score"] == 0.7
            assert result2["document"]["text"] == "doc2"

    @patch("local_reranker.api.MLXReranker")
    @patch("local_reranker.api.PyTorchReranker")
    def test_api_rerank_endpoint_without_documents(self, mock_pytorch, mock_mlx):
        """Test rerank endpoint with empty documents."""
        mock_mlx_instance = Mock()
        mock_mlx.return_value = mock_mlx_instance
        mock_mlx_instance.rerank.return_value = []

        with TestClient(app) as client:
            response = client.post(
                "/v1/rerank",
                json={"query": "test query", "documents": [], "return_documents": True},
            )

            assert response.status_code == 200
            response_data = response.json()
            assert response_data["results"] == []

    @patch("local_reranker.api.MLXReranker")
    @patch("local_reranker.api.PyTorchReranker")
    def test_api_health_check(self, mock_pytorch, mock_mlx):
        """Test health check endpoint."""
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}

    @patch("local_reranker.api.MLXReranker")
    @patch("local_reranker.api.PyTorchReranker")
    @patch("local_reranker.api.settings")
    @patch("local_reranker.api.get_effective_model_name")
    def test_api_mlx_model_loading_failure(
        self, mock_get_model_name, mock_settings, mock_pytorch, mock_mlx
    ):
        """Test API behavior when MLX model loading fails."""
        # Configure mock settings to use MLX backend
        mock_settings.backend_type = "mlx"
        mock_get_model_name.return_value = "jinaai/jina-reranker-v3-mlx"
        mock_mlx.side_effect = RuntimeError("MLX model loading failed")

        with TestClient(app) as client:
            response = client.post(
                "/v1/rerank",
                json={"query": "test query", "documents": ["doc1", "doc2"]},
            )

            # Should return 503 when model is not loaded
            assert response.status_code == 503
            assert "Service Unavailable" in response.json()["detail"]
