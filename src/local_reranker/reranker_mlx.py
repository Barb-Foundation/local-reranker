# -*- coding: utf-8 -*-
"""MLX-based reranker implementation for Apple Silicon optimization."""

from typing import List, Dict, Any, Optional
import logging
import os
import sys
import importlib.util

from .reranker import Reranker as RerankerProtocol
from .models import RerankRequest, RerankResult
from .batch_manager import BatchManager
from .batch_processor import process_batches
from .mlx_cross_encoder import MLXCrossEncoderReranker

logger = logging.getLogger(__name__)


class Reranker(RerankerProtocol):
    """MLX implementation of reranker protocol for Apple Silicon optimization."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v3-mlx",
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_concurrent_batches: Optional[int] = None,
    ):
        """Initialize MLX reranker with batching support.

        Args:
            model_name: The name of the MLX model to load.
            device: The device to run the model on. MLX auto-detects Apple Silicon GPU/CPU.
                    This parameter is kept for protocol compatibility but ignored.
            batch_size: Number of documents per batch (auto-detected if None).
            max_concurrent_batches: Maximum concurrent batches for processing.

        Raises:
            ImportError: If MLX dependencies are not installed.
            RuntimeError: If model loading fails.
        """
        self.model_name = model_name
        self.device = device

        self.batch_manager = BatchManager(
            batch_size=batch_size, max_concurrent_batches=max_concurrent_batches
        )

        try:
            model_path = self._prepare_model_files(model_name)
            self.model = self._load_mlx_reranker(model_path)
            logger.info(f"Successfully loaded MLX reranker with batching: {model_name}")

        except ImportError as e:
            raise ImportError(
                "MLX dependencies not found. Install with: pip install mlx mlx-lm safetensors"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load MLX model '{model_name}': {e}") from e

    def _prepare_model_files(self, model_name: str) -> str:
        """Prepare model files by downloading from HuggingFace if needed."""
        try:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                repo_id=model_name,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py"],
            )
            return model_path

        except ImportError:
            raise ImportError(
                "huggingface-hub not found. Install with: pip install huggingface-hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model files: {e}") from e

    def _load_mlx_reranker(self, model_path: str):
        """Load MLX reranker, falling back to the internal cross-encoder as needed."""
        rerank_file = os.path.join(model_path, "rerank.py")
        projector_path = os.path.join(model_path, "projector.safetensors")

        if os.path.exists(rerank_file):
            try:
                reranker = self._load_repo_reranker(
                    rerank_file=rerank_file,
                    model_path=model_path,
                    projector_path=projector_path,
                )
                logger.info(
                    "[MLX] Using repo-provided MLXReranker implementation from %s",
                    rerank_file,
                )
                return reranker
            except Exception as repo_error:
                logger.warning(
                    "[MLX] Failed to load repo-provided reranker at %s: %s. "
                    "Falling back to internal cross-encoder.",
                    rerank_file,
                    repo_error,
                )
        else:
            logger.info(
                "[MLX] rerank.py not found in %s. Using internal cross-encoder fallback.",
                model_path,
            )

        return MLXCrossEncoderReranker(
            model_path=model_path,
            projector_path=projector_path,
        )

    def _load_repo_reranker(
        self, rerank_file: str, model_path: str, projector_path: str
    ):
        """Load a repo-provided MLXReranker implementation from rerank.py."""
        spec = importlib.util.spec_from_file_location("mlx_reranker", rerank_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec from {rerank_file}")

        mlx_reranker_module = importlib.util.module_from_spec(spec)
        sys.modules["mlx_reranker"] = mlx_reranker_module
        spec.loader.exec_module(mlx_reranker_module)

        MLXRerankerImpl = mlx_reranker_module.MLXReranker
        return MLXRerankerImpl(
            model_path=model_path,
            projector_path=projector_path,
        )

    def _predict_batch(
        self, query: str, documents: List[str], return_documents: Optional[bool]
    ) -> List[Dict[str, Any]]:
        """Predict scores for a batch of documents.

        Args:
            query: The query text
            documents: List of document texts
            return_documents: Whether to include document content in results

        Returns:
            List of result dictionaries with keys: index, relevance_score, document (optional)
        """
        batch_results = self.model.rerank(
            query=query,
            documents=documents,
            top_n=None,
            return_embeddings=return_documents or False,
        )

        results = []
        for result in batch_results:
            result_dict = {
                "index": result["index"],
                "relevance_score": result["relevance_score"],
            }
            if return_documents and "document" in result:
                result_dict["document"] = result["document"]
            results.append(result_dict)

        return results

    def rerank(self, request: RerankRequest) -> List[RerankResult]:
        """Rerank documents using the MLX backend with batching support.

        Args:
            request: The rerank request containing query, documents, and options.

        Returns:
            A list of rerank results, sorted by relevance score (descending).

        Raises:
            ValueError: If the request is invalid.
            RuntimeError: If reranking fails.
        """
        if not request.query or not request.query.strip():
            logger.warning("[MLX] Empty query provided")
            return []

        return process_batches(
            request=request,
            batch_manager=self.batch_manager,
            model_predictor=self._predict_batch,
            backend_name="MLX",
            count_non_empty=False,
        )
