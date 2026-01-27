# -*- coding: utf-8 -*-
"""PyTorch implementation of the reranker protocol."""

import logging
from typing import List, Dict, Any, Optional

import torch
from sentence_transformers import CrossEncoder
from .reranker import Reranker as RerankerProtocol
from .models import RerankRequest, RerankResult
from .batch_manager import BatchManager
from .batch_processor import process_batches

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"


class Reranker(RerankerProtocol):
    """PyTorch implementation of the reranker protocol using CrossEncoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_concurrent_batches: Optional[int] = None,
    ):
        """Initializes the Reranker with batching support.

        Args:
            model_name: The name of the CrossEncoder model to load.
            device: The device to run the model on ('cpu', 'cuda', 'mps').
            batch_size: Number of documents per batch (auto-detected if None).
            max_concurrent_batches: Maximum concurrent batches for processing.
        """
        self.model_name = model_name
        self.device = device or self._get_best_device()
        self.batch_manager = BatchManager(
            batch_size=batch_size, max_concurrent_batches=max_concurrent_batches
        )

        logger.info(
            f"Initializing Reranker with model '{self.model_name}' on device '{self.device}' "
            f"with batching (batch_size={self.batch_manager.batch_size})"
        )

        try:
            logger.info(
                f"Attempting basic CrossEncoder initialization for '{self.model_name}'..."
            )
            self.model = CrossEncoder(
                model_name_or_path=self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            logger.info(
                f"Successfully loaded model '{self.model_name}' to device '{self.device}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to load model '{self.model_name}': {e}", exc_info=True
            )
            raise RuntimeError(
                f"Could not load model '{self.model_name}'. Ensure it's installed or accessible."
            ) from e

    def _get_best_device(self) -> str:
        """Auto-detects the best available device."""
        if torch.cuda.is_available():
            logger.info("CUDA detected. Using GPU.")
            return "cuda"
        elif torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                logger.info("MPS detected. Using Apple Silicon GPU.")
                return "mps"
            else:
                logger.warning("MPS available but not built. Falling back to CPU.")
                return "cpu"
        else:
            logger.info("No GPU detected (CUDA or MPS). Using CPU.")
            return "cpu"

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
        sentence_pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(sentence_pairs, show_progress_bar=False)

        results = []
        for idx, score in enumerate(scores):
            result = {
                "index": idx,
                "relevance_score": float(score),
            }
            if return_documents:
                result["document"] = documents[idx]
            results.append(result)

        return results

    def rerank(self, request: RerankRequest) -> List[RerankResult]:
        """Reranks documents based on the request using batching.

        Args:
            request: The rerank request containing query, documents, and options.

        Returns:
            A list of rerank results, sorted by relevance score (descending).
        """
        return process_batches(
            request=request,
            batch_manager=self.batch_manager,
            model_predictor=self._predict_batch,
            backend_name="PyTorch",
            count_non_empty=True,
        )
